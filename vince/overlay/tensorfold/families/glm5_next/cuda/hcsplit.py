"""TF_GLM_HC_SPLIT=1: a prompt chunk's hyper-connection glue split by rows between the ranks (decode untouched).

Unsplit, each exchange site (after the attention block and after the MLP/MoE block) all-gathers both ranks' fp32
partials [R, D], and both ranks run hc_post and the next hc_pre over all R rows. Split, rank r owns rows
[r H, r H + H) (H = R / 2; an odd chunk gets one zero pad row, which nothing reads): each rank sends the peer its
partial of the peer's rows (fp32), adds the two partials of its own rows rank 0's first (``glue.hc_post_pair``, the
same kernel as hc_post), runs hc_post and the next hc_pre on its own rows, and swaps its normed rows (bf16) for the
peer's, so both ranks hold the next block's full input. The residual streams (b.x) stay current for own rows only;
DFlash2 taps and the final stream mean (b.hidden) are computed on own rows and swapped like the normed rows. Every
glue kernel is row-independent and the partial sum order is unchanged, so every bit is the unsplit path's.

The group sums b.xs of the normed rows are not swapped: no prompt-chunk matmul reads them (4-bit projections of a
prompt chunk run ``prefill_matmul``, BF16 and FP8 ones ignore them). The tests compare split and unsplit states bit
for bit, so a later reader would show up there.

TF_GLM_PREFILL_OVERLAP=1 (with the split) runs the swaps on a second CUDA stream in row pieces: a site computes the
peer's rows of its partial first, piece by piece, each piece's swap starting as it is done, then its own rows; then
the glue piece by piece, each piece waiting only for its own partial rows and its normed rows swapped while the next
piece's glue runs. Events order the two streams; the arithmetic is the same kernels on the same rows.

TF_GLM_PREFILL_OVERLAP=2 also runs the next block's row-independent front (``glue``'s ``front``: KDA's input
projections, DSA's projections, norms and query expansion, the dense MLP's gate/up, the shared expert of the MoE) on
this rank's rows of each piece right after that piece's glue, while its normed rows (and the later pieces' partials)
are still on the wire; the block then runs its front on the peer's rows once they have arrived, and the rest as
before. Without it the main stream waited for the last piece's swap with nothing to do. The fronts are row-independent
kernels (the prompt matmul gives every row range the same bits), so every bit is still the unsplit path's.

TF_GLM_PROMPT_OWN_FRONT=1 (``own_front``): the glue before a DSA layer keeps the normed rows on their owner, whose
attention block runs its replicated front (q_a / kv_a and their norms, the indexer's projections and its top-k) on
them and shares the outputs (``share``: whole own shares, send/receive or the all-gather, on the side stream with the
overlap) instead.

TF_GLM_PROMPT_MICROBATCH=1 (``microbatch``): a chunk runs as two halves, each row-split by its own HcSplit over row
views of the prompt buffers, both on one side stream (``HcSplit(stream=...)``); a half's glue runs there too
(``glue_side``), under the other half's compute, and the main stream ``join``s it before that half's next block.

TF_GLM_PROMPT_PARTIALS=bf16 (``partials``; default fp32): each rank's partial of a site goes out as bf16 instead of fp32
(half the bytes of a layer's two partial exchanges, most of what a row-split layer sends), and every rank's partial of a
row is rounded to bf16 before the sum, which stays in fp32 and rank 0 first: the owner rounds its own partial as the
others round theirs (round-to-nearest-even in torch's casts and the kernels', the same), so the ranks hold the same
bits. Unsplit prompt chunks (a prompt's tail shorter than TF_GLM_HC_SPLIT_MIN_ROWS) and the all-gather exchange round
the same way in hc_post (``glue.hc_post``'s ``round_parts``), so a row's bits do not depend on the chunking, the
micro-batches or the exchange; decode windows, verify and the MTP head keep fp32 partials. The prompt's bits change
once against fp32 partials (each partial rounded to bf16 before the sum; the sum is rounded to bf16 as before), which
has not been quality-checked on Flash: opt-in.

TF_GLM_HC_EXCHANGE: ``p2p`` (the default) swaps rows with ncclSend/ncclRecv; ``gather`` uses the NCCL all-gather
instead (whole halves, one piece: the partial halves out of place into the unused all-gather buffer, the bf16 rows in
place), the same bytes on the wire, for when NCCL's point-to-point path is slower or misbehaves on a fabric."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import torch

from . import glue, pf_xchg

EXCHANGES = ("p2p", "gather")
# TF_GLM_PF_FILL_WHOLE=1 (prefill work, 2026-11-08): a row-split site fills its whole partial in one call (``partial``)
PF_FILL_WHOLE = os.environ.get("TF_GLM_PF_FILL_WHOLE", "0").strip() == "1"
# TF_GLM_PF_PART16=1 (2026-11-10, same bits, start time; with bf16 partials on the RoCE ring exchange past two ranks):
# a site's fills write this rank's bf16 partial themselves (``b.part16`` = send16, by row; forward's fills), instead
# of fp32 into b.part and a cast pass a peer and piece on the side stream; own rows are copied bf16 to bf16 in _post
PF_PART16 = os.environ.get("TF_GLM_PF_PART16", "0").strip() == "1"
# TF_GLM_PF_OWN_ROUTER=1 / pf switch own_router (2026-11-10, same bits): before a MoE block of a halved chunk the glue
# also runs the MoE router and its top-k on this rank's rows (row-independent kernels, the same ones on the same row
# tiles) and the picks and weights travel with the normed rows, so no rank routes the other ranks' rows
# (``glue_side``'s ``router``; ``routed`` tells forward.moe_block)
PF_OWN_ROUTER = os.environ.get("TF_GLM_PF_OWN_ROUTER", "0").strip() == "1"
# TF_GLM_PF_ZC / pf switch zc (2026-11-10, same bits; halved chunks with bf16 partials on the RoCE ring exchange): a
# site's hc_post reads the peers' partials in the exchange's receive slots (``glue.hc_post_zc``), with no copy launch.
# 1: on. 2: shadow check: the usual copy and hc_post make the result, the zero-copy kernel's output is only compared
# with it (pf_xchg's report prints the differing elements; every bit served is the usual path's).
PF_ZC = int(os.environ.get("TF_GLM_PF_ZC", "0").strip() or 0)
# TF_GLM_PF_XCHG_PART=1 (start time, every rank: a third exchange, ``pf_xchg.PART``, ~0.3 GiB pinned a rank) with pf
# switch si / TF_GLM_PF_SI (2026-11-10, same bits; flip only while idle: every rank must agree): a half's site fills
# its bf16 partial straight into that exchange's send slot (after christopherowen/dgx-spark-networking's
# feat/sparknet-pdl-direct-stage: the stage copy is skipped when the input is the slot), so its exchange is a doorbell,
# a wait and hc_post reading the receive slots (TF_GLM_PF_ZC's kernel). The partials get their own exchange because the
# two slots alternate by op: on the shared one a half's next fill would land in the slot the other half's exchange is
# still sending. si 2: shadow check: the usual path makes the result; the fills run a second time into the slot, sent
# over PART, and each rank compares what arrived that way with what arrived the usual way (pf_xchg's report).
PF_SI = int(os.environ.get("TF_GLM_PF_SI", "0").strip() or 0)
# TF_GLM_PF_KQ=1 / pf switch kq (2026-11-10, same bits, with TF_GLM_PF_FP8; flip only while idle): before a KDA layer of
# a halved chunk the glue quantizes this rank's normed rows (``quantize_rows``'s kernel, a row at a time: the bits the
# layer's projection would make) and ships the e4m3 bytes, group sums and row scales instead of the bf16 rows: no
# rank quantizes the others' rows, and that site's row exchange carries half the bytes (``quanted`` tells kda_front)
PF_KQ = os.environ.get("TF_GLM_PF_KQ", "0").strip() == "1"
# pf switch split_min (2026-11-10, same bits; every rank alike: flip only while idle): prompt chunks of fewer rows run
# unsplit (two all-gathers a layer instead of the row split's exchanges, which a short chunk only waits on); 0: the
# setting TF_GLM_HC_SPLIT_MIN_ROWS alone
PF_SPLIT_MIN = int(os.environ.get("TF_GLM_PF_SPLIT_MIN", "0").strip() or 0)
PARTIALS = ("fp32", "bf16")    # TF_GLM_PROMPT_PARTIALS: what a row-split prompt chunk's partials travel as
SHARES = 4               # own-row shares a layer keeps in flight at once (``HcSplit.share``: slots 0 .. SHARES - 1)


def _flag(env, name: str) -> bool:
    value = str(env.get(name, "") or "0").strip()
    if value not in ("0", "1"):
        raise ValueError(f"{name}: 0 or 1, not {value!r}")
    return value == "1"


def _int(env, name: str, default: int, lo: int, hi: int) -> int:
    value = str(env.get(name, "") or default).strip()
    if not value.isdecimal() or not lo <= int(value) <= hi:
        raise ValueError(f"{name}: a whole number from {lo} to {hi}, not {value!r}")
    return int(value)


def pad_rows(rows: int, world: int) -> int:
    """Rows a prompt buffer of ``rows`` rows holds past them for the split's pad rows: a chunk of ``rows`` rows rounded
    up to whole shares a rank (``HcSplit.padded``), so that a full chunk, and every shorter one, runs split at any
    world size. Prompt chunks are a multiple of 64 rows (TF_GLM_PREFILL_ROWS), so this is 0 wherever the world size
    divides 64 (two and four ranks: 2,048 = 2 x 1,024 = 4 x 512) and ``-rows % 3`` at three (2,048 = 3 x 683 - 1:
    one row; 2,112: none). Without that row every full chunk at three ranks failed ``applies`` and ran unsplit (an
    all-gather of every rank's whole partial at each of the 90 sites, the glue on every row, no overlap)."""

    return -int(rows) % int(world)


def buffer_rows(rows: int, world: int, settings: "SplitSettings | None") -> int:
    """Rows the prompt buffers of ``rows``-row chunks hold (``forward.Buffers``' ``rows + pad``): ``rows`` plus the
    split's pad rows when TF_GLM_HC_SPLIT is on past one rank, else ``rows``. The engine's buffers and its memory
    estimate both take it, so they agree; at two and four ranks it is ``rows`` (the same buffers and estimate)."""

    if settings is None or not settings.split or int(world) < 2:
        return int(rows)
    return int(rows) + pad_rows(rows, world)


@dataclass(frozen=True)
class SplitSettings:
    """What TF_GLM_HC_SPLIT / TF_GLM_PREFILL_OVERLAP ask for; both ranks must agree (the startup comparison)."""

    split: bool = False
    overlap: bool = False
    pieces: int = 2               # TF_GLM_OVERLAP_PIECES: row pieces a site's swaps run in (overlap, p2p only)
    min_rows: int = 64            # TF_GLM_HC_SPLIT_MIN_ROWS: shorter chunks (a prompt's tail) run unsplit
    exchange: str = "p2p"         # TF_GLM_HC_EXCHANGE: p2p (ncclSend/ncclRecv) or gather (the all-gather)
    fronts: bool = False          # TF_GLM_PREFILL_OVERLAP=2: the next block's front on own rows during the swaps
    own_front: bool = False       # TF_GLM_PROMPT_OWN_FRONT=1: DSA's replicated front on own rows (``own_front``)
    microbatch: bool = False      # TF_GLM_PROMPT_MICROBATCH=1: a chunk in two halves (``microbatch``)
    micro_min: int = 2048         # TF_GLM_PROMPT_MICROBATCH_MIN_ROWS: halves of fewer rows run the chunk whole
    partials: str = "fp32"        # TF_GLM_PROMPT_PARTIALS: fp32, or bf16 (rounded before the sum, half the bytes)

    def __post_init__(self) -> None:
        if self.fronts and not self.overlap:
            raise ValueError("the blocks' fronts run during the overlapped swaps: they need the overlap")
        if self.own_front and not self.split:
            raise ValueError("TF_GLM_PROMPT_OWN_FRONT=1 runs on the row split's own rows: it needs TF_GLM_HC_SPLIT=1")
        if self.microbatch and not self.split:
            raise ValueError("TF_GLM_PROMPT_MICROBATCH=1 row-splits each half: it needs TF_GLM_HC_SPLIT=1")
        if self.partials not in PARTIALS:
            raise ValueError(f"TF_GLM_PROMPT_PARTIALS: {' or '.join(PARTIALS)}, not {self.partials!r}")
        if self.partials != "fp32" and not self.split:
            raise ValueError("TF_GLM_PROMPT_PARTIALS=bf16 sends the row split's partials as bf16: it needs "
                             "TF_GLM_HC_SPLIT=1")

    @classmethod
    def from_env(cls, env=None) -> "SplitSettings":
        env = os.environ if env is None else env
        split = _flag(env, "TF_GLM_HC_SPLIT")
        level = str(env.get("TF_GLM_PREFILL_OVERLAP", "") or "0").strip()
        if level not in ("0", "1", "2"):
            raise ValueError(f"TF_GLM_PREFILL_OVERLAP: 0, 1 or 2, not {level!r}")
        overlap = level != "0"
        if overlap and not split:
            raise ValueError(f"TF_GLM_PREFILL_OVERLAP={level} overlaps the row-split exchanges: it needs "
                             "TF_GLM_HC_SPLIT=1")
        exchange = str(env.get("TF_GLM_HC_EXCHANGE", "") or "p2p").strip()
        if exchange not in EXCHANGES:
            raise ValueError(f"TF_GLM_HC_EXCHANGE: {' or '.join(EXCHANGES)}, not {exchange!r}")
        partials = str(env.get("TF_GLM_PROMPT_PARTIALS", "") or "fp32").strip()
        return cls(split, overlap, _int(env, "TF_GLM_OVERLAP_PIECES", 2, 1, 8),
                   _int(env, "TF_GLM_HC_SPLIT_MIN_ROWS", 64, 2, 16384), exchange, fronts=level == "2",
                   own_front=_flag(env, "TF_GLM_PROMPT_OWN_FRONT"),
                   microbatch=_flag(env, "TF_GLM_PROMPT_MICROBATCH"),
                   micro_min=_int(env, "TF_GLM_PROMPT_MICROBATCH_MIN_ROWS", 2048, 64, 16384), partials=partials)

    def piece_count(self) -> int:
        return self.pieces if self.overlap and self.exchange == "p2p" else 1

    def code(self) -> list[int]:
        if not self.split:
            return [0, 0, 0, 0, 0, 0, 0, 0]
        return [1, int(self.overlap) + int(self.fronts), self.piece_count(), self.min_rows,
                EXCHANGES.index(self.exchange), int(self.own_front), self.micro_min if self.microbatch else 0,
                PARTIALS.index(self.partials)]

    def describe(self) -> str:
        what = "rows split between the ranks"
        what += ", exchanges by " + ("send/receive" if self.exchange == "p2p" else "all-gather")
        if self.overlap:
            n = self.piece_count()
            what += f" on a second stream{f' in {n} pieces' if n > 1 else ''}"
            if self.fronts:
                what += ", the next block's front on own rows meanwhile"
        if self.own_front:
            what += "; DSA's replicated front (q_a / kv_a, the indexer, its top-k) on own rows, its outputs shared"
        if self.microbatch:
            what += f"; chunks of {2 * self.micro_min}+ rows in two halves, one's exchanges under the other's compute"
        if self.partials == "bf16":
            what += "; partials rounded to bf16 before the sum and sent as bf16 (TF_GLM_PROMPT_PARTIALS)"
        return f"prompt chunks' hyper-connections: {what} (chunks of {self.min_rows}+ rows)"


class HcSplit:
    """One prompt buffer's row split: rows, pieces, the second stream and its events.

    Past two ranks (``world`` N): rank r owns rows [r H, r H + H), H = ceil(R / N) (pad rows zero, never read).
    p2p: each rank sends every peer its partial of that peer's rows and receives each peer's partial of its own rows
    into a [N, n, D] staging block a piece (its own partial copied into its slot), summed rank 0 first by hc_post
    (``exchange_all``: one NCCL group, every pair a send and a receive, so it needs every pair connected); its glued
    rows go to every peer the same way. gather (for fabrics without every pair, e.g. a ring): the unsplit all-gather
    of every rank's whole partial, glue on own rows (``glue.hc_post_ranks``), then the in-place all-gather of the
    rows. Either way the partial sums are hc_post's, rank 0 first: the unsplit bits."""

    def __init__(self, w, b, settings: SplitSettings, stream=None) -> None:
        world = int(getattr(w, "world", 2))
        two = world == 2
        p2p = settings.exchange == "p2p"
        if world < 2 or w.comm is None or (p2p and not hasattr(w.comm, "exchange" if two else "exchange_all")):
            raise ValueError("TF_GLM_HC_SPLIT needs two ranks or more and a communicator with exchange "
                             "(send/receive), or TF_GLM_HC_EXCHANGE=gather")
        self.w, self.b, self.settings = w, b, settings
        self.world = world
        self.rank = w.rank
        self.peer = 1 - w.rank if two else None
        self.peers = [r for r in range(world) if r != w.rank]
        self.gather = not p2p
        # TF_GLM_PROMPT_PARTIALS=bf16: every partial rounded to bf16 before the sum (``round``); with p2p also sent as
        # bf16 (``half``: staged in the all-gather buffer's unused part, ``begin``). The all-gather exchange sends fp32
        # and rounds in hc_post: the same bits
        self.round = settings.partials == "bf16"
        self.half = self.round and p2p
        self.send16: torch.Tensor | None = None
        self.side_glue = False         # a half's split (microbatch.Halves): its glue always runs by ``glue_side``
        self.si = None                 # TF_GLM_PF_SI: this site's send-slot view [Rp, D] bf16 (pf_xchg.PART)
        self.zc_x = None               # the exchange ``zc`` is open on
        self.zc = None                 # TF_GLM_PF_ZC: the open exchange's (receive view, slot stride) for _post
        self.quanted = False           # TF_GLM_PF_KQ: the last glue shipped quantized rows, not the normed ones
        self.routed = False            # TF_GLM_PF_OWN_ROUTER: the last glue left every row's picks and weights
        self.direct = False            # TF_GLM_PF_PART16: the site's partial was filled as bf16 into send16
        # the plain NCCL all-gather (RoceComm sends small gathers over RoCE, whose kernel is not written for in place)
        self.all_gather = getattr(w.comm, "nccl", w.comm).all_gather
        self.active = False
        self.R = self.Rp = self.H = 0
        self.pieces: list[tuple[int, int]] = []
        self.got: list[torch.Tensor] = []
        self.stream = None
        self.pending = False             # a glue_side queued on the side stream that the main stream has not joined
        self.micro: list = []            # TF_GLM_PROMPT_MICROBATCH: the halves' splits of the chunk running now
        if settings.overlap:
            # TF_GLM_OVERLAP_PRIORITY: the side stream's priority, 0 (normal, the default) or -1 (high). Two Sparks, two
            # boots each: prefill the same, decode rounds after a high-priority prompt ~0.5-1% slower. ``stream``: one
            # to share (TF_GLM_PROMPT_MICROBATCH: both halves' splits on the buffers' one side stream)
            high = os.environ.get("TF_GLM_OVERLAP_PRIORITY", "0").strip() == "-1"
            self.stream = stream if stream is not None else torch.cuda.Stream(priority=-1 if high else 0)
            self.ev_own = torch.cuda.Event()
            n = settings.piece_count()
            self.ev_fill = [torch.cuda.Event() for _ in range(n)]
            self.ev_part = [torch.cuda.Event() for _ in range(n)]
            self.ev_glue = [torch.cuda.Event() for _ in range(n)]
            self.ev_done = torch.cuda.Event()
            # TF_GLM_PROMPT_OWN_FRONT: a layer's own-row shares (``share``): up to SHARES in flight on the side stream
            self.ev_share_in = [torch.cuda.Event() for _ in range(SHARES)]
            self.ev_share_out = [torch.cuda.Event() for _ in range(SHARES)]

    def padded(self, R: int) -> int:
        """R rounded up to a whole share a rank (two ranks: an odd chunk gets one pad row). At four ranks: to 16 rows
        where the buffers have them (prefill work, TF_GLM_PF_TAIL_MERGE's chunks of any length: a share of a
        multiple of 4 rows keeps every row buffer's exchange block on the RoCE exchange's 16-byte packs, so the DSA
        shares, the picks and the quantized rows do not fall back to NCCL or to every rank's own work); chunks on
        the 64-row grid, all there were before, are unchanged."""
        base = -(-R // self.world) * self.world
        if self.world == 4:
            wide = -(-R // 16) * 16
            if wide <= getattr(self.b, "alloc_rows", self.b.rows):
                return wide
        return base

    def applies(self, R: int) -> bool:
        """Whether a chunk of R rows runs split: long enough, and its pad rows fit the buffers (``pad_rows``)."""
        return R >= max(self.settings.min_rows, PF_SPLIT_MIN) and \
            self.padded(R) <= getattr(self.b, "alloc_rows", self.b.rows)

    def warm(self) -> None:
        """Open the NCCL send/receive connections to the peers at startup (every rank calls this together)."""

        if self.gather:
            return
        a = torch.zeros((1,), dtype=torch.float32, device="cuda")
        z = torch.empty_like(a)
        if self.world == 2:
            self.w.comm.exchange([a], [z], self.peer)
        else:
            outs = {p: [torch.empty_like(a)] for p in self.peers}
            self.w.comm.exchange_all({p: [a] for p in self.peers}, outs)
        torch.cuda.synchronize()
        # TF_GLM_PF_XCHG=1 (prefill work): the ring's exchanges as one-shot RDMA writes (pf_xchg), set up here, where
        # every rank comes together at startup; blocks of up to a half chunk's share of rows (fp32 partials)
        nccl = getattr(self.w.comm, "nccl", self.w.comm)
        if pf_xchg.ENABLED and pf_xchg.INSTANCE is None and self.world == 4 and getattr(nccl, "ring", False):
            rows = -(-int(getattr(self.b, "alloc_rows", self.b.rows)) // (2 * self.world)) + 64
            pf_xchg.INSTANCE = pf_xchg.RoceExchange(nccl, self.rank, self.world, rows * self.b.part.shape[1] * 4)
            # the own-row DSA shares' exchange (TF_GLM_PF_XCHG_SHARES, default on): its own proxy and stream; blocks of
            # a share's widest row buffer (the pools, the q_a rows: at most 4 KiB a row) for twice a half's rows
            if pf_xchg.SHARES:
                pf_xchg.SHARE = pf_xchg.RoceExchange(nccl, self.rank, self.world, 2 * rows * 4096)
            if os.environ.get("TF_GLM_PF_XCHG_PART", "0").strip() == "1":
                pf_xchg.PART = pf_xchg.RoceExchange(nccl, self.rank, self.world, rows * self.b.part.shape[1] * 2)
            if self.rank == 0:
                print(f"[tensorfold] prefill exchanges over the RoCE ring exchange (blocks up to "
                      f"{pf_xchg.INSTANCE.max_block >> 20} MiB{', DSA shares on their own' if pf_xchg.SHARE else ''})",
                      flush=True)

    # -- per chunk ------------------------------------------------------------------------------------------------
    def begin(self, R: int) -> None:
        b = self.b
        self.R, self.Rp = R, self.padded(R)
        self.H = H = self.Rp // self.world
        n = self.settings.piece_count()
        size = -(-H // n)
        if size > 64:                                 # whole 64-row tiles for the glue kernels where possible
            size = min(H, -(-size // 64) * 64)
        self.pieces = [(off, min(size, H - off)) for off in range(0, H, size)]
        # where the peers' partials of each piece of this rank's rows land, in the all-gather buffer (unused while
        # split). Two ranks: [n, D] fp32 a piece (gather: the peer's slot of the piece's [2, n, D]); more: a [N, n, D]
        # block a piece (p2p), or every rank's whole partial [N, Rp, D] (gather, one piece)
        d = b.part.shape[1]
        N = self.world
        if self.half:
            # bf16: the peers' partials of this rank's rows first ([n, D] a piece at two ranks, [N, n, D] past two:
            # Rp x D values in all), then this rank's bf16 partial of the peers' rows (``send16``, by row: Rp x D),
            # in the all-gather buffer's room for world x rows x D fp32
            g16 = b.gath.view(torch.bfloat16)
            if 2 * self.Rp * d > g16.numel():
                raise ValueError("TF_GLM_PROMPT_PARTIALS=bf16: the all-gather buffer cannot hold the chunk's staging")
            if N == 2:
                self.got = [g16[off * d:(off + n) * d].view(n, d) for off, n in self.pieces]
            else:
                self.got = [g16[N * off * d:N * (off + n) * d].view(N, n, d) for off, n in self.pieces]
            self.send16 = g16[self.Rp * d:2 * self.Rp * d].view(self.Rp, d)
        elif N == 2 and self.gather:
            self.got = [b.gath[(2 * off + self.peer * n) * d:(2 * off + self.peer * n + n) * d].view(n, d)
                        for off, n in self.pieces]
        elif N == 2:
            self.got = [b.gath[off * d:(off + n) * d].view(n, d) for off, n in self.pieces]
        elif self.gather:
            self.got = [b.gath[:N * self.Rp * d].view(N, self.Rp, d)]
        else:
            self.got = [b.gath[N * off * d:N * (off + n) * d].view(N, n, d) for off, n in self.pieces]
        if self.Rp > R:                                # the pad rows: zero streams and partials, never read
            b.x[R:self.Rp].zero_()
            b.part[R:self.Rp].zero_()
            if self.send16 is not None:           # (TF_GLM_PF_PART16 fills it directly: zeros as the cast gave)
                self.send16[R:self.Rp].zero_()
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream())
        self.active = True

    def finish(self) -> None:
        if self.stream is not None and self.active:
            torch.cuda.current_stream().wait_stream(self.stream)
        self.active = False
        self.pending = False
        for half in self.micro:          # a halved chunk's splits (a layer-sliced yield): their side work joined, they
            half.drain()                 # stay active for the chunk's next layers

    def drain(self) -> None:
        """The current stream waits for everything queued on the side stream; the split stays active."""
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        self.pending = False

    def join(self) -> None:
        """The current stream waits for this split's ``glue_side`` (nothing queued: nothing to wait for)."""
        if self.pending:
            torch.cuda.current_stream().wait_event(self.ev_done)
            self.pending = False

    def mine(self) -> int:
        return self.rank * self.H

    def theirs(self) -> int:
        return self.peer * self.H

    def _fill(self, fill: Callable[[int, int], None], lo: int, hi: int) -> None:
        hi = min(hi, self.R)
        if hi > lo:
            fill(lo, hi)

    def _swap_partial(self, j: int) -> None:
        """Piece j: this rank's partial of the peers' rows out, the peers' partials of this rank's rows in."""

        off, n = self.pieces[j]
        b = self.b
        if self.world > 2:
            if self.gather:                             # every rank's whole partial (one piece)
                self.all_gather(b.part[:self.Rp].reshape(-1), self.got[0].reshape(-1))
                return
            x = pf_xchg.INSTANCE
            if x is not None:
                # TF_GLM_PF_XCHG: peer p's rows of this piece (b.part rows p H + off ..) straight to it, its partial of
                # this rank's rows into got[j][p]: the same bytes as the NCCL exchange below (bf16 partials: each
                # peer's rows rounded into send16 first, on this stream, as below)
                d = b.part.shape[1]
                if self.half:
                    if not self.direct:
                        for p in self.peers:
                            lo = p * self.H + off
                            self.send16[lo:lo + n].copy_(b.part[lo:lo + n])
                    src = self.send16[off:]
                else:
                    src = b.part[off:]
                es = src.element_size()
                blk = n * d * es
                if x.fits(blk):
                    xp = pf_xchg.PART
                    if self.si is not None and PF_SI == 1:
                        # the blocks are in PART's send slot (the fills wrote them): doorbell, wait, hc_post
                        self.zc, self.zc_x = xp.exchange_open(None, 0, blk, staged=True) + (blk,), xp
                        return
                    if PF_ZC and self.half and self.side_glue and len(self.pieces) == 1:
                        # the peers' partials stay in the receive slots for this site's hc_post (``_post``, next on
                        # this stream: ``glue_side`` follows every site's partial before any other exchange)
                        self.zc_x = x
                        self.zc = x.exchange_open(src, self.H * d * es, blk) + (blk,)
                        if self.si is not None:         # shadow: the same blocks over PART from its send slot
                            first, stride, _ = self.zc
                            f2, s2 = xp.exchange_open(None, 0, blk, staged=True)
                            for p in self.peers:
                                xp.zc_bad += (first[p * stride:p * stride + blk // 2].view(torch.int16) !=
                                              f2[p * s2:p * s2 + blk // 2].view(torch.int16)).sum()
                            xp.zc_ops += 1
                            xp.exchange_close()
                        return
                    x.exchange(src, self.H * d * es, self.got[j], blk, blk)
                    return
            if self.half:                               # each peer's rows rounded to bf16 (on this stream)
                sends = {}
                for p in self.peers:
                    lo = p * self.H + off
                    sends[p] = [self.send16[lo:lo + n]]
                    if not self.direct:
                        sends[p][0].copy_(b.part[lo:lo + n])
            else:
                sends = {p: [b.part[p * self.H + off:p * self.H + off + n]] for p in self.peers}
            self.w.comm.exchange_all(sends, {p: [self.got[j][p]] for p in self.peers})
            return
        lo = self.theirs() + off
        send = b.part[lo:lo + n]
        if self.half:
            send = self.send16[lo:lo + n]
            send.copy_(b.part[lo:lo + n])
        if self.gather:
            d = send.shape[1]
            self.all_gather(send.reshape(-1), b.gath[2 * off * d:2 * (off + n) * d])
        else:
            self.w.comm.exchange([send], [self.got[j]], self.peer)

    def _xchg_rows(self, outs: list[torch.Tensor], off: int, n: int, x=None) -> bool:
        """TF_GLM_PF_XCHG: rows [mine + off, + n) of each row buffer in ``outs`` to every peer and the peers' rows
        [p H + off, + n) in, over the RoCE ring exchange ``x`` (default the main one; this rank's rows the source of
        every block, in place); False (nothing sent) when the exchange is off or a buffer does not fit, which every
        rank decides alike."""

        x = pf_xchg.INSTANCE if x is None else x
        if x is None or not outs:
            return False
        sizes = []
        for t in outs:
            if t.dim() != 2 or not t.is_contiguous():
                return False
            sizes.append(t.shape[1] * t.element_size())
        if not all(x.fits(n * rb) for rb in sizes):
            return False
        mine = self.mine()
        for t, rb in zip(outs, sizes):
            x.exchange(t[mine + off:], 0, t[off:], self.H * rb, n * rb)
        return True

    def _swap_rows(self, j: int, outs: list[torch.Tensor]) -> None:
        """Piece j of bf16 row buffers: this rank's rows out, the peers' rows in (gather: in place, whole shares)."""

        off, n = self.pieces[j]
        mine = self.mine() + off
        if self.gather:                                 # one piece: rank order is row order
            for t in outs:
                self.all_gather(t[mine:mine + n].reshape(-1), t[:self.Rp].reshape(-1))
            return
        if self.world > 2:
            if self._xchg_rows(outs, off, n):
                return
            self.w.comm.exchange_all({p: [t[mine:mine + n] for t in outs] for p in self.peers},
                                     {p: [t[p * self.H + off:p * self.H + off + n] for t in outs] for p in self.peers})
            return
        theirs = self.theirs() + off
        self.w.comm.exchange([t[mine:mine + n] for t in outs], [t[theirs:theirs + n] for t in outs], self.peer)

    def _swap_share(self, outs: list[torch.Tensor]) -> None:
        """Every rank's whole share of rows [r H, r H + H) of each row buffer in ``outs`` to every rank (this rank's
        out, the peers' in; pad rows travel too, never read): ``_swap_rows`` for the share in one piece."""

        H, mine = self.H, self.mine()
        if self.gather:                                 # rank order is row order
            for t in outs:
                self.all_gather(t[mine:mine + H].reshape(-1), t[:self.Rp].reshape(-1))
            return
        if self.world > 2:
            if self._xchg_rows(outs, 0, H):
                return
            self.w.comm.exchange_all({p: [t[mine:mine + H] for t in outs] for p in self.peers},
                                     {p: [t[p * H:p * H + H] for t in outs] for p in self.peers})
            return
        theirs = self.theirs()
        self.w.comm.exchange([t[mine:mine + H] for t in outs], [t[theirs:theirs + H] for t in outs], self.peer)

    def share(self, outs: list[torch.Tensor], slot: int = 0):
        """TF_GLM_PROMPT_OWN_FRONT: this rank's rows of ``outs`` (contiguous row buffers it wrote on the current
        stream) to every peer and theirs in. With the overlap: on the side stream after the current stream's work so
        far, returning the event the reader waits on (``wait``) before it reads the peers' rows; it must not write
        ``outs`` before that either. Without: on the current stream, None. Every rank issues its shares (and the
        glue's swaps) in one order, so the collectives match."""

        if self.stream is None:
            self._swap_share(outs)
            return None
        main = torch.cuda.current_stream()
        ev_in, ev_out = self.ev_share_in[slot], self.ev_share_out[slot]
        ev_in.record(main)
        x = pf_xchg.SHARE
        if x is not None and self.world > 2:
            # TF_GLM_PF_XCHG (prefill work): the own-row shares on their own exchange and stream, so they never wait
            # behind the other half's partials and glue on the split's side stream (disjoint rows, the same bytes)
            ss = pf_xchg.share_stream()
            ss.wait_event(ev_in)
            with torch.cuda.stream(ss):
                done = self._xchg_rows(outs, 0, self.H, x)
                if done:
                    ev_out.record(ss)
            if done:
                return ev_out
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ev_in)
            self._swap_share(outs)
            ev_out.record(self.stream)
        return ev_out

    @staticmethod
    def wait(ev) -> None:
        """The current stream waits for a ``share`` (None: it ran on the current stream)."""
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)

    def partial(self, fill: Callable[[int, int], None]) -> None:
        """This rank's fp32 partial rows (``fill(lo, hi)`` writes b.part[lo:hi]), the peers' rows sent to them and
        their partials of this rank's rows received (``got``)."""

        mine = self.mine()
        self.direct = PF_PART16 and self.half and self.world > 2 and not self.gather and self.stream is not None
        self.si = None
        xp = pf_xchg.PART
        if (PF_SI and self.direct and PF_ZC == 1 and xp is not None and self.side_glue and len(self.pieces) == 1
                and xp.fits(self.H * self.b.part.shape[1] * 2)):
            d = self.b.part.shape[1]
            self.si = xp.send_view(self.H * d * 2).view(self.Rp, d)
            if self.Rp > self.R:                        # the pad rows: zeros, as send16's
                self.si[self.R:self.Rp].zero_()
        if self.si is not None and PF_SI == 1:
            self.b.part16 = self.si
            try:
                self._partial(fill, mine)
            finally:
                self.b.part16 = None
            return
        if self.direct:
            self.b.part16 = self.send16
            try:
                self._partial(fill, mine)
            finally:
                self.b.part16 = None
            return
        self._partial(fill, mine)

    def _partial(self, fill: Callable[[int, int], None], mine: int) -> None:
        if self.stream is None or (self.world > 2 and self.gather):
            fill(0, self.R)
            if self.stream is None:
                for j in range(len(self.pieces)):
                    self._swap_partial(j)
                return
            main = torch.cuda.current_stream()          # past two ranks, gather: the whole partial, then its gather
            self.ev_fill[0].record(main)
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(self.ev_fill[0])
                self._swap_partial(0)
                self.ev_part[0].record(self.stream)
            return
        main = torch.cuda.current_stream()
        if PF_FILL_WHOLE and self.world > 2:
            # TF_GLM_PF_FILL_WHOLE (prefill work): the whole partial in one fill (one matmul over every row instead of
            # a call a peer and piece; rows are independent: the same bits), then each piece's exchange on the side
            # stream as before; the exchanges start once the fill is done (the other half's compute covers them
            # with TF_GLM_PROMPT_MICROBATCH)
            fill(0, self.R)
            self.ev_fill[0].record(main)
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(self.ev_fill[0])
                for j in range(len(self.pieces)):
                    self._swap_partial(j)
                    self.ev_part[j].record(self.stream)
            return
        others = [self.theirs()] if self.world == 2 else [p * self.H for p in self.peers]
        for j, (off, n) in enumerate(self.pieces):
            for start in others:
                self._fill(fill, start + off, start + off + n)
            if self.si is not None and PF_SI == 2:      # shadow: the peers' rows once more, into the send slot
                keep, self.b.part16 = self.b.part16, self.si
                try:
                    for start in others:
                        self._fill(fill, start + off, start + off + n)
                finally:
                    self.b.part16 = keep
            self.ev_fill[j].record(main)
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(self.ev_fill[j])
                self._swap_partial(j)
                self.ev_part[j].record(self.stream)
        self._fill(fill, mine, mine + self.H)

    def own(self) -> tuple[int, int]:
        """This rank's rows of the chunk, [lo, hi) (the pad rows left out)."""

        return self.mine(), min(self.mine() + self.H, self.R)

    def _post(self, j: int, lo: int, hi: int) -> None:
        """hc_post of this rank's rows lo .. hi (piece j): every rank's partial of them summed rank 0 first."""

        b = self.b
        x = b.x[lo:hi]
        own = b.part[lo:hi]
        if self.world == 2:
            got = self.got[j]
            if self.half:
                own = own.to(torch.bfloat16)
            g0, g1 = (own, got) if self.rank == 0 else (got, own)
            glue.hc_post_pair(x, x, g0, g1, b.post[lo:hi], b.comb[lo:hi], round_parts=self.round)
        elif self.gather:
            whole = self.got[0]
            glue.hc_post_ranks(x, x, whole[0, lo:hi], self.Rp * whole.shape[2], self.world, b.post[lo:hi],
                               b.comb[lo:hi], round_parts=self.round)
        elif self.zc is not None:
            first, stride, blk = self.zc
            self.zc = None
            xi = self.zc_x
            if self.si is not None and PF_SI == 1 and xi is pf_xchg.PART:
                own16 = self.si[lo:hi]                  # this rank's rows: its own block of the send slot
            else:
                own16 = self.send16[lo:hi]
                if not self.direct:
                    own16.copy_(own)
            if PF_ZC == 2:                              # shadow: compare, then the usual copy and hc_post
                xz = x.clone()
                glue.hc_post_zc(xz, xz, first, stride, own16, self.rank, self.world, b.post[lo:hi], b.comb[lo:hi])
                block = self.got[j]
                xi.exchange_close(block, blk, blk)
                block[self.rank].copy_(own16)
                glue.hc_post(x, x, block, b.post[lo:hi], b.comb[lo:hi], round_parts=self.round)
                xi.zc_bad += (x.view(torch.int16) != xz.view(torch.int16)).sum()
                xi.zc_ops += 1
            else:
                glue.hc_post_zc(x, x, first, stride, own16, self.rank, self.world, b.post[lo:hi], b.comb[lo:hi])
                xi.exchange_close()
        else:
            block = self.got[j]
            # bf16: rounded as the peers rounded theirs (TF_GLM_PF_PART16: already rounded, by the fill)
            block[self.rank].copy_(self.send16[lo:hi] if self.direct else own)
            glue.hc_post(x, x, block, b.post[lo:hi], b.comb[lo:hi], round_parts=self.round)

    def glue_side(self, hc=None, norm=None, taps: tuple[int, ...] = (), final: bool = False,
                  share: bool = True, router: Callable[[int, int], None] | None = None, quant=None) -> None:
        """``glue`` with its kernels on the side stream after the current stream's fills so far (TF_GLM_PROMPT_
        MICROBATCH: the half's glue under the other half's compute; ``join`` before reading what it wrote). The same
        kernels on the same rows in the same order, after the partials' exchange (earlier on the side stream). Without
        a side stream: ``glue`` in place."""

        self.routed = self.quanted = False
        if self.stream is None:
            self.glue(hc, norm, taps, final, share=share)
            return
        b = self.b
        # ``quant`` (TF_GLM_PF_KQ): the next layer's KDA input projection (a 4-bit one, with FP8 prompt activations)
        from . import forward as _fwd, qmm as _qmm
        if quant is not None and not (PF_KQ and _fwd.PF_FP8 and hc is not None and share and isinstance(quant, _qmm.Q4)
                                      and len(self.pieces) == 1 and self.world > 2 and not self.gather
                                      and pf_xchg.INSTANCE is not None and hasattr(b, "nq8")
                                      and quant.k == b.normed.shape[1] and quant.gs == 64 and all(
                                          pf_xchg.INSTANCE.fits(self.H * t.shape[1] * t.element_size())
                                          for t in (b.nq8, b.nqs, b.nqa))):
            quant = None
        # ``router(lo, hi)`` (TF_GLM_PF_OWN_ROUTER): own rows' picks and weights (b.pick, b.wts) after their hc_pre;
        # only where the exchange takes the rows (whole 16-byte packs, every rank alike) and the share is at least
        # one 128-row router tile (the kernel's tile for the whole half)
        if router is not None and not (PF_OWN_ROUTER and hc is not None and share and len(self.pieces) == 1
                                       and self.world > 2 and not self.gather and pf_xchg.INSTANCE is not None
                                       and self.H >= 128 and all(
                                           pf_xchg.INSTANCE.fits(self.H * t.shape[1] * t.element_size())
                                           for t in (b.pick, b.wts, b.normed))):
            router = None
        self.ev_own.record(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.ev_own)
            self._glue_rows(hc, norm, taps, final, share, router, quant)
            self.ev_done.record(self.stream)
        self.pending = True
        self.routed = router is not None
        self.quanted = quant is not None

    def _glue_rows(self, hc, norm, taps: tuple[int, ...], final: bool, share: bool, router=None, quant=None) -> None:
        """``glue``'s work on the current stream, piece by piece, its swaps in place (``glue_side``)."""

        b, c = self.b, self.w.cfg
        mine = self.mine()
        outs = [b.taps[slot] for slot in taps] + ([b.hidden] if final else []) + \
            ([b.nq8, b.nqs, b.nqa] if quant is not None else [b.normed] if hc is not None and share else []) + \
            ([b.pick, b.wts] if router is not None else [])
        for j, (off, n) in enumerate(self.pieces):
            lo, hi = mine + off, mine + off + n
            self._post(j, lo, hi)
            x = b.x[lo:hi]
            for slot in taps:
                glue.stream_mean(x, b.taps[slot][lo:hi])
            if final:
                glue.stream_mean(x, b.hidden[lo:hi])
            if hc is not None:
                glue.hc_pre(x, hc.fn, hc.base, hc.scale, norm, b.normed[lo:hi], b.xs[lo:hi], b.post[lo:hi],
                            b.comb[lo:hi], b.hcpart[lo:hi], c.eps, c.hc_eps, c.hc_iters, prompt=True)
            if router is not None:
                router(lo, hi)
            if quant is not None:                       # shared.quantize_rows's launch, into the row buffers
                from tensorfold.cuda.kernels import qmm as _shared
                xn = b.normed[lo:hi]
                _shared._quantize_rows[(hi - lo,)](xn, b.nq8[lo:hi], b.nqs[lo:hi], b.nqa[lo:hi], xn.stride(0),
                                                   K=xn.shape[1], GS=quant.gs, BK=256, num_warps=4)
            if outs:
                self._swap_rows(j, outs)

    def glue(self, hc=None, norm=None, taps: tuple[int, ...] = (), final: bool = False,
             front: Callable[[int, int], None] | None = None, share: bool = True) -> None:
        """hc_post of this rank's rows, then (in order) the layer's DFlash2 taps, the final stream mean (``final``)
        and the next hc_pre (``hc``, ``norm``); their rows swapped with the peers'. ``front(lo, hi)``
        (TF_GLM_PREFILL_OVERLAP=2): the next block's row-independent start on this rank's rows of each piece, on the
        main stream right after the piece's glue, while its swap (and the later pieces' partials) are on the wire;
        the caller runs it on the peers' rows (``own`` is what was done) after this returns. ``share`` False
        (TF_GLM_PROMPT_OWN_FRONT, before a DSA layer): the normed rows stay this rank's own (the DSA block's own-row
        front shares what the other ranks need); the taps and the final mean still travel."""

        b, c = self.b, self.w.cfg
        mine = self.mine()
        main = torch.cuda.current_stream()
        outs = [b.taps[slot] for slot in taps] + ([b.hidden] if final else []) + \
            ([b.normed] if hc is not None and share else [])
        for j, (off, n) in enumerate(self.pieces):
            lo, hi = mine + off, mine + off + n
            if self.stream is not None:
                main.wait_event(self.ev_part[0 if self.world > 2 and self.gather else j])
            self._post(j, lo, hi)
            x = b.x[lo:hi]
            for slot in taps:
                glue.stream_mean(x, b.taps[slot][lo:hi])
            if final:
                glue.stream_mean(x, b.hidden[lo:hi])
            if hc is not None:
                glue.hc_pre(x, hc.fn, hc.base, hc.scale, norm, b.normed[lo:hi], b.xs[lo:hi], b.post[lo:hi],
                            b.comb[lo:hi], b.hcpart[lo:hi], c.eps, c.hc_eps, c.hc_iters, prompt=True)
            if self.stream is None:
                if outs:
                    self._swap_rows(j, outs)
                if front is not None:
                    self._fill(front, lo, hi)
                continue
            if outs:
                self.ev_glue[j].record(main)
                with torch.cuda.stream(self.stream):
                    self.stream.wait_event(self.ev_glue[j])
                    self._swap_rows(j, outs)
            if front is not None:
                self._fill(front, lo, hi)
        if self.stream is not None:
            self.ev_done.record(self.stream)
            main.wait_event(self.ev_done)
