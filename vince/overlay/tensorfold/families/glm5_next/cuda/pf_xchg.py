"""Prefill work (2026-11-08), TF_GLM_PF_XCHG=1: the row split's exchanges (``hcsplit``: the partials of the other ranks'
rows out, theirs of this rank's rows in; the glued rows to every rank) on a four-Spark ring as one-shot RDMA writes
through a CPU proxy, instead of NCCL's two-rank link all-gathers (which ran the cables at ~31% of what ib_write_bw
measures on them: 196 Gb/s a direction).

Transport: b12x's "RoCEnante" one-shot protocol (https://github.com/local-inference-lab/b12x, Apache-2.0) as
TensorFold runs it (``tensorfold.cuda.roce``, ``roce_proxy.c`` with Mia's AI Lab's ring relay), in a separate copy of
the proxy (``pf_xchg_proxy.c``) with an exchange (all-to-all) mode: every rank's block for each neighbour goes to it
directly, its block for the rank across goes half by each neighbour, whose proxies relay it on. Design ideas from the
switchless-ring collectives of blaesus/dsgx10 (Andy Shu: one-shot writes, the opposite rank's halves relayed by the
neighbours both ways) and christopherowen/dgx-spark-networking (ring4 host relay). Bytes only move: every bit of the
prompt is NCCL's exchange's (``tools`` check: the greedy replies, ref_test.py).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import threading
from functools import lru_cache
from pathlib import Path

import torch

from tensorfold.cuda.roce import MAX_HCAS, SLOT_ALIGN, _hcas, _ipv4_gid, plan_routes, spin_limit

ENABLED = os.environ.get("TF_GLM_PF_XCHG", "0").strip() == "1"
PART = None               # TF_GLM_PF_XCHG_PART=1: a third exchange for the halves' partials (their fills write its send slots)
INSTANCE = None           # the process's exchange, made by hcsplit.HcSplit.warm at startup (every rank together)
SHARES = os.environ.get("TF_GLM_PF_XCHG_SHARES", "1").strip() == "1"
SHARE = None              # the own-row DSA shares' exchange (its own proxy, flags and stream; same startup point)
_SHARE_STREAM = None
BCAST = os.environ.get("TF_GLM_PF_XCHG_BCAST", "0").strip() == "1"   # same bytes on the wire; local to a rank's sends
TIMED = os.environ.get("TF_GLM_PF_XCHG_TIME", "0").strip() == "1"     # measurement: per-rank stage / wait / copy times


def report() -> None:
    """TF_GLM_PF_XCHG_TIME: each exchange's timings since the last call (pf_switches.refresh: every prompt's start)."""
    if TIMED or (INSTANCE is not None and INSTANCE.zc_ops) or (PART is not None and PART.zc_ops):
        for name, x in (("partials+rows", INSTANCE), ("shares", SHARE), ("direct partials", PART)):
            if x is not None:
                x.report(name)


def share_stream():
    """The shares' CUDA stream, made once."""
    global _SHARE_STREAM
    if _SHARE_STREAM is None:
        _SHARE_STREAM = torch.cuda.Stream()
    return _SHARE_STREAM
_SOURCE = Path(__file__).with_name("pf_xchg_proxy.c")
_LOCK = threading.Lock()
THREADS = 512
MAX_GRID = int(os.environ.get("TF_GLM_PF_XCHG_GRID", "32") or 32)
ABI = 1007


def _cache_dir() -> Path:
    root = os.environ.get("TORCH_EXTENSIONS_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "tensorfold")
    return Path(root) / "pf_roce"


@lru_cache(maxsize=1)
def proxy_library() -> ctypes.CDLL:
    with _LOCK:
        source = _SOURCE.read_bytes()
        target = _cache_dir() / f"pf_xchg_proxy-{hashlib.sha256(source).hexdigest()[:16]}.so"
        if not target.exists():
            cc = next((c for c in (os.environ.get("CC"), "gcc", "cc", "clang") if c and shutil.which(c)), None)
            if cc is None:
                raise RuntimeError("the RoCE exchange needs a C compiler and libibverbs headers")
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.{os.getpid()}")
            cmd = [cc, "-O2", "-std=gnu11", "-shared", "-fPIC", "-o", str(tmp), str(_SOURCE), "-libverbs", "-lpthread"]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError("building the RoCE exchange proxy failed: " + " ".join(cmd) + "\n" + done.stderr)
            os.replace(tmp, target)
        lib = ctypes.CDLL(str(target), use_errno=True)
    u64, p, i = ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int
    for name, res, args in (
            ("roce_abi_version", i, []), ("roce_layout", i, [i, u64, u64, ctypes.POINTER(u64)]),
            ("roce_blob_bytes", u64, []),
            ("roce_create", p, [i, i, ctypes.POINTER(ctypes.c_char_p), i, ctypes.POINTER(i), i, p, u64, u64, u64,
                                ctypes.c_char_p, u64]),
            ("roce_set_routes", i, [p, i, i, ctypes.POINTER(i), ctypes.POINTER(i), i]), ("roce_set_ring", i, [p]),
            ("roce_set_exchange", i, [p, u64]),
            ("roce_local_blob", i, [p, p, u64]), ("roce_connect", i, [p, p, u64]), ("roce_start", i, [p]),
            ("roce_stop", None, [p]), ("roce_failed", i, [p]), ("roce_error", ctypes.c_char_p, [p]),
            ("roce_stat", u64, [p, i]), ("roce_hca_stat", u64, [p, i, i]), ("roce_destroy", None, [p])):
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
    if lib.roce_abi_version() != ABI:
        raise RuntimeError("unexpected RoCE exchange proxy ABI version")
    return lib


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_glm_pf_xchg_v5", sources=[str(here / "pf_xchg.cpp"), str(here / "pf_xchg.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


class _Alias:
    """A pinned host range as a CUDA array (a GB10's memory is one: the kernels here already read and write the pinned
    region by its host address), so torch, Triton and the extensions take it as any device tensor."""

    def __init__(self, ptr: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 2}


class RoceExchange:
    """One rank's side of the ring exchange: the pinned region, the proxy thread, the launches. Collective: every
    rank of the ring constructs it at the same point (its setup all-gathers through ``comm``)."""

    def __init__(self, comm, rank: int, world: int, max_block_bytes: int) -> None:
        if world != 4:
            raise ValueError("the RoCE exchange runs on a ring of four ranks")
        lib = proxy_library()
        self.lib, self.rank, self.world = lib, rank, world
        self.max_block = -(-int(max_block_bytes) // SLOT_ALIGN) * SLOT_ALIGN
        self.slot_bytes = 2 * self.max_block              # [the block for this rank | the relay area]
        self.relay_off = self.max_block
        self.send_slot_bytes = world * self.max_block
        out = (ctypes.c_uint64 * 7)()
        if lib.roce_layout(world, self.slot_bytes, self.send_slot_bytes, out) != 0:
            raise ValueError("RoCE exchange layout refused")
        recv_off, flag_off, send_off, ctrl_off, total, self.flag_stride, self.slots = (int(v) for v in out)
        self.recv_off, self.send_off, self.open = recv_off, send_off, False
        self._gpu = None
        self.zc_bad = torch.zeros(1, dtype=torch.int64, device="cuda")     # TF_GLM_PF_ZC=2: mismatched elements
        self.zc_ops = 0
        self.region = torch.zeros(total, dtype=torch.uint8, pin_memory=True)
        base = self.region.data_ptr()
        self.recv, self.flag, self.send, self.ctrl = base + recv_off, base + flag_off, base + send_off, base + ctrl_off
        self.ctrl_words = self.region[ctrl_off:ctrl_off + 28].view(torch.int32).numpy()
        self.counters = torch.zeros(4, dtype=torch.int32, device="cuda")     # epoch, stage, copy, poison
        self.hcas = _hcas(world)
        fixed = os.environ.get("TF_ROCE_GID_INDEX") or os.environ.get("NCCL_IB_GID_INDEX")
        found = [_ipv4_gid(h) for h in self.hcas]
        gids = [int(fixed)] * len(self.hcas) if fixed else [f[0] if f is not None else 3 for f in found]
        self.addrs = [None if f is None else (f[1], f[2]) for f in found]
        tc = int(os.environ.get("NCCL_IB_TC", "0"), 0)
        names = (ctypes.c_char_p * len(self.hcas))(*[n.encode() for n in self.hcas])
        err = ctypes.create_string_buffer(512)
        self.ctx = lib.roce_create(world, rank, names, len(self.hcas), (ctypes.c_int * len(gids))(*gids), tc,
                                   ctypes.c_void_p(base), total, self.slot_bytes, self.send_slot_bytes, err, len(err))
        failed = 0 if self.ctx else 1
        blob_len = int(lib.roce_blob_bytes())
        blob = ctypes.create_string_buffer(blob_len)
        if not failed and lib.roce_local_blob(self.ctx, blob, blob_len) != 0:
            failed = 1
        words = -(-blob_len // 4)
        extra = 2 * MAX_HCAS
        mine = torch.zeros(4 + extra + words, dtype=torch.int32)
        mine[:4] = torch.tensor([failed, 0, self.slot_bytes >> 12, self.flag_stride])
        mine[4:4 + extra] = torch.tensor([0, -1] * MAX_HCAS, dtype=torch.int32)
        for h, a in enumerate(self.addrs):
            if a is not None:
                mine[4 + 2 * h] = a[0] - (1 << 32) if a[0] >= 1 << 31 else a[0]
                mine[5 + 2 * h] = a[1]
        mine[4 + extra:] = torch.frombuffer(bytearray(blob.raw.ljust(words * 4, b"\0")), dtype=torch.int32)
        got = torch.empty(world * mine.numel(), dtype=torch.int32, device="cuda")
        comm.all_gather(mine.cuda(), got)
        rows = got.view(world, -1).cpu()
        if int(rows[:, 0].sum()) or len(set(tuple(r[1:4].tolist()) for r in rows)) != 1:
            why = err.value.decode(errors="replace") if not self.ctx else "the ranks' exchange geometry differs"
            self.close()
            raise RuntimeError(f"RoCE exchange setup failed on some rank: {why}")
        addrs = [[None if int(r[5 + 2 * h]) < 0 else (int(r[4 + 2 * h]) & 0xFFFFFFFF, int(r[5 + 2 * h]))
                  for h in range(MAX_HCAS)] for r in rows]
        routes, self.n_flag = plan_routes(addrs, True)
        for p in range(world):
            if p == rank or (rank, p) not in routes:
                continue
            local, remote = zip(*routes[(rank, p)])
            n = len(local)
            if lib.roce_set_routes(self.ctx, p, n, (ctypes.c_int * n)(*local), (ctypes.c_int * n)(*remote),
                                   self.n_flag) != 0:
                raise RuntimeError(f"RoCE exchange routes: {lib.roce_error(self.ctx).decode(errors='replace')}")
        if lib.roce_set_ring(self.ctx) != 0 or lib.roce_set_exchange(self.ctx, self.relay_off) != 0:
            raise RuntimeError(f"RoCE exchange ring: {lib.roce_error(self.ctx).decode(errors='replace')}")
        blobs = b"".join(bytes(r[4 + extra:].numpy().tobytes())[:blob_len] for r in rows)
        self.spin = spin_limit()
        _ext()
        ok = lib.roce_connect(self.ctx, blobs, len(blobs)) == 0 and lib.roce_start(self.ctx) == 0
        verdict = torch.tensor([0 if ok else 1], dtype=torch.int32, device="cuda")
        both = torch.empty(world, dtype=torch.int32, device="cuda")
        comm.all_gather(verdict, both)
        if int(both.sum()):
            why = lib.roce_error(self.ctx).decode(errors="replace") if self.ctx else ""
            self.close()
            raise RuntimeError(f"RoCE exchange could not connect: {why}")
        self.ops = 0
        self.timed: list = []

    def report(self, name: str) -> None:
        """TF_GLM_PF_XCHG_TIME: this rank's exchange timings since the last report (finished ops only), one line."""
        done = [(nb, ev) for nb, ev in self.timed if ev[3].query()]
        self.timed = [t for t in self.timed if not t[1][3].query()]
        if self.zc_ops:
            print(f"[tensorfold] pf_xchg {name} rank {self.rank}: shadow check: {self.zc_ops} "
                  f"hc_posts compared, {int(self.zc_bad.item())} elements differ", flush=True)
            self.zc_ops = 0
        if not done:
            return
        st, wt, cp = ([ev[k - 1].elapsed_time(ev[k]) for _, ev in done] for k in (1, 2, 3))
        big = [i for i, (nb, _) in enumerate(done) if nb >= 4 << 20]
        ws = sorted(wt)
        wb = sorted(wt[i] for i in big) or [0.0]
        q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))]
        hs = [(int(self.lib.roce_hca_stat(self.ctx, h, 0)), int(self.lib.roce_hca_stat(self.ctx, h, 1)) >> 20)
              for h in range(len(self.hcas))]
        print(f"[tensorfold] pf_xchg {name} rank {self.rank}: {len(done)} ops, {sum(nb for nb, _ in done) >> 20} MiB a "
              f"peer; ms stage {sum(st):.1f} wait {sum(wt):.1f} copy {sum(cp):.1f}; wait median {q(ws, .5):.3f} p90 "
              f"{q(ws, .9):.3f} max {ws[-1]:.3f}, under 0.05 ms (this rank last) {sum(w < 0.05 for w in wt)}; "
              f"{len(big)} blocks of 4+ MiB: wait median {q(wb, .5):.3f} p90 {q(wb, .9):.3f}, under 0.05 ms "
              f"{sum(wt[i] < 0.05 for i in big)}; proxy posted {int(self.lib.roce_stat(self.ctx, 0))} relayed "
              f"{int(self.lib.roce_stat(self.ctx, 3))}; per HCA (writes, MiB) {hs}", flush=True)

    def fits(self, block_bytes: int) -> bool:
        return 0 < block_bytes <= self.max_block and block_bytes % 16 == 0

    def exchange(self, src: torch.Tensor, src_stride: int, dst: torch.Tensor, dst_stride: int, block_bytes: int) -> None:
        """Block p of ``src`` (its bytes from src + p * src_stride, ``block_bytes`` of them) to rank p, and rank p's
        block for this rank into dst + p * dst_stride, for every peer p (strides in bytes, multiples of 16; this rank's
        own block is neither sent nor written). On the current stream; every rank calls it in the same order."""

        if not self.fits(block_bytes) or src_stride % 16 or dst_stride % 16:
            raise ValueError(f"RoCE exchange: a {block_bytes}-byte block (at most {self.max_block}), strides of 16")
        packs = block_bytes // 16
        grid = max(1, min(MAX_GRID, -(-self.world * packs // (THREADS * 4))))
        if BCAST and src_stride == 0:
            # TF_GLM_PF_XCHG_BCAST: the same block to every peer (the glued rows): staged once, the proxy sends each
            # peer that one block (a stride of -1 tells the stage kernel)
            src_stride = -16
        if TIMED and not torch.cuda.is_current_stream_capturing():
            # TF_GLM_PF_XCHG_TIME=1 (measurement only): the three launches apart, a timing event between them
            # (``report`` prints each rank's sums at the next prompt's start); the same kernels and bytes
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            ev[0].record()
            for k in (1, 2, 3):
                _ext().exchange(src, src_stride // 16, dst, dst_stride // 16, packs, self.recv, self.flag, self.send,
                                self.ctrl, self.slot_bytes, self.send_slot_bytes, self.counters.data_ptr(), self.spin,
                                self.world, self.rank, self.slots, self.flag_stride, self.n_flag, -k, THREADS)
                ev[k].record()
            self.timed.append((block_bytes, ev))
        else:
            _ext().exchange(src, src_stride // 16, dst, dst_stride // 16, packs, self.recv, self.flag, self.send,
                            self.ctrl, self.slot_bytes, self.send_slot_bytes, self.counters.data_ptr(), self.spin,
                            self.world, self.rank, self.slots, self.flag_stride, self.n_flag, grid, THREADS)
        self.ops += 1
        if not torch.cuda.is_current_stream_capturing() and self.ops % 64 == 0:
            self.check()

    # -- TF_GLM_PF_ZC: the receive side without its copy ---------------------------------------------------------------
    def gpu(self) -> torch.Tensor:
        """The pinned region as a CUDA uint8 tensor (``_Alias``), made once."""
        if self._gpu is None:
            self._gpu = torch.as_tensor(_Alias(self.region.data_ptr(), self.region.numel()), device="cuda")
        return self._gpu

    def send_view(self, block_bytes: int) -> torch.Tensor:
        """TF_GLM_PF_SI: the next exchange's send slot as bf16 [world x block] (block p for rank p, this rank's own
        never sent), for the caller's kernels to write before ``exchange_open(None, ..., staged=True)``; nothing else
        may be issued on this exchange in between."""

        if not self.fits(block_bytes) or self.open:
            raise ValueError("RoCE exchange (send view): block size, or an exchange open")
        lo = self.send_off + ((self.ops + 1) & 1) * self.send_slot_bytes
        return self.gpu()[lo:lo + self.world * block_bytes].view(torch.bfloat16)

    def exchange_open(self, src: torch.Tensor | None, src_stride: int, block_bytes: int,
                      staged: bool = False) -> tuple[torch.Tensor, int]:
        """The exchange's stage and wait only: the peers' blocks stay in their receive slots. Returns a bf16 view of
        the pinned region from rank 0's slot of this sequence and the distance between ranks' slots in bf16 elements
        (rank p's block at p x that; this rank's own slot holds nothing). The caller reads them on this stream and
        then calls ``exchange_close`` before any other exchange is issued."""

        if not self.fits(block_bytes) or src_stride % 16 or self.open:
            raise ValueError("RoCE exchange (open): block size, stride, or one already open")
        packs = block_bytes // 16
        slot = (self.ops + 1) & 1                   # the device epoch is the ops done (every op advanced it)
        if staged:                                  # ``send_view``'s blocks: the stage launch only rings the doorbell
            src, src_stride = self.counters, -32
        for k in (1, 2):
            _ext().exchange(src, src_stride // 16, src, 0, packs, self.recv, self.flag, self.send, self.ctrl,
                            self.slot_bytes, self.send_slot_bytes, self.counters.data_ptr(), self.spin, self.world,
                            self.rank, self.slots, self.flag_stride, self.n_flag, -k, THREADS)
        self.open = True
        lo = self.recv_off + slot * self.slot_bytes
        hi = self.recv_off + ((self.world - 1) * self.slots + slot) * self.slot_bytes + block_bytes
        return self.gpu()[lo:hi].view(torch.bfloat16), self.slots * self.slot_bytes // 2

    def exchange_close(self, out: torch.Tensor | None = None, out_stride: int = 0, block_bytes: int = 0) -> None:
        """Ends ``exchange_open``: the copy launch with nothing to copy (or, with ``out``, the usual copy into it),
        which advances the device epoch."""

        packs = block_bytes // 16 if out is not None else 0
        t = out if out is not None else self.counters
        _ext().exchange(t, 0, t, out_stride // 16, packs, self.recv, self.flag, self.send, self.ctrl,
                        self.slot_bytes, self.send_slot_bytes, self.counters.data_ptr(), self.spin, self.world,
                        self.rank, self.slots, self.flag_stride, self.n_flag, -3, THREADS)
        self.open = False
        self.ops += 1
        if not torch.cuda.is_current_stream_capturing() and self.ops % 64 == 0:
            self.check()

    def exchange_timed(self, src, src_stride, dst, dst_stride, block_bytes) -> tuple[float, float, float]:
        """Timing only: the exchange's three launches with events between them: (stage, wait, copy) ms."""
        packs = block_bytes // 16
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        ev[0].record()
        for k in (1, 2, 3):
            _ext().exchange(src, src_stride // 16, dst, dst_stride // 16, packs, self.recv, self.flag, self.send,
                            self.ctrl, self.slot_bytes, self.send_slot_bytes, self.counters.data_ptr(), self.spin,
                            self.world, self.rank, self.slots, self.flag_stride, self.n_flag, -k, THREADS)
            ev[k].record()
        torch.cuda.synchronize()
        self.ops += 1
        return tuple(ev[k - 1].elapsed_time(ev[k]) for k in (1, 2, 3))

    def check(self) -> None:
        if self.ctrl_words[2] != 0 or (self.ctx and self.lib.roce_failed(self.ctx)):
            why = self.lib.roce_error(self.ctx).decode(errors="replace") if self.ctx else ""
            raise RuntimeError(f"RoCE exchange failed (sequence {int(self.ctrl_words[2])}, peer {int(self.ctrl_words[3])},"
                               f" stripe {int(self.ctrl_words[6])}){': ' + why if why else ''}")

    def close(self) -> None:
        if getattr(self, "ctx", None):
            self.lib.roce_stop(self.ctx)
            self.lib.roce_destroy(self.ctx)
            self.ctx = None
