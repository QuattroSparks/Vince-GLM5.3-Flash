"""Prefill work (2026-11-08): the same-bits scheduling switches, re-read from /cache/pf-switches.json (if it exists) at
the start of every prompt, so they can be compared without a restart. Every switch here leaves each row's arithmetic and
the order of the collectives as they are, so ranks that read the file a prompt apart still agree. Keys (absent: the
value from the environment at start):

  shared_whole  0 | 1 | 2   forward.PF_SHARED_WHOLE   (TF_GLM_PF_SHARED_WHOLE)
  fill_whole    0 | 1       hcsplit.PF_FILL_WHOLE     (TF_GLM_PF_FILL_WHOLE)
  gemm          0 | 1       pf_gemm.ENABLED           (TF_GLM_PF_GEMM)
  gemm_min_rows int         pf_gemm.MIN_ROWS          (TF_GLM_PF_GEMM_MIN_ROWS)
  kda_overlap   0 | 1       microbatch.PF_KDA_OVERLAP (TF_GLM_PF_KDA_OVERLAP)
  tap_tail      0 | 1       dflash2_multi.PF_TAP_TAIL (TF_GLM_PF_TAP_TAIL)
  lean          0 | 1       forward.PF_LEAN           (TF_GLM_PF_LEAN)
  part16        0 | 1       hcsplit.PF_PART16         (TF_GLM_PF_PART16)
  pf8_tile      0 | 1 | 2   forward.PF8_TILE          (TF_GLM_PF8_TILE; pf8_tile_blocks: forward.PF8_TILE_BLOCKS)
"""

from __future__ import annotations

import json
import os

PATH = os.environ.get("TF_GLM_PF_SWITCHES", "/cache/pf-switches.json")
_last: tuple | None = None
NO_REUSE = False          # no_reuse: rank 0 admits every prompt without its kept prefix (benchmarks, same-bits checks)


def refresh() -> None:
    global _last, NO_REUSE
    from . import pf_xchg as _x

    _x.report()
    try:
        st = os.stat(PATH)
    except OSError:
        return
    key = (st.st_mtime_ns, st.st_size)
    if key == _last:
        return
    try:
        with open(PATH) as f:
            sw = json.load(f)
    except (OSError, ValueError):
        return
    _last = key
    from . import forward, hcsplit, microbatch, pf_gemm

    if "shared_whole" in sw and int(sw["shared_whole"]) in (0, 1, 2):
        forward.PF_SHARED_WHOLE = int(sw["shared_whole"])
    if "fill_whole" in sw:
        hcsplit.PF_FILL_WHOLE = bool(int(sw["fill_whole"]))
    if "gemm" in sw:
        pf_gemm.ENABLED = bool(int(sw["gemm"]))
    if "gemm_min_rows" in sw:
        pf_gemm.MIN_ROWS = int(sw["gemm_min_rows"])
    if "kda_overlap" in sw:
        microbatch.PF_KDA_OVERLAP = bool(int(sw["kda_overlap"]))
    from . import dflash2_multi

    if "tap_tail" in sw:
        dflash2_multi.PF_TAP_TAIL = bool(int(sw["tap_tail"]))
    if "lean" in sw:
        forward.PF_LEAN = bool(int(sw["lean"]))
    if "part16" in sw:                # bytes on the wire unchanged: ranks a prompt apart still agree
        hcsplit.PF_PART16 = bool(int(sw["part16"]))
    if "own_router" in sw:            # every rank must switch at the same prompt: only flip while idle, then run
        hcsplit.PF_OWN_ROUTER = bool(int(sw["own_router"]))       # one short prompt before anything measured
    if "zc" in sw and int(sw["zc"]) in (0, 1, 2):     # local to a rank's receive side: ranks may differ
        hcsplit.PF_ZC = int(sw["zc"])
    if "si" in sw and int(sw["si"]) in (0, 1, 2):     # every rank at the same prompt: flip only while idle
        hcsplit.PF_SI = int(sw["si"])
    if "xgrid" in sw and 1 <= int(sw["xgrid"]) <= 256:     # the stage / copy launches' blocks (TF_GLM_PF_XCHG_GRID)
        _x.MAX_GRID = int(sw["xgrid"])
    if "kq" in sw:                    # every rank at the same prompt: flip only while idle
        hcsplit.PF_KQ = bool(int(sw["kq"]))
    if "micro_min" in sw and int(sw["micro_min"]) >= 0:        # every rank alike too
        microbatch.PF_MICRO_MIN = int(sw["micro_min"])
    if "split_min" in sw and int(sw["split_min"]) >= 0:        # every rank alike
        hcsplit.PF_SPLIT_MIN = int(sw["split_min"])
    if "balance" in sw:
        forward.PF_BALANCE = bool(int(sw["balance"]))
    if "tail_merge" in sw:            # every rank alike (the chunk plan): flip only while idle
        forward.PF_TAIL_MERGE = bool(int(sw["tail_merge"]))
    if "bcast" in sw:
        _x.BCAST = bool(int(sw["bcast"]))
    if "xchg_time" in sw:
        _x.TIMED = bool(int(sw["xchg_time"]))
    if "pf8_tile" in sw and int(sw["pf8_tile"]) in (0, 1, 2):
        forward.PF8_TILE = int(sw["pf8_tile"])
    if "pf8_tile_blocks" in sw:
        forward.PF8_TILE_BLOCKS = int(sw["pf8_tile_blocks"])
    NO_REUSE = bool(int(sw.get("no_reuse", 0)))
    print(f"[tensorfold] pf switches: shared_whole {forward.PF_SHARED_WHOLE}, fill_whole {int(hcsplit.PF_FILL_WHOLE)}, "
          f"gemm {int(pf_gemm.ENABLED)} (min rows {pf_gemm.MIN_ROWS}), kda_overlap {int(microbatch.PF_KDA_OVERLAP)}, "
          f"tap_tail {int(dflash2_multi.PF_TAP_TAIL)}, lean {int(forward.PF_LEAN)}, part16 {int(hcsplit.PF_PART16)}, "
          f"pf8_tile {forward.PF8_TILE} (<= {forward.PF8_TILE_BLOCKS} blocks), no_reuse {int(NO_REUSE)}", flush=True)
