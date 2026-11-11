"""4-rank test of pf_xchg.RoceExchange: correctness (every element), strided / same-rows sources, random sizes back
to back, and speed against NCCL's ring exchange_all: python xchg_test.py RANK MASTER PORT"""
import random
import sys

import torch

from tensorfold.cuda.comm import NCCL
from tensorfold.families.glm5_next.cuda.pf_xchg import RoceExchange

rank, master, port = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
torch.cuda.set_device(0)
comm = NCCL(rank, 4, master, port)
D = 4096
X = RoceExchange(comm, rank, 4, max_block_bytes=1024 * D * 4)
peers = [p for p in range(4) if p != rank]


def val(src, dst, rows, dtype):
    return float(src * 10 + dst) + 0.25 * (rows % 7)


_tn = [0]
def timed(f, reps=20):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    _tn[0] += 1
    comm.ready(f"t{_tn[0]}")
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        f()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[-1]


bad = 0
# 1. partial-shaped exchange: src [4, rows, D] (block p for rank p), dst [4, rows, D]
for rows, dtype in ((512, torch.float32), (768, torch.float32), (1024, torch.float32), (768, torch.bfloat16)):
    src = torch.empty((4, rows, D), dtype=dtype, device="cuda")
    for p in range(4):
        src[p].fill_(val(rank, p, rows, dtype))
    dst = torch.full((4, rows, D), -1.0, dtype=dtype, device="cuda")
    blk = rows * D * src.element_size()
    X.exchange(src, blk, dst, blk, blk)
    torch.cuda.synchronize()
    X.check()
    for p in peers:
        if not bool((dst[p] == val(p, rank, rows, dtype)).all()):
            bad += 1
            print(f"rank {rank}: WRONG block from {p} rows {rows} {dtype}", flush=True)
    if not bool((dst[rank] == -1.0).all()):
        bad += 1
        print(f"rank {rank}: own block overwritten", flush=True)
    med, lo, hi = timed(lambda: X.exchange(src, blk, dst, blk, blk))
    sends = {p: [src[p]] for p in peers}
    recvs = {p: [torch.empty_like(src[p])] for p in peers}
    m2, l2, h2 = timed(lambda: comm.exchange_all(sends, recvs))
    out = 3 * blk
    comm.ready(f"p{rows}{dtype}")
    if rank == 0:
        print(f"exchange {rows:5d} rows {str(dtype)[6:]:8s} ({blk / 2**20:.0f} MiB a peer): RoCE {med:.3f} ms "
              f"({out / med / 1e6:.1f} GB/s out, min {lo:.3f} max {hi:.3f}) | NCCL {m2:.3f} ms ({out / m2 / 1e6:.1f} GB/s)",
              flush=True)
print(f"rank {rank}: phase 1 done, ops {X.ops}", file=sys.stderr, flush=True)
# 2. rows-shaped swap: this rank's rows t[mine] to every rank (src stride 0), the peers' rows into t[p * H]
H = 768
t = torch.full((4 * H, D), -1.0, dtype=torch.bfloat16, device="cuda")
t[rank * H:(rank + 1) * H].fill_(float(rank + 1))
blk = H * D * 2
X.exchange(t[rank * H:], 0, t, blk, blk)
torch.cuda.synchronize()
for p in range(4):
    if not bool((t[p * H:(p + 1) * H] == float(p + 1)).all()):
        bad += 1
        print(f"rank {rank}: swap rows of {p} WRONG", flush=True)
med, lo, hi = timed(lambda: X.exchange(t[rank * H:], 0, t, blk, blk))
if rank == 0:
    print(f"swap {H} rows bf16 ({blk / 2**20:.0f} MiB a rank): RoCE {med:.3f} ms (min {lo:.3f} max {hi:.3f})",
          flush=True)
print(f"rank {rank}: phase 2 done, ops {X.ops}", file=sys.stderr, flush=True)
# 3. random sizes back to back (flow control), checked
rng = random.Random(1234)
for it in range(200):
    rows = rng.choice((1, 3, 64, 100, 256, 511, 768, 1024))
    src = torch.empty((4, rows, D), dtype=torch.float32, device="cuda")
    for p in range(4):
        src[p].fill_(float(it * 100 + rank * 10 + p))
    dst = torch.empty((4, rows, D), dtype=torch.float32, device="cuda")
    blk = rows * D * 4
    X.exchange(src, blk, dst, blk, blk)
    if it % 10 == 9:
        torch.cuda.synchronize()
    for p in peers:
        if not bool((dst[p] == float(it * 100 + p * 10 + rank)).all()):
            bad += 1
            if bad < 5:
                print(f"rank {rank}: random op {it} rows {rows} from {p} WRONG", flush=True)
torch.cuda.synchronize()
X.check()
comm.ready("random")
flags = torch.tensor([bad], dtype=torch.int32, device="cuda")
allbad = torch.empty(4, dtype=torch.int32, device="cuda")
comm.all_gather(flags, allbad)
if rank == 0:
    print(f"RESULT: {'ALL CORRECT' if int(allbad.sum()) == 0 else 'ERRORS ' + str(allbad.tolist())} "
          f"(200 random exchanges, ops {X.ops})", flush=True)
comm.ready("done")
X.close()
