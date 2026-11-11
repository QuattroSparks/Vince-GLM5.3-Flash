"""Prefill work (2026-11-08), TF_GLM_PF_GEMM=1: a prompt chunk's 4-bit dense projections (``qmm.prefill_matmul``,
the prefill kernel of qmm_prefill.cu) as a bf16 weight made once by the prefill kernel's own ops (``pf_dequant.cu``)
and a plain bf16 GEMM that runs the same m16n8k16 mma over k16 steps 0, 16, 32, ... into one fp32 chain: every
output the prefill kernel's bits (checked bit for bit by micro/pf_gemm_check.py), at a plain GEMM's speed.

The bf16 weights are made per call into a small cache keyed by the Q4 (``Cache``): a prompt chunk's two halves
(TF_GLM_PROMPT_MICROBATCH) run each layer's projections back to back, so the second half finds the first's."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("TF_GLM_PF_GEMM", "0") == "1"
# rows from which a prompt matmul takes this path (fewer: the prefill kernel, whose bits these are anyway)
MIN_ROWS = int(os.environ.get("TF_GLM_PF_GEMM_MIN_ROWS", "256") or 256)


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_glm_pf_dequant_v1", sources=[str(here / "pf_dequant.cpp"), str(here / "pf_dequant.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def dequant(q, out: torch.Tensor | None = None) -> torch.Tensor:
    """Q4 -> bf16 [npad, K] (the prefill kernel's B values)."""

    npad = q.scales.shape[1]
    if out is None:
        out = torch.empty((npad, q.k), dtype=torch.bfloat16, device=q.weight.device)
    _ext().dequant_q4(q.weight, q.scales, q.biases, out, q.k, q.gs)
    return out


@triton.jit
def _gemm(X, W, Y, M, N, K, sxm, swn, sym, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr,
          F32: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    group = GM * npn
    gid = pid // group
    fm = gid * GM
    gsz = min(npm - fm, GM)
    pm = fm + (pid % group) % gsz
    pn = (pid % group) // gsz
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    xm = rm < M
    xp = X + rm[:, None] * sxm + rk[None, :]
    wp = W + rn[None, :] * swn + rk[:, None]           # W [N, K]: the (k, n) tile, K contiguous
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(xp, mask=xm[:, None], other=0.0)
        b = tl.load(wp)
        acc = tl.dot(a, b, acc)                         # one chain over K, k16 steps in order
        xp += BK
        wp += BK
    yp = Y + rm[:, None] * sym + rn[None, :]
    ok = xm[:, None] & (rn[None, :] < N)
    if F32:
        tl.store(yp, acc, mask=ok)
    else:
        tl.store(yp, acc.to(tl.bfloat16), mask=ok)


# (BM, BN, BK, stages, warps, GM): the GB10 sweep's best at a rank's prompt shapes (micro/pf_gemm_check.py)
CONFIG = (128, 128, 32, 4, 4, 8)


def gemm(x: torch.Tensor, w: torch.Tensor, n: int, *, f32: bool, out: torch.Tensor | None = None,
         config: tuple | None = None) -> torch.Tensor:
    """x [M, K] bf16 @ w[:n]^T (w [npad, K] bf16) -> [M, n] fp32 or bf16."""

    M, K = x.shape
    if x.stride(1) != 1:
        x = x.contiguous()
    if out is None:
        out = torch.empty((M, n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    BM, BN, BK, stages, warps, GM = config or CONFIG
    if K % BK:
        raise ValueError(f"pf_gemm: K {K} is not a multiple of {BK}")
    grid = (triton.cdiv(M, BM) * triton.cdiv(n, BN),)
    _gemm[grid](x, w, out, M, n, K, x.stride(0), w.stride(0), out.stride(0), BM=BM, BN=BN, BK=BK, GM=GM, F32=f32,
                num_stages=stages, num_warps=warps)
    return out


class Cache:
    """The bf16 weights of the last few Q4s a prompt chunk used (by identity), each in a buffer reused once evicted.
    Every use records an event on the stream that used it; a buffer is rewritten only after its last use's event."""

    def __init__(self, slots: int = 8) -> None:
        self.slots = slots
        self.items: dict[int, list] = {}            # key -> [q, buffer, event of the last use]
        self.order: list[int] = []

    def get(self, q) -> tuple[torch.Tensor, list]:
        key = id(q)
        hit = self.items.get(key)
        npad = q.scales.shape[1]
        if hit is not None and hit[0] is q:
            self.order.remove(key)
            self.order.append(key)
            torch.cuda.current_stream().wait_event(hit[3])    # made (no-op on the stream that made it)
            return hit[1][:npad * q.k].view(npad, q.k), hit
        buf = None
        if len(self.order) >= self.slots:
            old = self.order.pop(0)
            _, buf, ev, _ = self.items.pop(old)
            if ev is not None:
                torch.cuda.current_stream().wait_event(ev)    # its last GEMM, on whatever stream, is done first
            if buf.numel() < npad * q.k:
                buf = None
        if buf is None:
            buf = torch.empty((npad * q.k,), dtype=torch.bfloat16, device=q.weight.device)
        w = buf[:npad * q.k].view(npad, q.k)
        dequant(q, w)
        made = torch.cuda.Event()
        made.record(torch.cuda.current_stream())
        item = [q, buf, None, made]
        self.items[key] = item
        self.order.append(key)
        return w, item

    @staticmethod
    def used(item) -> None:
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream())
        item[2] = ev


CACHE = Cache()


def prefill_matmul(x: torch.Tensor, q, *, f32: bool = False, out: torch.Tensor | None = None) -> torch.Tensor:
    """``qmm.prefill_matmul``'s bits through the bf16 weight and the GEMM."""

    w, item = CACHE.get(q)
    y = gemm(x, w, q.n, f32=f32, out=out)
    CACHE.used(item)
    return y
