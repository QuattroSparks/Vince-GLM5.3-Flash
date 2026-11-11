"""Decode work (2026-11-10): host-side settings for decode rounds that change no reply's bits.

TF_GLM_DEC_CPUS (off by default): the CPUs this rank's process may run on, set before any thread starts so every
thread inherits it (the engine thread that drives the rounds, the RoCE proxy threads that post each all-gather's RDMA
writes and relay the ring's stripes, CUDA's own threads). A GB10 has ten big cores (Cortex-X925, cpu_capacity ~1000)
and ten smaller ones (Cortex-A725, ~720); left alone, Linux keeps the round loop and the proxy's polling thread mostly
on the small ones. ``big``: the cores whose cpu_capacity is at least 90% of the largest; or a list like ``5-9,15-19``.
The idea of keeping the proxy thread on the big cores is Christopher Owen's (christopherowen/dgx-spark-networking,
Apache-2.0); here it is the whole process, by affinity mask, with no code of his."""

from __future__ import annotations

import glob
import os
import re


def big_cpus(root: str = "/sys/devices/system/cpu") -> list[int]:
    caps = {}
    for path in glob.glob(f"{root}/cpu[0-9]*/cpu_capacity"):
        try:
            caps[int(re.search(r"cpu(\d+)/cpu_capacity$", path).group(1))] = int(open(path).read())
        except (OSError, ValueError, AttributeError):
            continue
    if not caps:
        return []
    top = max(caps.values())
    return sorted(c for c, v in caps.items() if v >= 0.9 * top)


def parse_cpus(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return sorted(set(out))


def apply_affinity(env=None, log=print) -> list[int] | None:
    """Apply TF_GLM_DEC_CPUS to this process (None: off or nothing to do). Never fatal: a bad value or a refused
    mask leaves the process as it was and says so."""

    spec = ((os.environ if env is None else env).get("TF_GLM_DEC_CPUS", "") or "").strip().lower()
    if spec in ("", "0", "off"):
        return None
    try:
        cpus = big_cpus() if spec == "big" else parse_cpus(spec)
        allowed = sorted(set(cpus) & set(os.sched_getaffinity(0)))
        if not allowed:
            raise ValueError(f"no usable CPU in {cpus}")
        os.sched_setaffinity(0, allowed)
        for tid in os.listdir("/proc/self/task"):          # threads that already exist (new ones inherit)
            try:
                os.sched_setaffinity(int(tid), allowed)
            except (OSError, ValueError):
                pass
    except (OSError, ValueError) as exc:
        log(f"[tensorfold] TF_GLM_DEC_CPUS={spec!r} not applied: {exc}", flush=True)
        return None
    log(f"[tensorfold] TF_GLM_DEC_CPUS={spec}: this rank's threads run on CPUs {','.join(map(str, allowed))}", flush=True)
    return allowed
