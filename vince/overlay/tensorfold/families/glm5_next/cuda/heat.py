"""TF_HEAT_HIGH / TF_HEAT_LOW (degrees Celsius; unset: off): rank 0 holds a prompt's next chunk while its GPU is hot.

TensorFold's heat bands (upstream: Nemotron on CUDA and the GB10 runbook, from @Bizuayeu's GLM-5.3-Flash measurements
on three Sparks, #339 / #988), for this engine's multi-stream server: once this Spark's GPU reads above HIGH, a prompt
chunk that is due waits until it reads LOW or less. Only prompt chunks wait, and only while no stream is decoding
(a reply being written is never held); the other ranks wait for rank 0's next message as they do between any two
chunks. Tokens do not change: the same chunks run, later. A wait is cut at ``MOST`` seconds a chunk (under the
iteration watchdog's 300 s), and each wait is logged with the time held.

The reading is NVML's GPU temperature (what ``nvidia-smi`` shows); if NVML cannot be read, the hottest zone under
/sys/class/thermal. Read at most twice a second."""

from __future__ import annotations

import glob
import os
import time

MOST = 240.0          # seconds one chunk is held at most
EVERY = 0.5           # seconds between readings


def settings(env=None) -> tuple[float, float] | None:
    env = os.environ if env is None else env
    # TF_GLM_HEAT_HIGH / TF_GLM_HEAT_LOW are the same switches under the prefix a recipe forwards to every rank
    high = (env.get("TF_GLM_HEAT_HIGH", "") or env.get("TF_HEAT_HIGH", "")).strip()
    low = (env.get("TF_GLM_HEAT_LOW", "") or env.get("TF_HEAT_LOW", "")).strip()
    if high in ("", "0") and low in ("", "0"):
        return None
    try:
        h, l = float(high), float(low)
    except ValueError:
        h = l = -1.0
    if not 40.0 <= l < h <= 100.0:
        raise ValueError(f"TF_HEAT_HIGH / TF_HEAT_LOW: both, degrees Celsius, 40 <= LOW < HIGH <= 100, "
                         f"not {high!r} / {low!r}")
    return h, l


def _nvml():
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return lambda: float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    except Exception:
        return None


def _zones():
    paths = sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp"))

    def read() -> float:
        best = -1.0
        for p in paths:
            try:
                with open(p) as f:
                    best = max(best, int(f.read().strip()) / 1000.0)
            except (OSError, ValueError):
                pass
        return best

    return read if paths and read() > 0 else None


class Band:
    """``hold()``: whether a prompt chunk that is due should wait now (the caller sleeps a moment and asks again)."""

    def __init__(self, high: float, low: float, read=None, clock=time.monotonic) -> None:
        self.high, self.low, self.clock = high, low, clock
        self.source = "given"
        if read is None:
            read, self.source = _nvml(), "the GPU's temperature (NVML)"
            if read is None:
                read, self.source = _zones(), "the hottest thermal zone"
        self.read = read
        self.hot = False
        self.since = self.at = 0.0
        self.last = self.peak = -1.0
        self.waits, self.held_s = 0, 0.0

    @classmethod
    def from_env(cls, env=None) -> "Band | None":
        s = settings(env)
        if s is None:
            return None
        band = cls(*s)
        if band.read is None:
            print("[tensorfold] TF_HEAT_HIGH / TF_HEAT_LOW: no temperature can be read here, so no chunk waits",
                  flush=True)
            return None
        print(f"[tensorfold] heat bands: a prompt chunk waits while {band.source} is above {band.high:g} C, until "
              f"{band.low:g} C or less (at most {MOST:g} s a chunk; now {band.temp():g} C)", flush=True)
        return band

    def temp(self) -> float:
        now = self.clock()
        if self.last < 0 or now - self.at >= EVERY:
            try:
                self.last = float(self.read())
            except Exception:
                self.last = -1.0                          # unreadable: never hold on a guess
            self.at = now
        return self.last

    def hold(self) -> bool:
        t, now = self.temp(), self.clock()
        if not self.hot:
            if t <= self.high:
                return False
            self.hot, self.since, self.peak = True, now, t
            return True
        self.peak = max(self.peak, t)
        if 0 <= t > self.low and now - self.since < MOST:
            return True
        held = now - self.since
        self.hot = False
        self.waits += 1
        self.held_s += held
        why = "" if 0 <= t <= self.low else " (the most a chunk waits)" if t >= 0 else " (the reading failed)"
        print(f"[tensorfold] heat: a prompt chunk waited {held:.1f} s, {self.peak:g} -> {t:g} C{why}; "
              f"heat_wait_s {self.held_s:.1f} in {self.waits} waits", flush=True)
        return False
