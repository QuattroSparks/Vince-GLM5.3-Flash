"""Thermal guard for the benchmarks (2026-11-10: at a 2,300 MHz cap the Sparks reached 84-86 C and throttled under
back-to-back long prompts). ``guard()`` before every prompt: every rank's GPU temperature; at HOT or more on any,
wait until all are below COOL. ``peak`` keeps the highest reading seen; ``log`` every reading taken."""
import os, subprocess, time
from concurrent.futures import ThreadPoolExecutor

def _hosts():
    """This Spark and the three workers (ssh targets) of scripts/local.sh, in rank order."""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    out = subprocess.run(["bash", "-c", 'source scripts/local.sh >/dev/null 2>&1; echo "$WORKER $WORKER2 $WORKER3"'],
                         cwd=root, capture_output=True, text=True).stdout.split()
    return ["localhost"] + out

HOSTS = _hosts()
HOT, COOL = 75, 65     # the head's SW thermal slowdown counter moved at 78-82 C (2,190 MHz): stay under it
peak = [0] * len(HOSTS); waits = [0.0]; clocks = [0] * len(HOSTS)
Q = "nvidia-smi --query-gpu=temperature.gpu,clocks.current.graphics --format=csv,noheader,nounits"

def _one(h):
    try:
        cmd = ["bash", "-c", Q] if h == "localhost" else ["ssh", "-o", "ConnectTimeout=4", "-o", "BatchMode=yes", h, Q]
        t, c = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip().split(",")
        return int(t), int(c)
    except Exception:
        return -1, -1

def temps():
    with ThreadPoolExecutor(4) as ex:
        r = list(ex.map(_one, HOSTS))
    for i, (t, c) in enumerate(r):
        peak[i] = max(peak[i], t); clocks[i] = c
    return [t for t, _ in r]

def guard(say=print):
    if _slow0 is None:
        throttled_ms()
    t = temps()
    if max(t) >= HOT:
        t0 = time.time(); say(f"thermal guard: {t} C, waiting for < {COOL} C on every rank")
        while max(t) >= COOL and time.time() - t0 < 900:
            time.sleep(10); t = temps()
        waits[0] += time.time() - t0
        say(f"thermal guard: cooled to {t} C after {time.time() - t0:.0f} s")
    return t

QS = "nvidia-smi -q | awk '/SW Thermal Slowdown/ && /us/ {print $5}'"
_slow0 = None

def slowdown_us():
    """Each rank's cumulative SW thermal slowdown (microseconds), -1 if unreadable."""
    def one(h):
        try:
            cmd = ["bash", "-c", QS] if h == "localhost" else ["ssh", "-o", "ConnectTimeout=4", "-o", "BatchMode=yes", h, QS]
            return int(subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.split()[0])
        except Exception:
            return -1
    with ThreadPoolExecutor(4) as ex:
        return list(ex.map(one, HOSTS))

def throttled_ms():
    """SW thermal slowdown added on each rank since this process first asked (ms)."""
    global _slow0
    now = slowdown_us()
    if _slow0 is None:
        _slow0 = now
    return [round((a - b) / 1000) if a >= 0 and b >= 0 else -1 for a, b in zip(now, _slow0)]

def summary():
    return f"peak C {peak}, MHz {clocks}, cooling waits {waits[0]:.0f} s, thermal slowdown added ms {throttled_ms()}"
