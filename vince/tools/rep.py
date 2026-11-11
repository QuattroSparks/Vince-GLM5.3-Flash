#!/usr/bin/env python3
"""Repeated c1 prose runs, median and spread (rep.py [N=6] [label]):
  story   decode_check.py's prose: lighthouse story, T 0, thinking off, 400 tokens, tok/s over the whole request
  conc1   conc_bench.py's c1 decode: a different story a run (its topics), 256 tokens, whole request
  sdprose sparkDash's prose prompt, T 0, thinking off, 400 tokens, tok/s after the first token (Mia's table's protocol)
  sd07    the same at sparkDash's present defaults (T 0.7, thinking on)
  code    decode_check.py's code prompt, whole request"""
import json, statistics, sys, time
import dbench
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
label = sys.argv[2] if len(sys.argv) > 2 else "run"
TOPICS = ["a lighthouse keeper", "a mountain rescue", "a kidney surgeon", "a silver mine", "an orange grove",
          "a river ferry", "a theory exam", "a garden robot", "a valley flood", "an engine factory"]
def run(mode, prompt, n):
    r = dbench.wave([dbench.body_for(mode, prompt, n, {})])[0]
    t = r["usage"]["completion_tokens"]
    return t / (r["t1"] - r["t0"]), (t - 1) / (r["last"] - r["first"]), r["runtime"]
out = {}
dbench.wave([dbench.body_for("ours", dbench.PROMPTS["story"], 32, {})])
for name, mode, prompts, n, col in (
        ("story", "ours", [dbench.PROMPTS["story"]] * N, 400, 0),
        ("conc1", "ours", [f"Write a detailed story about {TOPICS[i % 10]} (version 4242-1-{i})." for i in range(N)], 256, 0),
        ("sdprose", "ours", [dbench.SD_PROSE] * N, 400, 1),
        ("sd07", "sd", [dbench.SD_PROSE] * N, 400, 1),
        ("code", "ours", [dbench.PROMPTS["lru"]] * max(3, N // 2), 400, 0)):
    v = [run(mode, p, n)[col] for p in prompts]
    out[name] = v
    print(f"[{label}] {name:8s} median {statistics.median(v):6.1f}  min {min(v):6.1f}  max {max(v):6.1f}  ({', '.join(f'{x:.1f}' for x in v)})", flush=True)
json.dump(out, open(f"rep-{label}.json", "w"), indent=1)
