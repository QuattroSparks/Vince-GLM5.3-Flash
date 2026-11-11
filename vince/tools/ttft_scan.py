#!/usr/bin/env python3
"""TTFT against prompt length, to split a short prompt's time: ttft_scan.py OUT.json WORDS[,WORDS..] [reps]
Each size once to warm its shape, then `reps` clean runs (idle before each, overlapped runs repeated)."""
import json, statistics, sys
import bench_prefill as B, bench2
url, model = "http://127.0.0.1:8000", "GLM-5.3-Flash-EXL3"
sizes = [int(x) for x in sys.argv[2].split(",")]; reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
out = []
for w in sizes:
    ts = []; pt = 0
    for rep in range(reps + 1):
        for attempt in range(4):
            r, why = bench2.guarded(url, model, B.prompt_text(w, 555000 + w * 10 + rep))
            if r is not None and not why: break
        else: continue
        ttft, total, usage = r; pt = (usage or {}).get("prompt_tokens", 0)
        if rep: ts.append(ttft)
    if ts:
        m = statistics.median(ts)
        print(f"words {w:6d} tokens {pt:6d} (mod 64 = {pt % 64:2d}): ttft median {m*1000:7.1f} ms (min {min(ts)*1000:7.1f}) = {pt/m:,.0f} tok/s", flush=True)
        out.append({"words": w, "tokens": pt, "ttft": ts})
import thermal
thermal.temps(); print("thermal:", thermal.summary(), flush=True)
json.dump({"rows": out, "thermal": {"peak": thermal.peak, "mhz": thermal.clocks, "wait_s": thermal.waits[0]}}, open(sys.argv[1], "w"), indent=1)
