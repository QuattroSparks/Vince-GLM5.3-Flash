#!/usr/bin/env python3
"""Copy-heavy decode bench: replies that repeat a long file (what an agent's file rewrite looks like), where the
verify-window width (TF_GLM_MAX_ROWS / TF_GLM_COPY_MAX) decides the speed. Temperature 0, thinking off.

  quote = repeat the file verbatim        edit = repeat it with one identifier renamed
  story / code = dbench's ordinary prompts, as the no-regression check

Prints tok/s after the first token and a hash of each reply (same hash = same reply).
Usage: editbench.py [--reps 3] [--json out.json] [--tokens 1500] [--file PATH] [--rename OLD:NEW]

The file defaults to one of this recipe's overlay sources. The saved figures (vince/results/rewrite-*.json) used the
first 230 lines of TensorFold's families/glm5_next/cuda/copy_drafts.py with CopySettings renamed to CopyConfig."""
import argparse
import hashlib
import json
import statistics
import time
import http.client
from pathlib import Path

MODEL = "GLM-5.3-Flash-EXL3"
SRC = Path(__file__).resolve().parents[1] / "overlay/tensorfold/families/glm5_next/cuda/decode.py"


def prompts(src=SRC, rename="Engine:Runner"):
    text = "".join(open(src).readlines()[:230])
    old, new = rename.split(":", 1)
    return {
        "quote": "Repeat the following Python file exactly, character for character, with no commentary and no "
                 "markdown fences.\n\n" + text,
        "edit": f"Rewrite the following Python file with the name {old} renamed to {new} everywhere. "
                "Change nothing else. Output only the file, no commentary, no markdown fences.\n\n" + text,
        "story": "Write a long, detailed story about a lighthouse keeper who discovers a secret map.",
        "code": "Write a Python module implementing a thread-safe LRU cache with TTL support, with docstrings and tests.",
    }


def run(prompt, n):
    body = dict(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=n, temperature=0, stream=True,
                stream_options={"include_usage": True}, chat_template_kwargs={"enable_thinking": False})
    c = http.client.HTTPConnection("127.0.0.1", 8000, timeout=900)
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    first = last = None
    usage = spec = None
    text = []
    for line in c.getresponse():
        if not line.startswith(b"data: "):
            continue
        d = line[6:].strip()
        if d == b"[DONE]":
            break
        j = json.loads(d)
        delta = (j.get("choices") or [{}])[0].get("delta") or {}
        if delta.get("content"):
            now = time.perf_counter()
            first = first or now
            last = now
            text.append(delta["content"])
        usage = j.get("usage") or usage
        spec = j.get("speculative") or spec
    out = "".join(text)
    done = (usage or {}).get("completion_tokens", 0)
    rate = (done - 1) / (last - first) if first and last and last > first else 0.0
    return dict(tokens=done, rate=rate, ttft=(first or t0) - t0, spec=spec,
                hash=hashlib.sha256(out.encode()).hexdigest()[:12])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=1500)
    ap.add_argument("--kinds", default="quote,edit,story,code")
    ap.add_argument("--json")
    ap.add_argument("--file", default=str(SRC))
    ap.add_argument("--rename", default="Engine:Runner")
    a = ap.parse_args()
    ps = prompts(a.file, a.rename)
    run(ps["story"], 32)                                      # warm-up
    res = {}
    for kind in a.kinds.split(","):
        n = a.tokens if kind in ("quote", "edit") else 400
        runs = [run(ps[kind], n) for _ in range(a.reps)]
        rates = [r["rate"] for r in runs]
        hashes = sorted({r["hash"] for r in runs})
        res[kind] = dict(median=statistics.median(rates), rates=rates, tokens=runs[0]["tokens"], hashes=hashes,
                         spec=runs[-1]["spec"])
        print(f"{kind:6s} {statistics.median(rates):7.1f} tok/s  ({' '.join(f'{x:.1f}' for x in rates)})  "
              f"{runs[0]['tokens']} tokens  reply {','.join(hashes)}  {json.dumps(runs[-1]['spec'])[:160]}", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
