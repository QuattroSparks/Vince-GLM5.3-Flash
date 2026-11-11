#!/usr/bin/env python3
"""Kept-state check for tail_merge: the reply of a follow-up that resumes from a first prompt's kept grid-point state
must be the same whether that state was kept by the two-chunk run or by the merged chunk.
  reuse2.py TAG first            send the first prompts (run with no_reuse=1 so each is prefilled cold and kept anew)
  reuse2.py TAG follow OUT.json  send the follow-ups (no_reuse=0), save replies and cached tokens
  reuse2.py TAG flood            40 small distinct prompts: pushes every earlier kept state out of the cache
  reuse2.py cmp A.json B.json"""
import json, sys
import ref_test as R
import bench_prefill as B
import zlib
def _h(*a): return zlib.crc32(repr(a).encode()) % 10**9   # (hash() of a str changes every process)
SIZES = [150, 600, 1500, 3050, 5000, 9100, 12100]
def first(tag, w):  return B.prompt_text(w, _h(tag, w))
def follow(tag, w): return first(tag, w) + "\n\nNow continue the notes with: " + B.prompt_text(90, _h(tag, w, 2))
if sys.argv[1] == "cmp":
    a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    bad = [w for w in a if a[w]["content"] != b[w]["content"]]
    print("KEPT STATES EQUIVALENT" if not bad else f"{len(bad)} DIFFERENT: {bad}", "| cached:", [a[w]["cached"] for w in a], "vs", [b[w]["cached"] for w in b])
    sys.exit(1 if bad or [a[w]["cached"] for w in a] != [b[w]["cached"] for w in b] else 0)
tag, mode = sys.argv[1], sys.argv[2]
if mode == "flood":
    import time
    for i in range(40):
        R.ask([{"role": "user", "content": B.prompt_text(120, int(time.time() * 1000) % 10**9 + i)}], 2, False)
    print("flooded"); sys.exit(0)
out = {}
for w in SIZES:
    if mode == "first":
        r = R.ask([{"role": "user", "content": first(tag, w)}], 8, False)
    else:
        r = R.ask([{"role": "user", "content": follow(tag, w)}], 40, False)
    u = r["usage"] or {}
    out[str(w)] = {"content": r["content"], "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0), "tokens": u.get("prompt_tokens")}
    print(mode, w, "prompt", out[str(w)]["tokens"], "cached", out[str(w)]["cached"], flush=True)
if mode == "follow":
    json.dump(out, open(sys.argv[3], "w"), indent=1)
