#!/usr/bin/env python3
"""Same-bits check on short prompts with every kind of last chunk: greedy replies (48 tokens) to fixed prompts of
~60 to ~17k tokens whose lengths end at different places past the 64-token grid.  short_ref.py save|check FILE"""
import json, sys
import ref_test as R
import bench_prefill as B
SIZES = [20, 60, 100, 200, 230, 500, 1000, 1010, 1990, 2000, 2030, 3000, 3020, 3030, 4000, 4090, 5000, 6000, 8000,
         9000, 12290, 12300, 14000, 16370, 16400]
def main():
    mode, path = sys.argv[1], sys.argv[2]
    got = {}
    for w in SIZES:
        text = B.prompt_text(w, 424200 + w) + "\n\nIn one sentence, which words appear most often above?"
        r = R.ask([{"role": "user", "content": text}], 48, False)
        got[str(w)] = {"content": r["content"], "reasoning": r["reasoning"], "tokens": (r["usage"] or {}).get("prompt_tokens")}
    if mode == "save":
        json.dump(got, open(path, "w"), indent=1); print("saved", path, [v["tokens"] % 64 for v in got.values()]); return
    ref = json.load(open(path))
    bad = [w for w in ref if (ref[w]["content"], ref[w]["reasoning"]) != (got[w]["content"], got[w]["reasoning"])]
    print("SHORT ALL IDENTICAL" if not bad else f"SHORT {len(bad)} DIFFERENT: {bad}")
    sys.exit(1 if bad else 0)
main()
