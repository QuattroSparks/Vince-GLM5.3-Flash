#!/usr/bin/env python3
"""Decode bench with both yardsticks on FIXED prompts (so a same-bits change shows as fewer ms a round or fewer rounds):

  sd   = sparkDash's DecodeBench protocol (server/collectors/DecodeBench.js): its own prompts, temperature 0.7, top_p 1,
         thinking on, max_tokens = min_tokens = 400, a 32-token warm-up, tok/s = (completion_tokens - 1) / (last - first
         token); here with a fixed seed so runs repeat.
  ours = decode_check.py's: temperature 0, thinking off, 400 tokens, tok/s over the whole request (prefill included);
         the post-first-token rate is printed beside it.

Per request also the server's own numbers: rounds, drafted, accepted, tokens a round, ms a round (decode seconds from
runtime.tokens_per_second). Usage: dbench.py [--modes sd,ours] [--kinds prose,code,story,lru] [--cs 1,2] [--reps 2]
[--json out.json] [--extra '{"draft": false}']"""
import argparse
import json
import statistics
import threading
import time
import http.client

MODEL = "GLM-5.3-Flash-EXL3"
SD_PROSE = ("Write a detailed step-by-step explanation of how a hash map works, "
            "including collision handling, resizing, and time complexity. Be thorough.")
TAIL = ("Output only Python source. No comments, no docstrings, no markdown fences. "
        "Then add tests and the helpers this needs. Keep writing code.")
PROMPTS = {
    "prose": SD_PROSE,
    "code": None,       # filled from sparkDash's llmPrompts.js at start (codeTask binary_search / merge_sort);
                        # a checkout of github.com/MiaAI-Lab/sparkDash in ~/sparkDash or SPARKDASH_DIR, and node
    "story": "Write a long, detailed story about a lighthouse keeper who discovers a secret map.",
    "lru": "Write a Python module implementing a thread-safe LRU cache with TTL support, with docstrings and tests.",
}


def sd_code_prompts():
    """sparkDash's code tasks, read from its source so the text is exactly its own."""
    import os
    import re
    import subprocess
    path = os.path.join(os.environ.get("SPARKDASH_DIR", os.path.expanduser("~/sparkDash")), "src/shared/llmPrompts.js")
    js = open(path).read()
    out = subprocess.run(["node", "--input-type=module", "-e",
                          "import(process.argv[1]).then(m=>console.log(JSON.stringify("
                          "[m.DECODE_CODE_TASKS, m.DECODE_CODE_WARMUP_PROMPT])))", path], capture_output=True, text=True)
    if out.returncode == 0:
        return json.loads(out.stdout)
    raise SystemExit("cannot read sparkDash's code prompts: " + out.stderr[:300] + str(len(js)) + str(re))


def stream(body, res, i):
    c = http.client.HTTPConnection("127.0.0.1", 8000, timeout=900)
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    first = last = None
    usage = spec = runtime = None
    n_events = 0
    for line in r:
        if not line.startswith(b"data: "):
            continue
        d = line[6:].strip()
        if d == b"[DONE]":
            break
        j = json.loads(d)
        ch = (j.get("choices") or [{}])[0].get("delta") or {}
        if ch.get("content") or ch.get("reasoning_content") or ch.get("reasoning") or ch.get("tool_calls"):
            now = time.perf_counter()
            first = first or now
            last = now
            n_events += 1
        usage = j.get("usage") or usage
        spec = j.get("speculative") or spec
        runtime = j.get("tensorfold") or j.get("runtime") or runtime
    t1 = time.perf_counter()
    res[i] = dict(t0=t0, first=first, last=last, t1=t1, usage=usage, spec=spec, runtime=runtime, events=n_events)


def total():
    import urllib.request
    return json.load(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=10))["requests_total"]


def wave(bodies, tries=4):
    """The bodies at once; a wave other requests got into (the server's requests_total moved by more than ours) is
    run again, up to ``tries`` times (then kept, with CONTAMINATED printed)."""
    for k in range(tries):
        n0 = total()
        res = [None] * len(bodies)
        ts = [threading.Thread(target=stream, args=(b, res, i)) for i, b in enumerate(bodies)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        time.sleep(0.05)
        if total() - n0 == len(bodies):
            return res
        print(f"  (other requests ran during this wave: {'again' if k + 1 < tries else 'CONTAMINATED, kept'})", flush=True)
        time.sleep(3)
    return res


def body_for(mode, prompt, n, extra):
    b = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "stream": True,
         "stream_options": {"include_usage": True}}
    if mode == "sd":
        b.update(temperature=0.7, top_p=1, min_tokens=n, seed=1234, chat_template_kwargs={"enable_thinking": True})
    elif mode == "sdoff":
        b.update(temperature=0.7, top_p=1, min_tokens=n, seed=1234, chat_template_kwargs={"enable_thinking": False})
    else:
        b.update(temperature=0, chat_template_kwargs={"enable_thinking": False})
    b.update(extra)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="sd,ours")
    ap.add_argument("--kinds", default="prose,code,story,lru")
    ap.add_argument("--cs", default="1,2")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--json", default="")
    ap.add_argument("--extra", default="{}")
    a = ap.parse_args()
    extra = json.loads(a.extra)
    tasks, warm = sd_code_prompts()
    PROMPTS["code"] = tasks[0]
    rows = []
    for mode in a.modes.split(","):
        for kind in a.kinds.split(","):
            for c in [int(x) for x in a.cs.split(",")]:
                if c == 1:
                    prompts = [PROMPTS[kind]]
                elif kind == "code":
                    prompts = tasks[:c]
                else:
                    prompts = [f"Request ID: decode-{i + 1}\n{PROMPTS[kind]}" for i in range(c)]
                wave([body_for(mode, warm if kind == "code" else PROMPTS[kind], 32, extra)])      # warm-up, as sparkDash
                reps = []
                for _ in range(a.reps):
                    res = wave([body_for(mode, p, a.n, extra) for p in prompts])
                    toks = [r["usage"]["completion_tokens"] for r in res]
                    post = [(t - 1) / (r["last"] - r["first"]) for t, r in zip(toks, res)]
                    whole = [t / (r["t1"] - r["t0"]) for t, r in zip(toks, res)]
                    agg_post = sum(t - 1 for t in toks) / (max(r["last"] for r in res) - min(r["first"] for r in res))
                    agg_whole = sum(toks) / (max(r["t1"] for r in res) - min(r["t0"] for r in res))
                    rt = [r["runtime"] or {} for r in res]
                    sp = rt
                    rounds = sum(s.get("rounds", 0) for s in sp)
                    dec_s = [x["decode_s"] for x in rt if x.get("decode_s")]
                    reps.append(dict(tokens=toks, post=post, whole=whole, agg_post=agg_post, agg_whole=agg_whole,
                                     rounds=rounds, drafted=sum(s.get("drafted", 0) for s in sp),
                                     accepted=sum(s.get("accepted", 0) for s in sp),
                                     ttft=[r["first"] - r["t0"] for r in res],
                                     round_ms=(1e3 * max(dec_s) / (rounds / c)) if dec_s and rounds else None,
                                     sha=[x.get("token_sha") for x in rt]))
                med = lambda k: statistics.median(r[k] for r in reps)
                r0 = reps[-1]
                row = dict(mode=mode, kind=kind, c=c, agg_post=med("agg_post"), agg_whole=med("agg_whole"),
                           per_post=statistics.median(statistics.mean(r["post"]) for r in reps),
                           rounds=r0["rounds"], tokens=sum(r0["tokens"]), drafted=r0["drafted"], accepted=r0["accepted"],
                           round_ms=r0["round_ms"], ttft_ms=1e3 * statistics.mean(r0["ttft"]), sha=r0["sha"], reps=reps)
                rows.append(row)
                tpr = row["tokens"] / row["rounds"] if row["rounds"] else 0
                dpr = row["drafted"] / row["rounds"] if row["rounds"] else 0
                acc = row["accepted"] / row["drafted"] if row["drafted"] else 0
                print(f"{mode:5s} {kind:5s} c{c}: post-first {row['agg_post']:6.1f} agg ({row['per_post']:6.1f}/req), "
                      f"whole {row['agg_whole']:6.1f} | rounds {row['rounds']:4d} tok/round {tpr:.2f} drafted/round "
                      f"{dpr:.2f} acc {acc:.2f} round ms {row['round_ms'] or 0:.1f} ttft {row['ttft_ms']:.0f} ms "
                      f"sha {','.join((s or '?')[:6] for s in row['sha'])}", flush=True)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
