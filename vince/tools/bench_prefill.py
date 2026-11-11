#!/usr/bin/env python3
"""Prefill benchmark for the GLM server: unique random-word prompts (no prompt-cache hits), max_tokens 1,
streamed; prefill tok/s = prompt_tokens / time to first token. Usage:
  bench_prefill.py [--url http://127.0.0.1:8000] [--sizes 8192,16384,32768] [--reps 3] [--json out.json]"""

import argparse
import json
import random
import statistics
import time
import urllib.request

WORDS = ("the of and to in is that for it as was with be by on not he this are or his from at which but have an they "
         "you were her she there been one all we their has would when if so no will more can who out about other into "
         "time some could them only new these then do two may first any like now my such its over than most also "
         "after where made many before must through back years much way well down should because each just those "
         "people how too little state good very make world still own see men work long here get both between life "
         "being under never day same another know while last might us great old year off come since against go came "
         "right used take three himself few house use during without again place around however home small found mrs "
         "thought went say part once general high upon school every don does got united left number course war until "
         "always away something fact though water less public put think almost hand enough far took head yet "
         "government system better set told nothing night end why called didn eyes find going look asked later knew "
         "river garden medicine doctor patient kidney ultrasound valley mountain engine silver orange theory").split()


def prompt_text(n_words: int, seed: int) -> str:
    r = random.Random(seed)
    head = f"Session {seed}-{r.random():.12f}. Read the following notes and reply with one word.\n\n"
    return head + " ".join(r.choice(WORDS) for _ in range(n_words))


def run(url: str, model: str, text: str, timeout: float = 900.0):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 1,
                       "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            ch = obj.get("choices") or []
            if ttft is None and ch and ((ch[0].get("delta") or {}).get("content") or (ch[0].get("delta") or {}).get("reasoning_content") or ch[0].get("finish_reason")):
                ttft = time.perf_counter() - t0
            if obj.get("usage"):
                usage = obj["usage"]
    total = time.perf_counter() - t0
    return (ttft or total), total, usage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    ap.add_argument("--sizes", default="8192,16384,32768")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=int(time.time()))
    ap.add_argument("--json", default="")
    ap.add_argument("--words-per-token", type=float, default=1.0)
    a = ap.parse_args()
    out = {"url": a.url, "seed": a.seed, "results": []}
    # warm-up (small, unique)
    run(a.url, a.model, prompt_text(200, a.seed - 1))
    for size in [int(s) for s in a.sizes.split(",")]:
        rates = []
        for rep in range(a.reps):
            seed = a.seed * 1000 + size * 10 + rep
            ttft, total, usage = run(a.url, a.model, prompt_text(int(size * a.words_per_token), seed))
            pt = (usage or {}).get("prompt_tokens", 0)
            cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            rate = (pt - cached) / ttft if ttft else 0.0
            rates.append(rate)
            print(f"size {size}: prompt_tokens {pt} cached {cached} ttft {ttft:.3f}s -> {rate:,.0f} tok/s", flush=True)
            out["results"].append({"size": size, "rep": rep, "prompt_tokens": pt, "cached": cached, "ttft": ttft,
                                   "rate": rate})
        print(f"== size {size}: median {statistics.median(rates):,.0f} tok/s (min {min(rates):,.0f}, max {max(rates):,.0f})",
              flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
