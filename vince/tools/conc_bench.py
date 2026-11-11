#!/usr/bin/env python3
"""Concurrency scaling, like pico-dgx-ds4's c1/c4/c8/c16 tables (distinct prompts, never repeated):
  prefill: c requests at once, each a different ~8k-token prompt, max_tokens 1 -> aggregate prompt tok/s
  decode:  c requests at once, each a different short prompt, 256 tokens (temperature 0, thinking off) -> aggregate
           completion tok/s (time to first token included)
Usage: conc_bench.py [--cs 1,2,4,8] [--json out.json]"""

import argparse
import json
import random
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000"
MODEL = "GLM-5.3-Flash-EXL3"
WORDS = ("river garden medicine doctor patient kidney ultrasound valley mountain engine silver orange theory the of "
         "and to in is that for it as was with be by on not he this are or his from at which but have an they you "
         "were her she there been one all we their has would when if so no will more can who out about").split()
TOPICS = ["a lighthouse keeper", "a mountain rescue", "a kidney surgeon", "a silver mine", "an orange grove",
          "a river ferry", "a theory exam", "a garden robot", "a valley flood", "an engine factory", "a night train",
          "a winter market", "a desert well", "a city library", "a jazz club", "a fishing boat"]


def post(body, timeout=1800):
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def prefill_prompt(seed, n_words=8000):
    r = random.Random(seed)
    return f"Notes {seed}-{r.random():.9f}.\n" + " ".join(r.choice(WORDS) for _ in range(n_words)) + "\n\nOne word?"


def run_group(bodies):
    out = [None] * len(bodies)

    def one(i):
        out[i] = post(bodies[i])

    t = time.perf_counter()
    ts = [threading.Thread(target=one, args=(i,)) for i in range(len(bodies))]
    [x.start() for x in ts]
    [x.join() for x in ts]
    return time.perf_counter() - t, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cs", default="1,2,4,8")
    ap.add_argument("--json", default="")
    ap.add_argument("--seed", type=int, default=4242)
    a = ap.parse_args()
    res = {"prefill": {}, "decode": {}}
    base = a.seed
    for c in [int(x) for x in a.cs.split(",")]:
        bodies = [{"model": MODEL, "messages": [{"role": "user", "content": prefill_prompt(base * 100 + c * 10 + i)}],
                   "max_tokens": 1, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
                  for i in range(c)]
        dt, out = run_group(bodies)
        toks = sum(o["usage"]["prompt_tokens"] - ((o["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
                                                   or 0) for o in out)
        res["prefill"][c] = toks / dt
        print(f"prefill c{c}: {toks} prompt tokens in {dt:.2f} s = {toks / dt:,.0f} tok/s aggregate", flush=True)
    for c in [int(x) for x in a.cs.split(",")]:
        bodies = [{"model": MODEL, "messages": [{"role": "user", "content":
                   f"Write a detailed story about {TOPICS[(i + c) % len(TOPICS)]} (version {base}-{c}-{i})."}],
                   "max_tokens": 256, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
                  for i in range(c)]
        dt, out = run_group(bodies)
        toks = sum(o["usage"]["completion_tokens"] for o in out)
        res["decode"][c] = toks / dt
        print(f"decode  c{c}: {toks} tokens in {dt:.2f} s = {toks / dt:,.1f} tok/s aggregate", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
