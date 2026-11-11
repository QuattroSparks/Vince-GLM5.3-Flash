#!/usr/bin/env python3
"""Decode speed sanity check (one request at a time, temperature 0, thinking off, 400 tokens): tok/s of completion
tokens over the whole request (prefill of a short prompt included), median of 3 per prompt."""
import json
import statistics
import time
import urllib.request

PROMPTS = {
    "prose": "Write a long, detailed story about a lighthouse keeper who discovers a secret map.",
    "code": "Write a Python module implementing a thread-safe LRU cache with TTL support, with docstrings and tests.",
}


def gen(prompt: str, n: int = 400):
    body = json.dumps({"model": "GLM-5.3-Flash-EXL3", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": n, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    d = json.load(urllib.request.urlopen(req, timeout=600))
    return d["usage"]["completion_tokens"] / (time.perf_counter() - t)


for name, p in PROMPTS.items():
    rates = [gen(p) for _ in range(3)]
    print(f"decode {name}: median {statistics.median(rates):.1f} tok/s ({', '.join(f'{r:.1f}' for r in rates)})",
          flush=True)
