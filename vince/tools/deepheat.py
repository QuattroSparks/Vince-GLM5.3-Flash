#!/usr/bin/env python3
"""Back-to-back long cold prompts, each followed by a 300-token reply: decode speed at depth, the head's GPU
temperature through the run, and whether the heat bands held prompt chunks (the server's log says how long).
Usage: deepheat.py [--words 90000] [--n 3] [--json out.json]"""
import argparse, json, random, subprocess, threading, time, http.client

MODEL = "GLM-5.3-Flash-EXL3"
WORDS = ("lighthouse harbor map keeper storm lantern tide granite signal ledger north cable winter salt engine copper "
         "glass ladder archive morning river window stone garden letter bridge candle market silver thread").split()


def temp():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip() or -1)


def run(words, seed):
    rnd = random.Random(seed)
    text = " ".join(rnd.choice(WORDS) for _ in range(words))
    body = dict(model=MODEL, messages=[{"role": "user", "content": text + "\n\nIgnore the words above. Write a long, "
                "detailed story about a lighthouse keeper who discovers a secret map."}], max_tokens=300, temperature=0,
                stream=True, stream_options={"include_usage": True}, chat_template_kwargs={"enable_thinking": False})
    c = http.client.HTTPConnection("127.0.0.1", 8000, timeout=1800)
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    first = last = None
    usage = None
    for line in c.getresponse():
        if not line.startswith(b"data: "):
            continue
        d = line[6:].strip()
        if d == b"[DONE]":
            break
        j = json.loads(d)
        if ((j.get("choices") or [{}])[0].get("delta") or {}).get("content"):
            last = time.perf_counter()
            first = first or last
        usage = j.get("usage") or usage
    pt, ct = usage["prompt_tokens"], usage["completion_tokens"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return dict(prompt=pt, cached=cached, ttft=first - t0, prefill=(pt - cached) / (first - t0),
                decode=(ct - 1) / (last - first))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=int, default=90000)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--json")
    a = ap.parse_args()
    temps, stop = [], threading.Event()

    def watch():
        while not stop.is_set():
            temps.append(temp())
            time.sleep(1)
    threading.Thread(target=watch, daemon=True).start()
    out = []
    base = int(time.time())
    for i in range(a.n):
        k = len(temps)
        r = run(a.words, base + i)
        r["peak_c"] = max(temps[k:] or [-1])
        out.append(r)
        print(f"run {i + 1}: {r['prompt']} tokens (cached {r['cached']}), first token {r['ttft']:.1f} s = "
              f"{r['prefill']:.0f} tok/s, decode at depth {r['decode']:.1f} tok/s, head peak {r['peak_c']} C", flush=True)
    stop.set()
    print(f"head GPU: start {temps[0]} C, peak {max(temps)} C, end {temps[-1]} C")
    if a.json:
        json.dump(dict(runs=out, temps=temps), open(a.json, "w"))


if __name__ == "__main__":
    main()
