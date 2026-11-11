#!/usr/bin/env python3
"""bench_prefill.py's measurement, guarded against other clients: before each prompt wait for an idle server, watch
/health while it runs, and repeat a prompt during which another request ran (a decoding stream beside it makes the
prompt fill in 1,024-row chunks at ~2,000 tok/s). Same prompts and seeds as bench_prefill.py.
  bench2.py --seed 777 --sizes 8192,16384,32768 --reps 5 --json out.json"""
import argparse, json, statistics, threading, time, urllib.request
import bench_prefill as B
import thermal

def health(url):
    try:
        with urllib.request.urlopen(url + "/health", timeout=5) as r:
            return json.load(r)
    except Exception:
        return None

def wait_idle(url, limit=300):
    t = time.time()
    while time.time() - t < limit:
        h = health(url)
        if h and h["requests_running"] == 0:
            return True
        time.sleep(2)
    return False

def guarded(url, model, text):
    thermal.guard()
    if not wait_idle(url):
        return None, "busy"
    seen = [0]; stop = threading.Event()
    def watch():
        while not stop.is_set():
            h = health(url)
            if h:
                seen[0] = max(seen[0], h["requests_running"])
            stop.wait(0.4)
    h0 = health(url); th = threading.Thread(target=watch, daemon=True); th.start()
    r = B.run(url, model, text)
    stop.set(); th.join()
    if r[0] > 8:                      # a long prompt: let the Sparks breathe before the next one (thermal)
        time.sleep(min(20.0, 0.5 * r[0]))
    h1 = health(url)
    others = (h1["requests_total"] - h0["requests_total"] - 1) if h0 and h1 else 0
    return r, ("" if seen[0] <= 1 and others <= 0 else f"another request ran (running {seen[0]}, +{others})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    ap.add_argument("--sizes", default="8192,16384,32768"); ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=777); ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = {"seed": a.seed, "results": [], "discarded": 0}
    wait_idle(a.url); B.run(a.url, a.model, B.prompt_text(200, a.seed - 1))
    for size in [int(s) for s in a.sizes.split(",")]:
        rates = []
        for rep in range(a.reps):
            seed = a.seed * 1000 + size * 10 + rep
            for attempt in range(4):
                r, why = guarded(a.url, a.model, B.prompt_text(size, seed))
                if r is not None and not why:
                    break
                out["discarded"] += 1
                print(f"size {size} rep {rep}: discarded ({why})", flush=True)
            else:
                continue
            ttft, total, usage = r
            pt = (usage or {}).get("prompt_tokens", 0)
            cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            rate = (pt - cached) / ttft if ttft else 0.0
            rates.append(rate)
            out["results"].append({"size": size, "rep": rep, "prompt_tokens": pt, "cached": cached, "ttft": ttft, "rate": rate})
        if rates:
            print(f"== size {size}: median {statistics.median(rates):,.0f} tok/s (min {min(rates):,.0f}, max {max(rates):,.0f}, n {len(rates)})", flush=True)
        else:
            print(f"== size {size}: no clean run", flush=True)
    thermal.temps()
    print(f"discarded {out['discarded']} runs other requests got into; {thermal.summary()}", flush=True)
    out["thermal"] = {"peak": thermal.peak, "mhz": thermal.clocks, "wait_s": thermal.waits[0]}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)

if __name__ == "__main__":
    main()
