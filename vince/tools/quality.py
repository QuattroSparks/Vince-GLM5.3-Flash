#!/usr/bin/env python3
"""Quality suite for prompt-precision variants of GLM-5.3-Flash, with ground truth (greedy, thinking off):
  1. arith: 60 multi-step arithmetic word problems (exact integer answers; short prompts)
  2. recall: code words from long documents (8k-64k tokens; each note's word and section are known)
  3. flips: how far each greedy reply stays identical to a reference run's (characters of common prefix)
Usage: quality.py run OUT.json [--ref REF.json]   (prints accuracy; with --ref, flips against it)"""

import json
import random
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000"
MODEL = "GLM-5.3-Flash-EXL3"
NAMES = ["Ada", "Bruno", "Chiara", "Dario", "Elena", "Fabio", "Giulia", "Marco", "Sofia", "Luca"]
ITEMS = ["apples", "books", "coins", "stamps", "marbles", "pencils", "tickets", "shells"]


def ask(content, max_tokens=64, thinking=False):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
                       "temperature": 0, "chat_template_kwargs": {"enable_thinking": thinking}}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["choices"][0]["message"].get("content") or ""


def arith_problems(n=60, seed=5):
    r = random.Random(seed)
    out = []
    for i in range(n):
        a, b, c, d = r.randint(12, 90), r.randint(2, 9), r.randint(3, 40), r.randint(2, 6)
        who, what = r.choice(NAMES), r.choice(ITEMS)
        total = a * b - c
        ans = total // d if total % d == 0 else None
        if ans is None:
            c = c + total % d
            total = a * b - c
            ans = total // d
        q = (f"{who} has {b} boxes with {a} {what} in each box. {who} gives away {c} {what}, then shares the rest "
             f"equally among {d} friends. How many {what} does each friend get? Answer with the number only.")
        out.append((f"arith{i}", q, str(ans)))
    return out


WORDS = ("river garden medicine doctor patient kidney ultrasound valley mountain engine silver orange theory "
         "the of and to in is that for it as was with be by on not he this are or his from at which but have an "
         "they you were her she there been one all we their has would when if so no will more can who out about").split()


def recall_docs(sizes=(8000, 16000, 32000, 64000), seed=9):
    out = []
    for k, n_words in enumerate(sizes):
        r = random.Random(seed + k)
        lines, notes = [], {}
        sec = 0
        for i in range(0, n_words, 40):
            lines.append(" ".join(r.choice(WORDS) for _ in range(40)))
            if i % 2000 == 0:
                word = f"{r.choice(WORDS)}-{r.randint(100, 999)}"
                notes[sec] = word
                lines.append(f"Note: the code word for section {sec} is {word}.")
                sec += 1
        doc = "\n".join(lines)
        picks = r.sample(sorted(notes), min(5, len(notes)))
        for s in picks:
            out.append((f"recall{n_words}-{s}", doc + f"\n\nWhat is the code word for section {s}? Answer with the code word "
                        "only.", notes[s]))
    return out


def run(items, par=4):
    got = {}
    lock = threading.Lock()
    queue = list(items)

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                name, q, _ = queue.pop(0)
            a = ask(q, 2048, True) if name.startswith("think") else ask(q)
            with lock:
                got[name] = a

    ts = [threading.Thread(target=worker) for _ in range(par)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return got


def main():
    out = sys.argv[2]
    ref = json.load(open(sys.argv[sys.argv.index("--ref") + 1])) if "--ref" in sys.argv else None
    t = time.time()
    arith, recall = arith_problems(), recall_docs()
    think = [("think" + n, q, a) for n, q, a in arith_problems(40, seed=77)]
    if "--think-only" in sys.argv:
        arith, recall = [], []
    got = run(arith + recall + think)
    score = {}
    import re
    def last_int(t):
        m = re.findall(r"-?\d+", t.replace(",", ""))
        return m[-1] if m else ""
    for label, items in (("arith", arith), ("recall", recall), ("think", think)):
        if not items:
            continue
        if label == "think":
            ok = sum(1 for name, _, ans in items if last_int(got[name]) == ans)
        else:
            ok = sum(1 for name, _, ans in items if ans in got[name].replace(",", ""))
        score[label] = (ok, len(items))
        print(f"{label}: {ok}/{len(items)} correct", flush=True)
    if ref is not None:
        keys = [k for k in got if k in ref["replies"]]
        same = sum(1 for k in keys if got[k] == ref["replies"][k])
        print(f"flips: {len(keys) - same} of {len(keys)} replies differ from the reference", flush=True)
    json.dump({"replies": got, "score": score, "seconds": time.time() - t}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
