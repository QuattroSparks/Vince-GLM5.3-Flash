#!/usr/bin/env python3
"""Same-bits check of a build against a reference: greedy (temperature 0) replies to fixed prompts that exercise
multi-chunk prefill (KDA, DSA dense and sparse rows, MoE, the row split, micro-batches), one request at a time.
  ref_test.py save ref.json      # on the reference build
  ref_test.py check ref.json     # on the candidate: every reply must be identical, token for token"""

import json
import random
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000"
MODEL = "GLM-5.3-Flash-EXL3"
WORDS = ("river garden medicine doctor patient kidney ultrasound valley mountain engine silver orange theory "
         "the of and to in is that for it as was with be by on not he this are or his from at which but have an "
         "they you were her she there been one all we their has would when if so no will more can who out about "
         "other into time some could them only new these then do two may first any like now my such its over than "
         "most also after where made many before must through back years much way well down should because each "
         "just those people how too little state good very make world still own see men work long here get").split()


def doc(n_words: int, seed: int) -> str:
    r = random.Random(seed)
    lines = []
    for i in range(0, n_words, 40):
        lines.append(f"[{i // 40}] " + " ".join(r.choice(WORDS) for _ in range(min(40, n_words - i))))
        if i % 2000 == 0:
            lines.append(f"Note {i}: the code word for section {i // 2000} is {r.choice(WORDS)}-{r.randint(100, 999)}.")
    return "\n".join(lines)


PROMPTS = [
    ("short", [{"role": "user", "content": "Explain in a few sentences why the sky is blue, then list three uses of "
                                            "ultrasound in medicine."}], 220, False),
    ("doc6k", [{"role": "user", "content": doc(5600, 11) + "\n\nList every code word mentioned in the notes above, "
                                                           "with its section number."}], 200, False),
    ("doc20k", [{"role": "user", "content": doc(19500, 22) + "\n\nWhat is the code word for section 3 and for "
                                                            "section 7? Then summarise the text in two lines."}],
     200, False),
    ("doc33k", [{"role": "user", "content": doc(32500, 33) + "\n\nWhich code words appear in the notes? Answer "
                                                            "briefly."}], 160, False),
    ("think", [{"role": "user", "content": "Write a Python function that returns the n-th prime number, with a short "
                                            "docstring. Think it through first."}], 320, True),
]


def ask(messages, max_tokens, thinking):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": thinking}}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        d = json.load(resp)
    dt = time.perf_counter() - t
    m = d["choices"][0]["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
            "usage": d.get("usage"), "seconds": dt}


def main() -> None:
    mode, path = sys.argv[1], sys.argv[2]
    got = {}
    for name, messages, mt, thinking in PROMPTS:
        r = ask(messages, mt, thinking)
        u = r["usage"] or {}
        print(f"{name}: {u.get('prompt_tokens')} prompt tokens ({(u.get('prompt_tokens_details') or {}).get('cached_tokens', 0)} "
              f"cached), {u.get('completion_tokens')} out, {r['seconds']:.1f} s", flush=True)
        got[name] = r
    if mode == "save":
        json.dump(got, open(path, "w"), indent=1)
        print("saved", path)
        return
    ref = json.load(open(path))
    bad = 0
    for name in ref:
        a, b = ref[name], got[name]
        same = a["content"] == b["content"] and a["reasoning"] == b["reasoning"]
        if not same:
            bad += 1
            ra, rb = a["reasoning"] + "\x00" + a["content"], b["reasoning"] + "\x00" + b["content"]
            i = next((i for i, (x, y) in enumerate(zip(ra, rb)) if x != y), min(len(ra), len(rb)))
            print(f"DIFFERENT {name}: first difference at char {i}: ref {ra[max(0, i - 40):i + 40]!r} / got "
                  f"{rb[max(0, i - 40):i + 40]!r}")
        else:
            print(f"same {name}")
    print("ALL IDENTICAL" if not bad else f"{bad} DIFFERENT")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
