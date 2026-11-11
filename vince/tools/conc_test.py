#!/usr/bin/env python3
"""The ref_test prompts all at once (concurrent): every greedy reply must equal the solo reference's."""
import json, sys, threading
import ref_test as R
ref = json.load(open(sys.argv[1]))
got = {}
def one(name, messages, mt, thinking):
    got[name] = R.ask(messages, mt, thinking)
ts = [threading.Thread(target=one, args=p) for p in R.PROMPTS]
[t.start() for t in ts]; [t.join() for t in ts]
bad = [n for n in ref if (ref[n]["content"], ref[n]["reasoning"]) != (got[n]["content"], got[n]["reasoning"])]
for n in ref: print(n, "same" if n not in bad else "DIFFERENT", f"{got[n]['seconds']:.1f} s", (got[n]['usage'] or {}).get('prompt_tokens_details'))
print("CONCURRENT == SOLO" if not bad else f"{len(bad)} DIFFERENT")
