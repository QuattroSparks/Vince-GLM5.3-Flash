#!/usr/bin/env bash
# Vince GLM5.3 Flash: check the running server against this recipe's saved results (restarts nothing).
#   ./vince/check.sh          same bits: greedy replies to 5 fixed prompts (short, 6k, 20k, 33k, thinking) must equal
#                             the saved ones of this mode (results/replies-default.json, or replies-stock.json for
#                             VINCE_MODE=exact/stock), solo and all five at once; then prefill 8k/16k/32k (fixed seed)
#   ./vince/check.sh full     + the quality suite (vs the exact build's replies), concurrency c1-c8, the 195k needle
# Prompt-cache reuse is off during the checks and back on after. Run it while nobody else uses the server.
set -uo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd); T=$ROOT/vince/tools; R=$ROOT/vince/results
OUT=${OUT:-$HOME/.cache/vince-glm53-check/$(date +%Y%m%d-%H%M%S)}; mkdir -p "$OUT"
ref=$R/replies-default.json; [[ "${VINCE_MODE:-fast}" == fast ]] || ref=$R/replies-stock.json
"$T/switches.sh" no_reuse=1 >/dev/null; trap '"$T/switches.sh" >/dev/null' EXIT
cd "$T"
echo "== same bits vs $(basename "$ref")";      python3 ref_test.py check "$ref" | tail -1
echo "== concurrent == solo";                   python3 conc_test.py "$ref" | tail -1
echo "== decode, one request (saved v2.1: prose 72.9 / code 97.4 tok/s whole request; v2 70.3 / 94.2)"
python3 decode_check.py | tail -2
echo "== prefill (saved at the 2,200 MHz cap: ~3,880 / 3,980 / 4,088 tok/s fast (v2.3; v2.4 4,114 at 32k; v2.2 3,762 / 3,832 / 3,999, with the head at 2,288 MHz 3,813 / 3,899 / 4,052; v2 3,390 / 3,703 / 3,766; v1 3,112 / 3,384 / 3,437), 2,627 / 2,818 / 2,842 exact)"
python3 bench_prefill.py --seed 777 --sizes 8192,16384,32768 --reps 3 --json "$OUT/prefill.json" | tail -4
if [[ "${1:-}" == full ]]; then
  echo "== quality (saved: think 40/40, recall 19/19, no-think arith 15/60)"
  python3 quality.py run "$OUT/quality.json" --ref "$R/quality-exact.json" | tail -3
  echo "== concurrency";                         python3 conc_bench.py --json "$OUT/concurrency.json"
  echo "== 195k needle";                         (cd "$ROOT" && PORT=8000 python3 tools/needle.py vince-check | tail -1)
fi
echo "results in $OUT"
