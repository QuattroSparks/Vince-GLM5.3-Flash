#!/usr/bin/env bash
# Vince GLM5.3 Flash: serve GLM-5.3-Flash on four DGX Sparks cabled as a ring (TP=4, Mia's TensorFold Python engine
# plus this recipe's image layer), through an OpenAI-compatible API on HOST:PORT (127.0.0.1:8000).
#
# Usage: ./start-vince.sh [restart] [extra tensorfold serve args]
#   ./start-vince.sh                              start (does nothing if it already runs)
#   ./start-vince.sh restart                      stop the running server and start again (~6 min)
#   VINCE_MODE=exact ./start-vince.sh restart     every greedy reply bit-identical to Mia's stock build
#   VINCE_MODE=stock ./start-vince.sh restart     Mia's image and settings, unchanged
#   DRY_RUN=1 ./start-vince.sh                    run the checks and print every rank's docker command; change nothing
#   ./start-vince.sh --help                       this text
#   ./stop-vince.sh                               stop all four ranks
#
# Modes (VINCE_MODE, vince/modes.sh; README.md "Modes"):
#   fast   (default) the fastest prefill; four prompt-precision switches make its replies NOT bit-identical to Mia's
#          stock build (quality-checked: README.md "Results")
#   exact  the RoCE ring exchange and the decode switches only: replies bit-identical to stock
#   stock  Mia's build
# A single switch can be overridden the same way: TF_GLM_PREFILL_ROWS=8192 ./start-vince.sh restart
#
# What it does: checks scripts/local.sh, the mode, ssh to the three workers and this recipe's image on all four Sparks
# (built from Mia's image plus vince/overlay where it is missing: vince/build-image.sh), then hands over to Mia's
# ./start-tp4-switchless.sh, which checks the weights on every Spark, the ring, the port and the memory, starts the
# four ranks and waits for the first reply.
#
# Exit codes: 0 serving (or DRY_RUN printed); 2 usage or settings; 3 a worker cannot be reached over ssh; 4 the image
# is missing and could not be built; anything else comes from Mia's start.sh (its message says why).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
case "${1:-}" in
  help|-h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
esac
say() { printf '[vince] %s\n' "$*"; }
fail() { local rc=$1; shift; printf '[vince] ERROR: %s\n' "$*" >&2; exit "$rc"; }

mode=${VINCE_MODE:-fast}
[[ $mode =~ ^(fast|exact|stock)$ ]] || fail 2 "VINCE_MODE is fast, exact or stock, not '$mode'"
[[ -f scripts/local.sh ]] ||
  fail 2 "scripts/local.sh is missing: cp vince/local.sh.example scripts/local.sh, then fill in your three workers (README.md \"Quick start\")"
grep -q 'vince/modes.sh' scripts/local.sh ||
  fail 2 "scripts/local.sh does not source vince/modes.sh (its last line in vince/local.sh.example): the modes would not apply"
# this setup's values as Mia's scripts will see them (the environment wins over scripts/local.sh)
read -r w1 w2 w3 image < <(bash -c 'e=${IMAGE-}; source scripts/local.sh >/dev/null 2>&1; echo "${WORKER:--} ${WORKER2:--} ${WORKER3:--} ${e:-${IMAGE:-tensorfold-glm53:v0.6.0}}"')
[[ $w1 != - && $w2 != - && $w3 != - ]] ||
  fail 2 "scripts/local.sh must set WORKER, WORKER2 and WORKER3 (the ring's order from this Spark; vince/local.sh.example)"
say "mode $mode: image $image$( [[ $mode == fast ]] && echo " (replies not bit-identical to Mia's stock build; VINCE_MODE=exact for that)")"
[[ ${DRY_RUN:-0} == 1 ]] && say "DRY_RUN=1: checks only; nothing is built, stopped or started"

i=0
for w in "$w1" "$w2" "$w3"; do
  i=$((i + 1))
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$w" true 2>/dev/null ||
    fail 3 "cannot reach rank $i ($w) over ssh without a password: check the address in scripts/local.sh and run ssh-copy-id $w"
done
say "ssh: ranks 1-3 answer ($w1, $w2, $w3)"

if [[ $mode != stock ]]; then
  if [[ ${DRY_RUN:-0} == 1 ]]; then
    ./vince/build-image.sh --check || say "DRY_RUN: a real start builds the image first (vince/build-image.sh)"
  else
    ./vince/build-image.sh ||
      fail 4 "this recipe's image is not on every Spark (see above). It is built from Mia's tensorfold-glm53:v0.6.0: if that is missing, VINCE_MODE=stock ./start-vince.sh prepares all four Sparks (and serves stock)"
  fi
fi
say "handing over to Mia's start-tp4-switchless.sh (weights on every Spark, the ring, port, memory; then the four ranks)"
export ENGINE=python
exec ./start-tp4-switchless.sh "$@"
