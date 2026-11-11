#!/usr/bin/env bash
# The prefill experiments' runtime switch file (pf_switches.py) on all four Sparks, for this recipe's image. Writes
# every key (a key left out keeps its old value): ./vince/tools/switches.sh no_reuse=1   (back: no arguments = all off)
# no_reuse=1 turns prompt-cache reuse off, for the same-bits checks; the others are experiments that measured slower.
# v2's five switches (tap_tail, lean, part16, own_router, zc) are set by vince/modes.sh's environment at start and
# are deliberately not written here: a key in the file would override them. The same goes for v2.2's fill_whole, kq and
# bcast (fill_whole was one of this file's keys until v2.1).
set -euo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
declare -A S=([no_reuse]=0 [shared_whole]=0 [gemm]=0 [gemm_min_rows]=256 [kda_overlap]=0)
for kv in "$@"; do k=${kv%%=*}; [[ -n ${S[$k]+x} && $kv == *=* ]] || { echo "unknown: $kv (keys: ${!S[*]})" >&2; exit 2; }; S[$k]=${kv#*=}; done
J=$(for k in no_reuse shared_whole gemm gemm_min_rows kda_overlap; do printf '"%s":%s,' "$k" "${S[$k]}"; done)
J="{${J%,}}"; python3 -c "import json,sys; json.loads(sys.argv[1])" "$J"
LABEL=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "${TAG:-tensorfold-glm53:vince-v2.4}")
read -r -a WORKERS <<<"$(cd "$ROOT" && bash -c 'source scripts/local.sh >/dev/null 2>&1; echo "$WORKER $WORKER2 $WORKER3"')"
echo "$J" > "$HOME/.cache/tensorfold-glm53/$LABEL/pf-switches.json"
for n in "${WORKERS[@]}"; do ssh -o BatchMode=yes "$n" "cat > ~/.cache/tensorfold-glm53/$LABEL/pf-switches.json" <<<"$J" & done
wait
echo "switches $J -> label $LABEL on 4 Sparks"
