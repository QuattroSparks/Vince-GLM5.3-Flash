#!/usr/bin/env bash
# Vince GLM5.3 Flash: build this recipe's image tensorfold-glm53:vince-v2.5 on the four Sparks = Mia's
# tensorfold-glm53:v0.6.0 plus vince/overlay (the changed TensorFold files, as sources: no compiled .pyc files),
# labelled with the overlay's own hash, so its compiled kernels get their own folder under ~/.cache/tensorfold-glm53
# and never mix with another build's.
#   ./vince/build-image.sh           build where it is missing or different (a no-op when all four have it)
#   ./vince/build-image.sh --force   rebuild on all four
#   ./vince/build-image.sh --check   only report; exit 1 if a Spark lacks it
# TAG=... and ONLY="local user@host" override the tag and the Sparks.
# Needs Mia's image on each Spark (if missing: VINCE_MODE=stock ./start-vince.sh pulls or builds it on all four,
# through Mia's scripts/prepare.sh).
set -euo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
MODE=build; case "${1:-}" in --force) MODE=force ;; --check) MODE=check ;; "") ;;
  -h|--help|help) sed -n '2,11s/^# \{0,1\}//p' "$0"; exit 0 ;; *) echo "usage: $0 [--force|--check]" >&2; exit 2 ;; esac
[[ -f "$ROOT/scripts/local.sh" ]] || { echo "scripts/local.sh is missing: cp vince/local.sh.example scripts/local.sh and fill it in" >&2; exit 2; }
TAG=${TAG:-tensorfold-glm53:vince-v2.5}
BASE=${BASE:-tensorfold-glm53:v0.6.0}
read -r -a WORKERS <<<"$(cd "$ROOT" && bash -c 'source scripts/local.sh >/dev/null 2>&1; echo "$WORKER $WORKER2 $WORKER3"')"
NODES=(local "${WORKERS[@]}"); [[ -z "${ONLY:-}" ]] || read -r -a NODES <<<"$ONLY"   # ONLY="local" builds here only
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)
# the label: 12 hex of the overlay's files and paths (sorted as en_US.UTF-8, as when the live image was built)
LABEL=$(cd "$ROOT/vince/overlay" && find . -type f -print0 | LC_ALL=en_US.UTF-8 sort -z | xargs -0 sha256sum | sha256sum | cut -c1-12)

on() { if [[ $1 == local ]]; then bash -c "$2"; else "${SSH[@]}" "$1" "$2"; fi; }
label_of() { on "$1" "docker image inspect -f '{{index .Config.Labels \"tf.patches\"}}' $2 2>/dev/null" || true; }

CTX=$(mktemp -d); trap 'rm -rf "$CTX"' EXIT
cp -a "$ROOT/vince/overlay/." "$CTX/"
cat > "$CTX/Dockerfile" <<EOF
FROM $BASE
COPY tensorfold/ /usr/local/lib/python3.12/dist-packages/tensorfold/
COPY extra/bin/tensorfold /usr/local/bin/tensorfold
LABEL tf.patches=$LABEL tf.pf.base=$BASE
EOF

# kernel caches to start this build's from, the first that exists: those of the development builds this recipe was
# measured on (they exist only on the author's Sparks), else Mia's image's own
SEEDS=${SEEDS:-90e14f29ea76 10ad9c1018df f0e955050f72 9416aa451117 830adf78a9af fc21336e5bbd 10294884a520 f2dc974b7908}
# The build client sometimes never returns on a worker although the image is tagged: it is bounded (BUILD_TIMEOUT,
# 180 s) and the image's label decides whether the build worked.
build() {   # <node>: build there, then seed its kernel cache (unchanged kernels need no recompiling)
  local n=$1 have base_label
  have=$(label_of "$n" "$TAG")
  if [[ $MODE != force && $have == "$LABEL" ]]; then echo "$n: $TAG ok (label $LABEL)"; return 0; fi
  if [[ $MODE == check ]]; then echo "$n: $TAG ${have:+has label $have, not $LABEL}${have:-missing}"; return 1; fi
  base_label=$(label_of "$n" "$BASE")
  [[ -n $base_label ]] || { echo "$n: Mia's $BASE is missing (VINCE_MODE=stock ./start-vince.sh prepares all four Sparks)"; return 1; }
  local remote_ctx=.cache/vince-glm53-build
  if [[ $n != local ]]; then
    "${SSH[@]}" "$n" "rm -rf ~/$remote_ctx && mkdir -p ~/$remote_ctx"
    rsync -a "$CTX/" "$n:$remote_ctx/"
  fi
  local dir; [[ $n == local ]] && dir=$CTX || dir="\$HOME/$remote_ctx"
  on "$n" "cd $dir && { timeout -k 5 ${BUILD_TIMEOUT:-180} docker build -q -t $TAG . >/dev/null 2>&1; [ \"\$(docker image inspect -f '{{index .Config.Labels \"tf.patches\"}}' $TAG 2>/dev/null)\" = $LABEL ]; } &&
    c=\$HOME/.cache/tensorfold-glm53; if [ ! -d \$c/$LABEL ]; then for s in $SEEDS $base_label; do
      if [ -d \$c/\$s ]; then cp -a \$c/\$s \$c/$LABEL; rm -f \$c/$LABEL/pf-switches.json; break; fi; done; fi"
  [[ $n == local ]] || "${SSH[@]}" "$n" "rm -rf ~/$remote_ctx"
  echo "$n: $TAG built (label $LABEL)"
}

rc=0; pids=()
for n in "${NODES[@]}"; do build "$n" & pids+=($!); done
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit $rc
