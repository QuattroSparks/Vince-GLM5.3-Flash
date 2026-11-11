#!/usr/bin/env bash
# Optional: NVIDIA Parakeet TDT 0.6B v3 speech-to-text beside the GLM server, for a voice chat.
# An OpenAI-compatible POST /v1/audio/transcriptions on one Spark, as a systemd user service. Not part of the recipe's
# start: nothing here runs unless you run it. It takes about 5 GiB of that Spark's memory (see README.md here).
#
#   ./install.sh                 install and start on the recipe's first worker (WORKER in scripts/local.sh)
#   ./install.sh status | stop | start | uninstall
#   PARAKEET_HOST=user@spark ./install.sh [...]     on the Spark you name, over ssh
#
# Never on the head by default: the head carries rank 0, the API and whatever else you run there, and it is the Spark
# that runs hottest. PARAKEET_HOST=local installs on this machine anyway, if that is what you want.
#
# Settings (environment):
#   PARAKEET_HOME      where it lives on that Spark                    (default ~/parakeet-stt)
#   PARAKEET_PORT      port                                            (default 8189)
#   PARAKEET_BIND      address it listens on                           (default 127.0.0.1: this machine only)
#   PARAKEET_API_KEY   if set, requests must send it as a Bearer key   (default: none)
#   PARAKEET_UNIT      systemd user unit name                          (default parakeet-stt)
#   PARAKEET_REVISION  the model's revision on Hugging Face            (default: the one this was run with)
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
CMD=${1:-install}
case "$CMD" in install|start|stop|status|uninstall) ;; -h|--help|help) sed -n '2,20s/^# \{0,1\}//p' "$0"; exit 0 ;;
  *) echo "usage: $0 [install|status|start|stop|uninstall]" >&2; exit 2 ;; esac
PHOME=${PARAKEET_HOME:-'~/parakeet-stt'}
PORT=${PARAKEET_PORT:-8189}
BIND=${PARAKEET_BIND:-127.0.0.1}
UNIT=${PARAKEET_UNIT:-parakeet-stt}
MODEL=nvidia/parakeet-tdt-0.6b-v3
REVISION=${PARAKEET_REVISION:-541d1f99c6b0c3cd0b11a95167540bb8edefd82b}
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1024 && PORT <= 65535 )) || { echo "PARAKEET_PORT is 1024 to 65535, not $PORT" >&2; exit 2; }
[[ "$UNIT" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "PARAKEET_UNIT: letters, digits, . _ - only" >&2; exit 2; }

if [[ -z "${PARAKEET_HOST:-}" && "${PARAKEET_REMOTE:-0}" != 1 ]]; then      # no target named: the first worker
  ROOT=$(cd "$HERE/../../.." && pwd)
  [[ -f "$ROOT/scripts/local.sh" ]] && PARAKEET_HOST=$(cd "$ROOT" && bash -c 'source scripts/local.sh >/dev/null 2>&1; echo "${WORKER:-}"')
  [[ -n "${PARAKEET_HOST:-}" ]] || { echo "no worker found in scripts/local.sh: name the Spark with PARAKEET_HOST=user@spark" \
    "(PARAKEET_HOST=local for this machine, which is normally the head)" >&2; exit 2; }
  echo "[parakeet] target: $PARAKEET_HOST (the recipe's first worker; PARAKEET_HOST=... names another)"
fi
if [[ "${PARAKEET_HOST:-}" == local ]]; then
  echo "[parakeet] installing on this machine as asked. If it is the head, its memory and heat are shared with rank 0."
  PARAKEET_HOST=
fi
if [[ -n "${PARAKEET_HOST:-}" ]]; then          # the same script on the other Spark, with the same settings
  dir=".cache/parakeet-stt-install"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$PARAKEET_HOST" "mkdir -p ~/$dir"
  scp -q -o BatchMode=yes "$HERE/install.sh" "$HERE/server.py" "$HERE/requirements.txt" "$PARAKEET_HOST:$dir/"
  exec ssh -o BatchMode=yes "$PARAKEET_HOST" "PARAKEET_HOME=$(printf %q "$PHOME") PARAKEET_PORT=$PORT PARAKEET_BIND=$(printf %q "$BIND") \
    PARAKEET_REMOTE=1 PARAKEET_UNIT=$UNIT PARAKEET_REVISION=$REVISION PARAKEET_FORCE=${PARAKEET_FORCE:-0} PARAKEET_API_KEY=$(printf %q "${PARAKEET_API_KEY:-}") bash ~/$dir/install.sh $CMD"
fi

PHOME=${PHOME/#\~/$HOME}
UNITFILE=$HOME/.config/systemd/user/$UNIT.service
ENVFILE=$HOME/.config/$UNIT.env
probe=$BIND; [[ $BIND == 0.0.0.0 ]] && probe=127.0.0.1
say() { echo "[parakeet] $*"; }

case "$CMD" in
  status) systemctl --user --no-pager status "$UNIT" 2>&1 | sed -n '1,6p' || true
          curl -s -m 5 "http://$probe:$PORT/health" || echo "no answer on $probe:$PORT"; echo; exit 0 ;;
  start|stop) systemctl --user "$CMD" "$UNIT"; say "$CMD: done"; exit 0 ;;
  uninstall)
    systemctl --user disable --now "$UNIT" 2>/dev/null || true
    rm -f "$UNITFILE" "$ENVFILE"; systemctl --user daemon-reload
    say "service removed. The files stay in $PHOME (venv and model, about 8 GB): delete that folder yourself to free them."
    exit 0 ;;
esac

# ---- install ----
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: this needs the Spark's GPU" >&2; exit 1; }
PY=$(command -v python3.12 || command -v python3) || { echo "python3 not found" >&2; exit 1; }
"$PY" -c 'import sys, venv; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null ||
  { echo "Python 3.10 or newer with the venv module is needed (Ubuntu: the python3-venv package)" >&2; exit 1; }
free_gib=$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)
(( free_gib >= 8 )) || { echo "only $free_gib GiB of memory is available here; Parakeet takes about 5 GiB and the GLM rank needs its headroom." \
  "Set PARAKEET_FORCE=1 to install anyway." >&2; [[ "${PARAKEET_FORCE:-0}" == 1 ]] || exit 1; }

mkdir -p "$PHOME/models" "$HOME/.config/systemd/user"
cp "$HERE/server.py" "$HERE/requirements.txt" "$PHOME/"
[[ -x "$PHOME/venv/bin/python" ]] || { say "creating $PHOME/venv"; "$PY" -m venv "$PHOME/venv"; }
if "$PHOME/venv/bin/python" - "$PHOME/requirements.txt" <<'PY' 2>/dev/null
import sys
from importlib.metadata import version
for line in open(sys.argv[1]):
    line = line.strip()
    if "==" in line and not line.startswith(("#", "-")):
        name, want = line.split("==")
        if version(name).split("+")[0] != want:           # torch reports 2.x.y+cu130
            sys.exit(1)
PY
then say "the Python packages are already installed at the pinned versions"
else
  say "installing the Python packages (the first time: about 6 GB, mostly PyTorch)"
  "$PHOME/venv/bin/python" -m pip install --quiet --disable-pip-version-check -r "$PHOME/requirements.txt"
fi

MDIR=$PHOME/models/parakeet-tdt-0.6b-v3
if [[ -f "$MDIR/model.safetensors" && "$(cat "$MDIR/REVISION" 2>/dev/null)" == "$REVISION" ]]; then
  say "model already here at revision ${REVISION:0:12}"
else
  say "downloading $MODEL @ ${REVISION:0:12} (2.4 GB; CC BY 4.0, by NVIDIA)"
  "$PHOME/venv/bin/python" - "$MODEL" "$REVISION" "$MDIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])
PY
  echo "$REVISION" > "$MDIR/REVISION"
fi

( umask 077; { echo "PARAKEET_DIR=$MDIR"; echo "HF_HUB_OFFLINE=1"; [[ -z "${PARAKEET_API_KEY:-}" ]] || echo "PARAKEET_API_KEY=$PARAKEET_API_KEY"; } > "$ENVFILE" )
cat > "$UNITFILE" <<UNITEOF
[Unit]
Description=Parakeet TDT 0.6B v3 speech-to-text, OpenAI /v1/audio/transcriptions on $BIND:$PORT
After=network-online.target

[Service]
WorkingDirectory=$PHOME
EnvironmentFile=$ENVFILE
ExecStart=$PHOME/venv/bin/uvicorn server:app --host $BIND --port $PORT --log-level info
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
UNITEOF
systemctl --user daemon-reload
systemctl --user enable "$UNIT" >/dev/null 2>&1
systemctl --user restart "$UNIT"
loginctl enable-linger "$USER" 2>/dev/null || say "could not enable linger: the service will start at your login, not at boot"

say "waiting for the model to load"
for _ in $(seq 1 150); do
  curl -s -m 3 "http://$probe:$PORT/health" 2>/dev/null | grep -q '"ok": *true' && { ok=1; break; }; sleep 2
done
[[ "${ok:-0}" == 1 ]] || { echo "no healthy answer on $probe:$PORT after 5 minutes: journalctl --user -u $UNIT" >&2; exit 1; }
say "serving on http://$BIND:$PORT  (model $MODEL)"
[[ $BIND == 127.0.0.1 ]] && say "it answers that Spark only. For the head, a phone or another machine set PARAKEET_BIND to the Spark's address on that network and re-run."
[[ $BIND != 127.0.0.1 && -z "${PARAKEET_API_KEY:-}" ]] && say "WARNING: it listens on $BIND with no key: anyone who reaches that address can send audio. Set PARAKEET_API_KEY or firewall the port."
say "try:  curl -s http://$probe:$PORT/v1/audio/transcriptions -F file=@some.wav"
