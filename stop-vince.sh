#!/usr/bin/env bash
# Vince GLM5.3 Flash: stop all four ranks (Mia's ./stop.sh; each rank's log is kept under ~/.cache/tensorfold-glm53/logs).
#   DRY_RUN=1 ./stop-vince.sh   says what it would stop, and stops nothing
set -euo pipefail
case "${1:-}" in help|-h|--help) sed -n '2,3s/^# \{0,1\}//p' "$0"; exit 0 ;; esac
export ENGINE=python
exec "$(dirname "$(readlink -f "$0")")/stop.sh" "$@"
