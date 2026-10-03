#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
VIOEXPLAIN_PYTHON="${VIOEXPLAIN_PYTHON:-python3}"
command="${1:-help}"
if [ "$#" -gt 0 ]; then shift; fi
case "$command" in
  smoke|prototype-smoke) "$VIOEXPLAIN_PYTHON" -m vioexplain.cli "$command" "$@" ;;
  test) "$VIOEXPLAIN_PYTHON" -m pytest -q "$@" ;;
  formal) "$VIOEXPLAIN_PYTHON" -m run_vioexplain.formal_runner "$@" ;;
  *) printf '%s\n' 'Usage: bash run.sh smoke [--out FILE] | prototype-smoke | test | formal --help' ;;
esac
