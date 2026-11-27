#!/usr/bin/env bash
# Stream only aggregate JSON from A800; read receipts on A800, write products here.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
script=experiments/t2t/serving_slots/cross_race_figure.py
host=${A800_HOST:-A800-80g-2}
opts=(-o BatchMode=yes -o ConnectTimeout=8)
if [[ -n ${A800_MUX:-} ]]; then opts+=(-S "$A800_MUX"); fi
if [[ -n ${A800_PORT:-} ]]; then opts+=(-p "$A800_PORT"); fi
python=${PYTHON:-python3}
"$python" -B -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Set PYTHON to a Python 3.10+ interpreter.")'
ssh "${opts[@]}" "$host" \
  'cd /gs/runtime/cross-race-1128/repo &&
   /root/miniconda3/envs/rl/bin/python -B - gather /gs/results/cross-race-1128 \
     --logs /gs/runtime/cross-race-1128 --samples /gs/code/samples.json' < "$script" |
  "$python" -B -m experiments.t2t.serving_slots.cross_race_figure render "$@"
