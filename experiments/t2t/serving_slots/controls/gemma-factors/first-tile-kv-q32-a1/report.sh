#!/usr/bin/env bash
set -euo pipefail
rt=/gs/runtime/gemma-first-tile-kv-q32-a1-1128
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="$rt:/gs/runtime/cold-frontier-last-a1-1128:/gs/runtime/verified-eos-1128/repo"
n=${1:-90}
exec /usr/bin/python3 -B -m experiments.t2t.serving_slots.gemma_first_tile_kv_report \
    --checkpoint "$n" --out "$rt/compact-$n"
