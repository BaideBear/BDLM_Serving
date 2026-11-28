#!/usr/bin/env bash
set -euo pipefail
umask 077
rt=/gs/runtime/gemma-first-tile-kv-q32-a1-1128
cd "$rt"
test ! -e /proc/272603
test ! -e /gs/results/gemma-first-tile-kv-q32-a1-1128
export CUDA_VISIBLE_DEVICES=GPU-861c9b5d-1e73-2550-f832-b73988bc64ff
test "$(nvidia-smi -i 1 --query-gpu=uuid,memory.used --format=csv,noheader,nounits)" = "$CUDA_VISIBLE_DEVICES, 0"
pids=$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits)
test -z "$pids"
export PYTHONPATH="$rt:/gs/runtime/cold-frontier-last-a1-1128:/gs/runtime/verified-eos-1128/repo:/gs/env/deps:/gs/env/fixed-baseline-deps"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="$rt/triton"
left=$((1795843443-$(date +%s)))
test "$left" -gt 0
exec flock -n "$rt/run.lock" timeout -s KILL "$left" /root/miniconda3/envs/rl/bin/python -Bu -m experiments.t2t.serving_slots.gemma_first_tile_kv_eval
