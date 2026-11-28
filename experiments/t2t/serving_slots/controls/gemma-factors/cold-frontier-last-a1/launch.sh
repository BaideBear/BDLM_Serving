#!/usr/bin/env bash
set -euo pipefail
umask 077
rt=/gs/runtime/cold-frontier-last-a1-1128
cd "$rt"
export CUDA_VISIBLE_DEVICES=GPU-861c9b5d-1e73-2550-f832-b73988bc64ff
test ! -e /gs/results/cold-frontier-last-a1-1128
test "$(nvidia-smi -i 1 --query-gpu=uuid,memory.used --format=csv,noheader,nounits)" = "$CUDA_VISIBLE_DEVICES, 0"
pids=$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits)
test -z "$pids"
export PYTHONPATH="$rt:/gs/runtime/verified-eos-1128/repo:/gs/env/deps:/gs/env/fixed-baseline-deps"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export TRITON_CACHE_DIR="$rt/triton"
remain=$((1795843443-$(date +%s)))
test "$remain" -gt 0
exec flock -n "$rt/run.lock" timeout -s KILL "$remain" /root/miniconda3/envs/rl/bin/python -uB \
  -m experiments.t2t.serving_slots.gemma_cold_frontier_last_eval
