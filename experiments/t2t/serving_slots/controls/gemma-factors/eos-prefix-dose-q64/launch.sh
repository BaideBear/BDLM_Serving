#!/usr/bin/env bash
set -euo pipefail
umask 077
rt=/gs/runtime/eos-prefix-dose-q64-1128
base=/gs/runtime/verified-eos-1128/repo
cd "$rt"
export CUDA_VISIBLE_DEVICES=GPU-f7fcacba-741b-a1ea-b370-4780cd948f6d
export DOSE_END_EPOCH=$(date -d '2026-11-28 13:24:00 +0800' +%s)
remaining=$((DOSE_END_EPOCH-$(date +%s)))
test "$remaining" -gt 0
test ! -e /gs/results/eos-prefix-dose-q64-1128
test "$(nvidia-smi -i 7 --query-gpu=uuid,memory.used --format=csv,noheader,nounits)" = "$CUDA_VISIBLE_DEVICES, 0"
pids=$(nvidia-smi -i 7 --query-compute-apps=pid --format=csv,noheader,nounits)
test -z "$pids"
export PYTHONPATH="$rt:$base:/gs/env/deps:/gs/env/fixed-baseline-deps"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export TRITON_CACHE_DIR="$rt/triton"
exec flock -n "$rt/run.lock" timeout -s KILL "$remaining" /root/miniconda3/envs/rl/bin/python -uB \
    -m experiments.t2t.serving_slots.gemma_eos_prefix_dose_eval
