#!/usr/bin/env bash
set -euo pipefail
umask 077
rt=/gs/runtime/waiting-seat-q64-1128
cd "$rt/repo"
uuid=GPU-f7fcacba-741b-a1ea-b370-4780cd948f6d
test -s /gs/results/eos-handle-1128-v2/complete.json
test ! -d /proc/122082
test ! -e /gs/results/waiting-seat-q64-1128
test "$(nvidia-smi -i 7 --query-gpu=uuid,memory.used --format=csv,noheader,nounits)" = "$uuid, 0"
pids=$(nvidia-smi -i 7 --query-compute-apps=pid --format=csv,noheader,nounits)
test -z "$pids"
export CUDA_VISIBLE_DEVICES="$uuid" PYTHONPATH="$PWD:/gs/env/deps:/gs/env/fixed-baseline-deps"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export TRITON_CACHE_DIR="$rt/triton"
exec flock -n "$rt/launch.lock" timeout 3600 /root/miniconda3/envs/rl/bin/python -uB \
    -m experiments.t2t.serving_slots.gemma_waiting_seat_eval
