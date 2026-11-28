#!/usr/bin/env bash
set -euo pipefail
umask 077
rt=/gs/runtime/gemma-ibs-prefix-bridge-a1-1128
src=/gs/work/gemma-ibs512-1127
cd "$rt"
test ! -e /gs/results/gemma-ibs-prefix-bridge-a1-1128
test -s checkpoint.json
export CUDA_VISIBLE_DEVICES=GPU-861c9b5d-1e73-2550-f832-b73988bc64ff
test "$(nvidia-smi -i 1 --query-gpu=uuid,memory.used --format=csv,noheader,nounits)" = "$CUDA_VISIBLE_DEVICES, 0"
apps=$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits)
test -z "$apps"
export PYTHONPATH="$rt:$src:$src/search-r1-deps:/gs/env/deps:/gs/env/fixed-baseline-deps:/gs/runtime/verified-eos-1128/repo"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 TRITON_CACHE_DIR="$rt/triton"
left=$((1795849200-$(date +%s)-2))
test "$left" -gt 60
exec flock -n "$rt/run.lock" timeout -s KILL "$left" /root/miniconda3/envs/search-r1/bin/python -Bu -m experiments.t2t.serving_slots.gemma_ibs_prefix_bridge_eval
