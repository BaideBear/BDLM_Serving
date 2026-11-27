#!/usr/bin/env bash
# Run both fusion arms for one fixed offer on one otherwise idle device. No holder.
set -euo pipefail
case "${1:-}:${2:-}" in 2:128|4:512) ;; *) exit 2 ;; esac
i=$1
q=$2
deadline=${3:?provide the study deadline in Unix seconds}
[[ $deadline =~ ^[0-9]+$ && $(date +%s) -lt $deadline ]]
uuid=$(nvidia-smi -i "$i" --query-gpu=uuid --format=csv,noheader)
[[ -z $(nvidia-smi -i "$i" --query-compute-apps=pid --format=csv,noheader,nounits) ]]
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$uuid" OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:/gs/env/deps:/gs/env/fixed-baseline-deps"
export TRITON_CACHE_DIR="/gs/runtime/block-history-1127/triton-g$i"
exec timeout -k 3 "$((deadline - $(date +%s)))" /root/miniconda3/envs/rl/bin/python \
    -u -B -m experiments.t2t.serving_slots.cross_race --suite profile-close \
    --profile-close-budget "$q" --samples /gs/code/samples.json \
    --checkpoint /gs/models/diffusiongemma-cd40195 --output /gs/results/profile-close-1128 \
    --deadline "$deadline"
