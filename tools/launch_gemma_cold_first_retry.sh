#!/usr/bin/env bash
# One-shot Gemma cold-first-retry pilot/full90 on an otherwise idle A5/A6.
set -euo pipefail
gpu=${1:?physical GPU index}
deadline=${2:?deadline epoch}
out=${3:?new receipt directory}
q=${4:?fixed Q32 or Q64}
limit=${5:-3}
auto=${6:-}
case "$gpu:$q:$limit" in
  5:64:3|5:64:90|6:32:3|6:32:90) ;;
  *) exit 2 ;;
esac
[[ $deadline =~ ^[0-9]+$ && ! -e $out && $(date +%s) -lt $deadline ]]
[[ -z $(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits) ]]
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
case "$gpu:$uuid" in
  5:GPU-608770c1-ce99-323c-07a5-ab197153dd32) ;;
  6:GPU-8c80a093-8254-7eb4-db21-6a1bbe2d5d83) ;;
  *) exit 2 ;;
esac
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$uuid" OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:/gs/env/deps:/gs/env/fixed-baseline-deps"
export TRITON_CACHE_DIR="/gs/runtime/block-history-1127/triton-g${gpu}"
opts=(--out "$out" --verified /gs/results/verified-eos-1128
      --samples /gs/code/samples.json --checkpoint /gs/models/diffusiongemma-cd40195
      --deadline "$deadline" --q "$q" --limit "$limit")
[[ $auto == auto ]] && opts+=(--auto-full90)
exec timeout -k 5 "$((deadline - $(date +%s)))" /root/miniconda3/envs/rl/bin/python \
  -u -B -m experiments.t2t.serving_slots.gemma_cold_first_retry_eval "${opts[@]}"
