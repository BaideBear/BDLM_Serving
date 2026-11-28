#!/usr/bin/env bash
# One-shot Gemma warm resolved-debt resample on idle A3.
set -euo pipefail
gpu=${1:?physical GPU index}
deadline=${2:?deadline epoch}
out=${3:?new receipt directory}
q=${4:?fixed Q32 or Q64}
limit=${5:-3}
auto=${6:-}
case "$gpu:$q:$limit" in
  3:32:3|3:32:90|3:64:3|3:64:90) ;;
  *) exit 2 ;;
esac
[[ $deadline =~ ^[0-9]+$ && ! -e $out && $(date +%s) -lt $deadline ]]
[[ -z $(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits) ]]
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
[[ $uuid == GPU-94c61e79-d80b-cae2-d85f-6bd2a2c99a1c ]] || exit 2
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$uuid" OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:/gs/env/deps:/gs/env/fixed-baseline-deps"
export TRITON_CACHE_DIR="/gs/runtime/block-history-1127/triton-g${gpu}"
opts=(--out "$out" --verified /gs/results/verified-eos-1128
      --samples /gs/code/samples.json --checkpoint /gs/models/diffusiongemma-cd40195
      --deadline "$deadline" --q "$q" --limit "$limit")
[[ $auto == auto ]] && opts+=(--auto-full90)
exec timeout -k 5 "$((deadline - $(date +%s)))" /root/miniconda3/envs/rl/bin/python \
  -u -B -m experiments.t2t.serving_slots.gemma_warm_debt_resample_eval "${opts[@]}"
