#!/usr/bin/env bash
# A4/A7 are also the actual GPUs used by the respective fixed controls.
set -euo pipefail
cd "$(dirname "$0")/.."
q=${1:?fixed offer}; deadline=${2:?deadline epoch}
scope=${3:-all}
case "$scope" in all) suffix= ;; solo) [[ $q == 64 ]]; suffix=-solo ;; *) exit 2 ;; esac
case "$q" in 32) gpu=4 ;; 64) gpu=7 ;; *) exit 2 ;; esac
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
[[ -z $(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader,nounits) ]]
[[ $(nvidia-smi -i "$uuid" --query-gpu=memory.used --format=csv,noheader,nounits) -lt 100 ]]
remaining=$((deadline-$(date +%s))); ((remaining>180))
out="/gs/results/llada-frontier-accept-q$q$suffix-1128"
test ! -e "$out"
export CUDA_VISIBLE_DEVICES="$uuid" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
export PYTHONPATH="$PWD:/gs/env/fixed-baseline-deps"
exec timeout -k 5 "$remaining" /gs/env/venv-llada/bin/python -Bu \
    -m experiments.t2t.serving_slots.llada_frontier_accept_eval run --q "$q" --scope "$scope" \
    --checkpoint /gs/models/LLaDA2.1-mini --reference /gs/work/llada-refs/llada21 \
    --samples /gs/code/samples.json --out "$out" --deadline "$deadline"
