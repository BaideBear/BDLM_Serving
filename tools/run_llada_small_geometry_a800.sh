#!/usr/bin/env bash
# Run actual work only; a busy device or existing run directory is not a retry.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:?study}; deadline=${2:?epoch}
q=${3:-8}
case "$mode/$q" in
  extend/8)
    gpu=0
    module=llada_frontier_seat
    out=/gs/results/llada-frontier-q8-extend-1128
    extra=(--limit 90 --extend-from /gs/results/llada-frontier-seat-q8-1128)
    ;;
  window/8|window/16)
    if [[ $q == 16 ]]; then gpu=5; else gpu=6; fi
    module=llada_frontier_seat
    out="/gs/results/llada-window-q$q-1128"
    extra=(--factor window --limit 30)
    ;;
  *) exit 2 ;;
esac
test ! -e "$out"
remaining=$((deadline-$(date +%s)))
(( remaining>120 ))
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
[[ -z $(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader,nounits) ]]
export CUDA_VISIBLE_DEVICES="$uuid" OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:/gs/env/fixed-baseline-deps" TRITON_CACHE_DIR="/gs/runtime/llada-admission-1128/triton-$gpu"
exec timeout -k 3 "$remaining" /gs/env/venv-llada/bin/python -Bu -m "experiments.t2t.serving_slots.$module" \
    --out "$out" --offer "$q" "${extra[@]}" --deadline "$deadline" \
    --reference /gs/work/llada-refs/llada21 --checkpoint /gs/models/LLaDA2.1-mini --samples /gs/code/samples.json
