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
  window90/8|window90/16)
    if [[ $q == 16 ]]; then gpu=5; else gpu=6; fi
    module=llada_frontier_seat
    parent="/gs/results/llada-window-q$q-1128"
    /usr/bin/python3 -B -m experiments.t2t.serving_slots.llada_seat_report \
      "$parent" benchmarks/frozen90/manifest.json >/dev/null
    out="/gs/results/llada-window-q$q-extend-1128"
    extra=(--factor window --limit 90 --extend-from "$parent")
    ;;
  window90/4)
    gpu=0
    module=llada_frontier_seat
    # A0 is assigned to the 90-pair Q8 frontier extension first.
    /usr/bin/python3 -B -m experiments.t2t.serving_slots.llada_seat_report \
      /gs/results/llada-frontier-q8-extend-1128 benchmarks/frozen90/manifest.json >/dev/null
    out=/gs/results/llada-window-q4-1128
    extra=(--factor window --limit 90)
    ;;
  paid90/4)
    gpu=0
    module=llada_frontier_seat
    parent=/gs/results/llada-window-q4-1128
    out=/gs/results/llada-window-paid-q4-1128
    extra=(--factor paid --limit 90 --paired-window "$parent")
    ;;
  successor/64)
    gpu=1
    module=llada_frontier_seat
    out=/gs/results/llada-quarter-successor-q64-1128
    extra=(--factor successor --limit 30)
    # The three-arm interior study owns A1 until it exits with a full cohort.
    /usr/bin/python3 -B -m experiments.t2t.serving_slots.llada_seat_report \
      /gs/results/llada-interior-q8-1128 benchmarks/frozen90/manifest.json >/dev/null
    ;;
  *) exit 2 ;;
esac
test ! -e "$out"
remaining=$((deadline-$(date +%s)))
(( remaining>120 ))
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
[[ -z $(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader,nounits) ]]
[[ $(nvidia-smi -i "$uuid" --query-gpu=memory.used --format=csv,noheader,nounits) -lt 100 ]]
export CUDA_VISIBLE_DEVICES="$uuid" OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:/gs/env/fixed-baseline-deps" TRITON_CACHE_DIR="/gs/runtime/llada-admission-1128/triton-$gpu"
exec timeout -k 3 "$remaining" /gs/env/venv-llada/bin/python -Bu -m "experiments.t2t.serving_slots.$module" \
    --out "$out" --offer "$q" "${extra[@]}" --deadline "$deadline" \
    --reference /gs/work/llada-refs/llada21 --checkpoint /gs/models/LLaDA2.1-mini --samples /gs/code/samples.json
