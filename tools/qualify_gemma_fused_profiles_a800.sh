#!/usr/bin/env bash
# Follow the one registered capacity worker; never preempt a job or hold a card.
set -euo pipefail
previous=${1:?registered worker PID}
deadline=${2:?deadline epoch}
study=${3:-capacity}
case $study in
    capacity) profiles=(fixed90_fused fixed90_dense_fused) ;;
    density) profiles=(phase_high_fused) ;;
    dense-stop|draft-shadow) profiles=(dense_b_stop_fused) ;;
    *) exit 2 ;;
esac
[[ $previous =~ ^[0-9]+$ && $deadline =~ ^[0-9]+$ ]]
if [[ -e /proc/$previous/stat ]]; then
    args=$(tr '\0' ' ' < "/proc/$previous/cmdline")
    controls=/gs/results/paid-close-1128
    if [[ $study == dense-stop ]]; then controls=/gs/results/capacity-head-1128; fi
    if [[ $study == draft-shadow ]]; then
        [[ $args == *"qualify_gemma_fused_profiles_a800.sh "*" dense-stop" ]]
    else
        [[ $args == *"cross_race --suite $study --capacity-controls $controls "* ]]
    fi
    gen=$(awk '{print $22}' /proc/"$previous"/stat)
    while [[ -e /proc/$previous/stat && $(awk '{print $22}' "/proc/$previous/stat" 2>/dev/null) == "$gen" ]]; do
        [[ $(date +%s) -lt $deadline ]] || exit 1
        sleep 2
    done
fi
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 4 --query-gpu=uuid --format=csv,noheader)"
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:/gs/env/deps:/gs/env/fixed-baseline-deps"
export TRITON_CACHE_DIR=/gs/runtime/block-history-1127/triton-g4
for profile in "${profiles[@]}"; do
    [[ -z $(nvidia-smi -i 4 --query-compute-apps=pid --format=csv,noheader,nounits) ]]
    [[ $(date +%s) -lt $deadline ]]
    out="/gs/results/qualify-$profile-1128"
    opts=()
    if [[ $study == draft-shadow ]]; then opts=(--dense-shadow); out=/gs/results/draft-shadow-1128; fi
    timeout -k 3 "$((deadline-$(date +%s)))" /root/miniconda3/envs/rl/bin/python -B -u \
        -m experiments.t2t.serving_slots.strict_handle_qualify --model gemma --profile "$profile" "${opts[@]}" \
        --checkpoint /gs/models/diffusiongemma-cd40195 --samples /gs/code/samples.json \
        --receipts /gs/results --out "$out" --deadline "$deadline"
done
