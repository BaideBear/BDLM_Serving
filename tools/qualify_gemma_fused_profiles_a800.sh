#!/usr/bin/env bash
# Follow the one registered capacity worker; never preempt a job or hold a card.
set -euo pipefail
previous=${1:?registered capacity worker PID}
deadline=${2:?deadline epoch}
[[ $previous =~ ^[0-9]+$ && $deadline =~ ^[0-9]+$ ]]
if [[ -e /proc/$previous/stat ]]; then
    args=$(tr '\0' ' ' < "/proc/$previous/cmdline")
    [[ $args == *'cross_race --suite capacity --capacity-controls /gs/results/paid-close-1128 '* ]]
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
for profile in fixed90_fused fixed90_dense_fused; do
    [[ -z $(nvidia-smi -i 4 --query-compute-apps=pid --format=csv,noheader,nounits) ]]
    [[ $(date +%s) -lt $deadline ]]
    timeout -k 3 "$((deadline-$(date +%s)))" /root/miniconda3/envs/rl/bin/python -B -u \
        -m experiments.t2t.serving_slots.strict_handle_qualify --model gemma --profile "$profile" \
        --checkpoint /gs/models/diffusiongemma-cd40195 --samples /gs/code/samples.json \
        --receipts /gs/results --out "/gs/results/qualify-$profile-1128" --deadline "$deadline"
done
