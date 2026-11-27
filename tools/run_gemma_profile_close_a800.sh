#!/usr/bin/env bash
# Run both fusion arms for one fixed offer on one otherwise idle device. No holder.
set -euo pipefail
case "${1:-}:${2:-}" in 2:128|4:512|4:capacity|4:density) ;; *) exit 2 ;; esac
i=$1
q=$2
deadline=${3:?provide the study deadline in Unix seconds}
[[ $deadline =~ ^[0-9]+$ && $(date +%s) -lt $deadline ]]
if [[ $# -eq 4 ]]; then
    prior=$4
    [[ $2 == capacity && $prior =~ ^[0-9]+$ ]]
    if [[ -e /proc/$prior/cmdline ]]; then
        previous=$(tr '\0' ' ' < "/proc/$prior/cmdline")
        [[ $previous == *'cross_race --suite profile-close --profile-close-budget 512 '* ]]
        generation=$(awk '{print $22}' "/proc/$prior/stat")
        while [[ -e /proc/$prior/stat && $(awk '{print $22}' "/proc/$prior/stat" 2>/dev/null) == "$generation" ]]; do
            [[ $(date +%s) -lt $deadline ]] || exit 1
            sleep 2
        done
    fi
fi
uuid=$(nvidia-smi -i "$i" --query-gpu=uuid --format=csv,noheader)
[[ -z $(nvidia-smi -i "$i" --query-compute-apps=pid --format=csv,noheader,nounits) ]]
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$uuid" OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:/gs/env/deps:/gs/env/fixed-baseline-deps"
export TRITON_CACHE_DIR="/gs/runtime/block-history-1127/triton-g$i"
if [[ $q == capacity ]]; then
    args=(--suite capacity --capacity-controls /gs/results/paid-close-1128 --output /gs/results/capacity-head-1128)
elif [[ $q == density ]]; then
    args=(--suite density --capacity-controls /gs/results/paid-close-1128 --output /gs/results/phase-density-1128)
else
    args=(--suite profile-close --profile-close-budget "$q" --output /gs/results/profile-close-1128)
fi
exec timeout -k 3 "$((deadline - $(date +%s)))" /root/miniconda3/envs/rl/bin/python \
    -u -B -m experiments.t2t.serving_slots.cross_race "${args[@]}" --samples /gs/code/samples.json \
    --checkpoint /gs/models/diffusiongemma-cd40195 \
    --deadline "$deadline"
