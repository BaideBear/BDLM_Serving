#!/usr/bin/env bash
set -euo pipefail
src=${1:?/gs/work/gemma-verified-eos-prefix-q64-a0-<id>}
deadline=${2:?unix deadline}
root=${3:?/dev/shm output directory}
refs=${4:?directory containing sealed C64-pair-*.json}
[[ $src =~ ^/gs/work/gemma-verified-eos-prefix-q64-a0-[0-9A-Za-z-]+$ && $(hostname -s) == n1 ]]
[[ $deadline =~ ^[0-9]+$ && $(date +%s) -lt $deadline && $root == /dev/shm/* && $refs == /dev/shm/* ]]
gpu=GPU-d3c3fc9c-b15d-9776-9094-7d3b85b705c1
idle() {
    [[ $(nvidia-smi -i 0 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits) == "$gpu, 0, 0" ]]
    [[ -z $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits) ]]
}
idle || { echo 'BUSY physical A0 (external PID or non-idle); not touching it'; exit 75; }
exec 9>>/dev/shm/gpu-a0.lock
flock -n 9 || { echo 'BUSY A0 lock'; exit 75; }
exec 8>>/dev/shm/formal-7a38ffc-A0.lock
flock -n 8 || { echo 'BUSY formal A0 lock'; exit 75; }
if [[ -d /dev/shm/unified-gemma-qual-1520 ]]; then
    exec 7>>/dev/shm/unified-gemma-qual-1520/A0.lock
    flock -n 7 || { echo 'BUSY qual A0 lock'; exit 75; }
fi
idle || exit 75
sleep 1
idle || exit 75
mkdir -p "$root" /dev/shm/triton-a0 /dev/shm/tmp-a0 /dev/shm/hf-a0
release() {
    local rc=$? mem apps
    trap - EXIT
    mem=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
    apps=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)
    exec 7>&- 8>&- 9>&- 2>/dev/null || exec 8>&- 9>&- 2>/dev/null || true
    printf 'RELEASE rc=%s host=%s launcher=%s gpu=%s memory_mib=%s apps=%q locks=closed date=%s\n' \
        "$rc" "$(hostname -s)" "$$" "$gpu" "$mem" "$apps" "$(date -Is)"
    exit "$rc"
}
trap release EXIT
export CUDA_VISIBLE_DEVICES="$gpu" TRITON_CACHE_DIR=/dev/shm/triton-a0 \
       TMPDIR=/dev/shm/tmp-a0 HF_HOME=/dev/shm/hf-a0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$src:/gs/env/deps:/gs/env/fixed-baseline-deps"
cd "$src"
sha256sum bdlm_serving/methods/t2t/prefix_coverage.py \
          bdlm_serving/methods/t2t/budget_family.py \
          experiments/t2t/serving_slots/gemma_fixed_dual.py \
          experiments/t2t/serving_slots/gemma_verified_eos.py \
          experiments/t2t/serving_slots/gemma_verified_eos_eval.py \
          experiments/t2t/serving_slots/gemma_verified_eos_prefix_identity_a0.py \
          tools/launch_gemma_verified_eos_prefix_identity_a0.sh
printf 'LOCKED host=%s launcher=%s gpu=%s cases=%s date=%s\n' \
    "$(hostname -s)" "$$" "$gpu" '7,13,25,33,55,71' "$(date -Is)"
timeout -k 15s "$((deadline - $(date +%s)))" /root/miniconda3/envs/rl/bin/python -u -B \
    -m experiments.t2t.serving_slots.gemma_verified_eos_prefix_identity_a0 \
    --root "$root" --deadline "$deadline" --references "$refs"
