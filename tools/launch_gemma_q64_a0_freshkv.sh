#!/usr/bin/env bash
# A0-only Gemma Q64 paid-IBS same-state fresh-KV diagnostic; no holder, no disk logs.
set -euo pipefail
umask 077
src=${1:?source overlay}
deadline=${2:?deadline epoch}
root=${3:?/dev/shm output directory}
cases=${4:-7,13,25,33,55,71}
probe=${5:-full}
[[ $probe == full || $probe == paid-prefix || $probe == native-prefix ]]
[[ $src =~ ^/gs/work/gemma-q64-a0-freshkv-[0-9A-Za-z-]+$ && $(hostname -s) == n1 ]]
[[ $deadline =~ ^[0-9]+$ && $(date +%s) -lt $deadline && $root == /dev/shm/* ]]
gpu=GPU-d3c3fc9c-b15d-9776-9094-7d3b85b705c1
idle() {
    [[ $(nvidia-smi -i 0 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits) == "$gpu, 0, 0" ]]
    [[ -z $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits) ]]
}
# External PID guard: never kill or evict; refuse to start if A0 is busy.
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
base=/gs/work/gemma-paid-agreement-q128
export PYTHONPATH="$src:$base:/gs/work/gemma-strict-ibs-shadow-sf:/gs/env/deps:/gs/env/fixed-baseline-deps"
cd "$base"
sha256sum "$src"/experiments/t2t/serving_slots/gemma_freshkv.py \
          "$src"/experiments/t2t/serving_slots/gemma_paid_agreement_q64_a0_freshkv.py \
          "$src"/tools/launch_gemma_q64_a0_freshkv.sh \
          "$base"/experiments/t2t/serving_slots/gemma_ibs_certified_eos.py \
          "$base"/experiments/t2t/serving_slots/gemma_paid_agreement.py \
          "$base"/experiments/t2t/serving_slots/gemma_paid_agreement_eval.py \
          "$base"/bdlm_serving/runtime/diffusiongemma_optimus.py
printf 'LOCKED host=%s launcher=%s gpu=%s cases=%s date=%s\n' \
    "$(hostname -s)" "$$" "$gpu" "$cases" "$(date -Is)"
timeout -k 15s "$((deadline - $(date +%s)))" /root/miniconda3/envs/rl/bin/python -u -B \
    -m experiments.t2t.serving_slots.gemma_paid_agreement_q64_a0_freshkv \
    --root "$root" --deadline "$deadline" --gpu-uuid "$gpu" --cases "$cases" --probe "$probe"
