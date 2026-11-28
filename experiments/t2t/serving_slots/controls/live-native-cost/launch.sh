#!/usr/bin/env bash
set -euo pipefail
umask 077
src=${1:?source}
[[ $src == /gs/work/live-native-cost-* && $(hostname -s) == n1 ]]
gpu=GPU-d3c3fc9c-b15d-9776-9094-7d3b85b705c1
root=/gs/runtime/verified-eos-1128/repo
ibs=/gs/work/gemma-ibs512-1127
idle() {
    [[ $(nvidia-smi -i 0 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits) == "$gpu, 0, 0" ]]
    [[ -z $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits) ]]
}
idle || exit 75
exec 9>>/dev/shm/gpu-a0.lock
flock -n 9 || exit 75
exec 8>>/dev/shm/formal-7a38ffc-A0.lock
flock -n 8 || exit 75
if [[ -d /dev/shm/unified-gemma-qual-1520 ]]; then
    exec 7>>/dev/shm/unified-gemma-qual-1520/A0.lock
    flock -n 7 || exit 75
fi
idle || exit 75
sleep 1
idle || exit 75
scratch=$(mktemp -d /dev/shm/live-native-cost.XXXXXXXX)
release() {
    local rc=$? mem apps
    trap - EXIT
    rmdir "$scratch"
    mem=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
    apps=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)
    exec 7>&- 8>&- 9>&-
    printf 'RELEASE rc=%s launcher=%s gpu=%s memory_mib=%s apps=%q locks=closed date=%s\n' \
        "$rc" "$$" "$gpu" "$mem" "$apps" "$(date -Is)"
    exit "$rc"
}
trap release EXIT
export CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$root:$ibs/search-r1-deps:/gs/env/deps:/gs/env/fixed-baseline-deps"
cd "$root"
sha256sum experiments/t2t/serving_slots/gemma_{verified_eos,verified_eos_eval,fixed_dual,wave_kv,strict_request,strict_rows,strict_sampler,unified_audit}.py \
    "$src/probe.py" "$src/launch.sh"
printf 'LOCKED launcher=%s gpu=%s date=%s\n' "$$" "$gpu" "$(date -Is)"
timeout -k 15s 10m unshare --mount --propagation private bash -c '
    set -euo pipefail
    mount -t tmpfs -o size=256m,nr_inodes=65536,mode=0700,nodev,nosuid tmpfs "$1"
    export TMPDIR="$1" HF_HOME="$1/hf" TRITON_CACHE_DIR="$1/triton"
    export XDG_CACHE_HOME="$1/xdg" TORCHINDUCTOR_CACHE_DIR="$1/inductor" CUDA_CACHE_PATH="$1/cuda"
    exec /root/miniconda3/envs/search-r1/bin/python -Bu "$2/probe.py"
' bash "$scratch" "$src" </dev/null
