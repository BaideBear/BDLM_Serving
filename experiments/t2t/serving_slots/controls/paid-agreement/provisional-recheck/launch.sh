#!/usr/bin/env bash
set -euo pipefail
umask 077
src=${1:?source overlay path}
[[ $src =~ ^/gs/work/gemma-provisional-recheck-a3-[a-zA-Z0-9-]+$ && $(hostname -s) == n1 ]]
root=/gs/work/gemma-equal-dose-a3-v1
gpu=GPU-94c61e79-d80b-cae2-d85f-6bd2a2c99a1c
idle() {
    [[ $(nvidia-smi -i 3 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits) == "$gpu, 0, 0" ]]
    [[ -z $(nvidia-smi -i 3 --query-compute-apps=pid --format=csv,noheader,nounits) ]]
}
idle || { echo 'BUSY physical A3'; exit 75; }
exec 9>>/dev/shm/gpu-a3.lock
flock -n 9 || { echo 'BUSY physical A3 lock'; exit 75; }
idle || exit 75
sleep 1
idle || exit 75
scratch=$(mktemp -d /dev/shm/gemma-provisional-recheck.XXXXXXXX)
release() {
    local rc=$? mem apps
    trap - EXIT
    rmdir "$scratch"
    mem=$(nvidia-smi -i 3 --query-gpu=memory.used --format=csv,noheader,nounits)
    apps=$(nvidia-smi -i 3 --query-compute-apps=pid --format=csv,noheader,nounits)
    exec 9>&-
    printf 'RELEASE rc=%s launcher=%s gpu=%s memory_mib=%s apps=%q locks=closed date=%s\n' \
        "$rc" "$$" "$gpu" "$mem" "$apps" "$(date -Is)"
    exit "$rc"
}
trap release EXIT
export CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$src:$root:/gs/work/gemma-strict-ibs-shadow-sf:/gs/env/deps:/gs/env/fixed-baseline-deps"
cd "$root"
sha256sum "$src"/experiments/t2t/serving_slots/*.py "$src/launch.sh" "$src/run.py"
printf 'LOCKED launcher=%s gpu=%s date=%s\n' "$$" "$gpu" "$(date -Is)"
timeout -k 15s 15m unshare --mount --propagation private bash -c '
    set -euo pipefail
    mount -t tmpfs -o size=128m,nr_inodes=65536,mode=0700,nodev,nosuid tmpfs "$1"
    export TMPDIR="$1" HF_HOME="$1/hf" TRITON_CACHE_DIR="$1/triton"
    export XDG_CACHE_HOME="$1/xdg" TORCHINDUCTOR_CACHE_DIR="$1/inductor" CUDA_CACHE_PATH="$1/cuda"
    export RECHECK_REFERENCE="$1/ref"
    mkdir "$RECHECK_REFERENCE"
    tar -xf - -C "$RECHECK_REFERENCE"
    exec /root/miniconda3/envs/rl/bin/python -Bu "$2/run.py"
' bash "$scratch" "$src"
