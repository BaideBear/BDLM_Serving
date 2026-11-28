#!/usr/bin/env bash
set -euo pipefail
umask 077
src=${1:?source directory under /gs/work}
head=${2:?source base SHA}
q=${3:?fixed physical decoder offer}
[[ $q == 32 || $q == 64 || $q == 128 ]]
gpu=GPU-d3c3fc9c-b15d-9776-9094-7d3b85b705c1
[[ $src == /gs/work/gemma-ibs-certified-eos-* && $head =~ ^[0-9a-f]{40}$ ]]
[[ $(hostname -s) == n1 ]]
root=/gs/work/gemma-ibs512-1127
python=/root/miniconda3/envs/search-r1/bin/python
idle() {
    [[ $(nvidia-smi -i 0 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits) == "$gpu, 0, 0" ]]
    [[ -z $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits) ]]
}
idle || { echo "BUSY A800 physical 0 before flock"; exit 75; }
exec 9>>/dev/shm/gpu-a0.lock
flock -n 9 || { echo "BUSY A800 physical 0 shared lock"; exit 75; }
exec 8>>/dev/shm/formal-7a38ffc-A0.lock
flock -n 8 || { echo "BUSY A800 physical 0 formal lock"; exit 75; }
if [[ -d /dev/shm/unified-gemma-qual-1520 ]]; then
    exec 7>>/dev/shm/unified-gemma-qual-1520/A0.lock
    flock -n 7 || { echo "BUSY A800 physical 0 unified lock"; exit 75; }
fi
idle || { echo "BUSY A800 physical 0 under flock"; exit 75; }
sleep 1
idle || { echo "BUSY A800 physical 0 at second check"; exit 75; }

scratch=$(mktemp -d /dev/shm/ibs-certified-eos.XXXXXXXX)
child=
release() {
    local rc=$?
    trap - EXIT
    rmdir "$scratch" || true
    local apps mem
    apps=$(nvidia-smi -i 0 --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits)
    mem=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
    exec 7>&- 8>&- 9>&-
    printf 'RELEASE rc=%s launcher=%s worker_group=%s gpu=%s memory_mib=%s apps=%q locks=closed date=%s\n' \
        "$rc" "$$" "${child:-none}" "$gpu" "$mem" "$apps" "$(date -Is)"
    exit "$rc"
}
interrupt() {
    if [[ -n $child ]]; then
        kill -TERM "$child" 2>/dev/null || true
        wait "$child" || true
    fi
    exit 130
}
trap release EXIT
trap interrupt HUP INT TERM
export CUDA_VISIBLE_DEVICES="$gpu" IBS_SOURCE="$src" IBS_LOCAL_HEAD="$head" HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$src:$root:$root/search-r1-deps:/gs/env/deps:/gs/env/fixed-baseline-deps"
printf 'LOCKED host=%s pid=%s gpu=%s scratch=%s q=%s date=%s\n' \
    "$(hostname)" "$$" "$gpu" "$scratch" "$q" "$(date -Is)"
timeout -k 15s 2h unshare --mount --propagation private bash -c '
    set -euo pipefail
    scratch=$1
    mount -t tmpfs -o size=256m,nr_inodes=65536,mode=0700,nodev,nosuid tmpfs "$scratch"
    export TMPDIR="$scratch/tmp" HF_HOME="$scratch/hf" TRITON_CACHE_DIR="$scratch/triton"
    export XDG_CACHE_HOME="$scratch/xdg" TORCHINDUCTOR_CACHE_DIR="$scratch/inductor"
    export CUDA_CACHE_PATH="$scratch/cuda"
    mkdir "$TMPDIR" "$HF_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
    cd "$IBS_SOURCE"
    "$2" -B -m unittest discover -s tests -p test_gemma_ibs_certified_eos.py >&2
    exec "$2" -Bu -m tools.run_gemma_ibs_certified_eos --q "$3"
' bash "$scratch" "$python" "$q" < /dev/null &
child=$!
wait "$child"
