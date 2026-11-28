#!/usr/bin/env bash
set -euo pipefail
umask 077
src=${1:?source}
[[ $src =~ ^/gs/work/gemma-q32-lastmiss-a3-[a-zA-Z0-9]+$ && $(hostname -s) == n1 ]]
root=/gs/work/gemma-paid-agreement-q128
parent=/gs/work/gemma-live-q128-a3-7iDh2EqM
gpu=GPU-94c61e79-d80b-cae2-d85f-6bd2a2c99a1c
fmt=--format=csv,noheader,nounits
idle() {
    local state apps
    state=$(nvidia-smi -i 3 --query-gpu=uuid,memory.used,utilization.gpu "$fmt") || return 1
    apps=$(nvidia-smi -i 3 --query-compute-apps=pid "$fmt") || return 1
    [[ $state == "$gpu, 0, 0" && -z $apps ]]
}
(cd "$src" && sha256sum --strict -c SHA256SUMS)
printf '%s  %s\n' \
    9244a6d1efbf89e5ad9142bc32435deb568b69592ecc736db44da478255c1a9f "$parent/run.py" \
    669b0d577aece64e5b56cf9b13d3f17b08abe8d7b4e6bfea5ceeef4285777018 "$parent/reference.py" \
    5a1e9c122d5c9b4c850d896b6608554c1ce85e7c729d6686d0a44be4fc19b2b1 "$parent/live-runtime.sha256" \
    cded57b7e365f0265b1c088e993c358b5f70acdf48055fce396540b4d9d691bc \
    /gs/env/deps/transformers/models/diffusion_gemma/generation_diffusion_gemma.py \
    9cf41b0a89a51ee3c1eab33605963cf5616579b9043edae69fecad836a876c8a \
    /gs/runtime/gemma-ibs-prefix-bridge-a1-1128/checkpoint.json | sha256sum -c -
(cd "$root" && sha256sum --strict -c "$parent/live-runtime.sha256")
idle || exit 75
exec 9>>/dev/shm/gpu-a3.lock
flock -n 9 || exit 75
idle || exit 75
sleep 1
idle || exit 75
scratch=$(mktemp -d /dev/shm/gemma-q32-lastmiss.XXXXXXXX)
release() {
    local rc=$? state apps
    trap - EXIT
    rmdir "$scratch" || rc=74
    state=$(nvidia-smi -i 3 --query-gpu=uuid,memory.used,utilization.gpu "$fmt") || state=UNKNOWN
    apps=$(nvidia-smi -i 3 --query-compute-apps=pid "$fmt") || apps=UNKNOWN
    exec 9>&-
    printf 'RELEASE rc=%s state=%q apps=%q lock=closed date=%s\n' "$rc" "$state" "$apps" "$(date -Is)"
    exit "$rc"
}
trap release EXIT
export CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$root:/gs/work/gemma-strict-ibs-shadow-sf:/gs/env/deps:/gs/env/fixed-baseline-deps"
cd "$root"
timeout -k 15s 14m unshare --mount --propagation private bash -c '
    set -euo pipefail
    mount -t tmpfs -o size=128m,nr_inodes=65536,mode=0700,nodev,nosuid tmpfs "$1"
    export TMPDIR="$1" HF_HOME="$1/hf" TRITON_CACHE_DIR="$1/triton"
    export XDG_CACHE_HOME="$1/xdg" TORCHINDUCTOR_CACHE_DIR="$1/inductor" CUDA_CACHE_PATH="$1/cuda"
    exec /root/miniconda3/envs/rl/bin/python -Bu "$2/run.py"
' bash "$scratch" "$src"
