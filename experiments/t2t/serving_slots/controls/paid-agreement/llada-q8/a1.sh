#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=${1:?}
deadline=${2:?}
uuid=GPU-861c9b5d-1e73-2550-f832-b73988bc64ff
[[ $(hostname -s) == n1 && $PWD == /dev/shm/llada-q8-paid.* ]]
[[ $stage == six || $stage == remaining ]]
(( deadline > $(date +%s) + 60 && deadline <= $(date +%s) + 7200 ))
idle() {
    [[ $(nvidia-smi -i 1 --query-gpu=index,uuid,memory.used,utilization.gpu \
        --format=csv,noheader,nounits) == "1, $uuid, 0, 0" ]]
    [[ -z $(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits) ]]
}
idle
exec 8>>/dev/shm/gpu-a1.tail-overlap-v1.lock
flock -n 8
idle
export CUDA_VISIBLE_DEVICES="$uuid" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/gs/code/formal90-7a38ffc/src:/gs/env/fixed-baseline-deps
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export TMPDIR="$PWD/tmp" HF_HOME="$PWD/hf" TORCH_HOME="$PWD/torch" CUDA_CACHE_PATH="$PWD/cuda"
export TRITON_CACHE_DIR="$PWD/triton" XDG_CACHE_HOME="$PWD/cache"
export TORCHINDUCTOR_CACHE_DIR="$PWD/inductor" TORCH_EXTENSIONS_DIR="$PWD/extensions"
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" \
    "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
pid= watch=
cleanup() {
    if [[ -n $pid ]]; then kill -TERM "$pid" 2>/dev/null || :; fi
    if [[ -n $watch ]]; then kill -TERM "$watch" 2>/dev/null || :; fi
}
trap cleanup EXIT
trap 'exit 143' HUP INT TERM
/gs/env/venv-llada/bin/python -Bu study.py run "$stage" reference.json "$deadline" &
pid=$!
printf 'A1_START stage=%s pid=%s uuid=%s\n' "$stage" "$pid" "$uuid" >&2
(
    while kill -0 "$pid" 2>/dev/null; do
        if ! pids=$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits); then
            reason=nvidia-smi
        elif [[ -n $(grep -Ev "^ *($pid)? *$" <<<"$pids") ]]; then
            reason=foreign-pid
        elif (( $(date +%s) >= deadline )); then
            reason=deadline
        elif (( $(du -sk "$PWD" | cut -f1) > 262144 )); then
            reason=scratch-cap
        else
            sleep 5
            continue
        fi
        printf 'A1_STOP stage=%s reason=%s\n' "$stage" "$reason" >&2
        kill -TERM "$pid" 2>/dev/null || :
        break
    done
) &
watch=$!
set +e
wait "$pid"
rc=$?
set -e
pid=
printf 'A1_EXIT stage=%s rc=%s\n' "$stage" "$rc" >&2
exit "$rc"
