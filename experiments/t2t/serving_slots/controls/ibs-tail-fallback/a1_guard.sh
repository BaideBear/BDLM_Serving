#!/usr/bin/env bash
# Invoked by relay.sh inside its new private /dev/shm directory.
set -euo pipefail
umask 077
w=$PWD
deadline=${1:?deadline}
sha=${2:?runner hash}
uuid=GPU-861c9b5d-1e73-2550-f832-b73988bc64ff
[[ $(hostname -s) == n1 && $w == /dev/shm/ibs-tail-fallback.* ]]
[[ $sha =~ ^[a-f0-9]{64}$ && $deadline =~ ^[0-9]+$ ]]
[[ $(sha256sum study.py | cut -d' ' -f1) == "$sha" ]]
((deadline-$(date +%s)>120 && deadline-$(date +%s)<=3600))
idle() {
    [[ $(nvidia-smi -i 1 --query-gpu=index,uuid,memory.used,utilization.gpu \
            --format=csv,noheader,nounits) == "1, $uuid, 0, 0" ]]
    [[ -z $(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader,nounits) ]]
}
idle
exec 8>>/dev/shm/gpu-a1.tail-overlap-v1.lock
flock -n 8
idle
export CUDA_VISIBLE_DEVICES="$uuid" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export TMPDIR="$w/tmp" HF_HOME="$w/hf" TORCH_HOME="$w/torch" CUDA_CACHE_PATH="$w/cuda" TRITON_CACHE_DIR="$w/triton"
export XDG_CACHE_HOME="$w/cache" TORCHINDUCTOR_CACHE_DIR="$w/inductor" TORCH_EXTENSIONS_DIR="$w/extensions"
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" \
    "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
worker= watch=
cleanup() {
    if [[ -n $worker ]]; then kill -TERM "$worker" 2>/dev/null || :; fi
    if [[ -n $watch ]]; then kill -TERM "$watch" 2>/dev/null || :; fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT HUP
/gs/env/venv-llada/bin/python -Bu study.py run reference.json "$deadline" "$sha" &
worker=$!
echo "A1_START worker=$worker uuid=$uuid deadline=$deadline" >&2
(
    while kill -0 "$worker" 2>/dev/null; do
        if ! apps=$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader,nounits); then
            reason=query-failed
        elif [[ -n $(grep -Ev "^ *($worker)? *$" <<<"$apps") ]]; then
            reason=foreign-pid
        elif (( $(date +%s) >= deadline )); then
            reason=deadline
        elif (( $(du -sk "$w" | cut -f1) > 262144 )); then
            reason=scratch-cap
        else
            sleep 5
            continue
        fi
        echo "A1_GUARD_STOP reason=$reason worker=$worker" >&2
        kill -TERM "$worker" 2>/dev/null || :
        break
    done
) &
watch=$!
set +e
wait "$worker"
rc=$?
set -e
worker=
echo "RUNNER_EXIT=$rc" >&2
exit "$rc"
