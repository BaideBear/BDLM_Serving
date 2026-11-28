#!/usr/bin/env bash
# n164 H0 only. Source and registration are frozen before handing off the board.
set -euo pipefail
umask 077
src=$(realpath "${1:?clean registered source}")
reg=$(realpath "${2:?registration.json}")
version=${3:?registration id}
end=${4:?authorized end epoch}
out=${5:?new n164 ledger directory}
base=/data1/project_data/gx-erqw-20261123
rt=$base/runtime
py=$rt/venv/bin/python
hold=/data1/project_data/gx-megrez-20261122/gpu-holders/gpu-hold
export CUDA_VISIBLE_DEVICES=GPU-ceadb3cd-e63f-1446-a2d5-a291144f8efb
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export BDLM_LLADA_REFERENCE=$rt/t2t-native-1125/frozen90-native/llada21
module=experiments.t2t.serving_slots.shared_fixed_shard
cd "$src"
exec 9>"$rt/shared-formal-H0.lock"
flock -n 9
mkdir "$out"
pidlist() {
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits | sort -n | xargs
}
owner() {
    "$hold" status 0 | "$py" -c 'import json,os,sys
h,=json.load(sys.stdin); i=h["instance"]; p=h["pid"]
assert h["running"] and i["uid"]==os.getuid() and p==i["pid"]
assert h["gpu_status"]["uuid"]==i["gpu_uuid"]==os.environ["CUDA_VISIBLE_DEVICES"]
assert i["boot_id"]==open("/proc/sys/kernel/random/boot_id").read().strip()
assert i["start_ticks"]==int(open("/proc/%d/stat"%p).read().split(") ")[1].split()[19])
print(p)'
}
args=(--board n164:H0 --registration "$reg" --version "$version" --samples "$rt/t2t-native-1125/prefix90/samples.json")
config() {
    export PYTHONPATH="$src:$rt/fixed-baseline-deps"
    ckpt=/data0/ysy/models/LLaDA2.1-mini
    if [[ $1 == gemma ]]; then
        export PYTHONPATH="$src:$rt/diffusiongemma-native-20261125/deps:$rt/fixed-baseline-deps"
        ckpt=$base/models/diffusiongemma-cd40195
    fi
}
for model in gemma llada21; do
    config "$model"
    "$py" -Bm "$module" run "${args[@]}" --model "$model" --checkpoint "$ckpt" --preflight
done
initial=$(owner)
[[ $(pidlist) == "$initial" && $(date +%s) -lt $((end-180)) ]]
released=0
worker=
restore() {
    rc=$?
    trap - EXIT TERM INT HUP
    if [[ -n $worker ]]; then
        kill -TERM "$worker" 2>/dev/null || true
        wait "$worker" || true
    fi
    if (( released )); then
        if [[ -z $(pidlist) ]]; then "$hold" start 0 --timeout 30 >&2; fi
        new=$(owner)
        [[ $(pidlist) == "$new" ]]
        printf '%s\n' "$new" >"$out/returned"
    fi
    exit "$rc"
}
trap restore EXIT
trap 'exit 130' INT TERM HUP
released=1
"$hold" stop 0 >&2
for model in gemma llada21; do
    [[ -z $(pidlist) ]]
    config "$model"
    left=$((end-$(date +%s)-90))
    (( left > 60 ))
    timeout -k 20 "$left" "$py" -Bu -m "$module" run "${args[@]}" --model "$model" --checkpoint "$ckpt" \
        >"$out/$model.ndjson" 2>"$out/$model.log" &
    worker=$!
    printf '%s\n' "$worker" >"$out/$model.supervisor.pid"
    wait "$worker"
    worker=
    [[ $(wc -l <"$out/$model.ndjson") -eq 75 ]]
done
