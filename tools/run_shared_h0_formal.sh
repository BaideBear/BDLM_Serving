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
if [[ -n ${BDLM_AFTER_H0_MECHANISM:-} ]]; then
    [[ $# == 5 && -z ${BDLM_AFTER_H0_FORMAL:-} && -z ${BDLM_AFTER_H0_SCREEN:-} ]]
    wait_limit=$((end-$(date +%s)-180))
    (( wait_limit > 0 ))
    flock -w "$wait_limit" 9
    [[ -s $BDLM_AFTER_H0_MECHANISM/H0/returned ]]
    "$py" -Bm tools.shared_mechanism_report "$reg" "$BDLM_AFTER_H0_MECHANISM" --board n164:H0 --require-complete >/dev/null
elif [[ -n ${BDLM_AFTER_H0_SCREEN:-} ]]; then
    [[ $# == 6 && -z ${BDLM_AFTER_H0_FORMAL:-} ]]
    wait_limit=$((end-$(date +%s)-180))
    (( wait_limit > 0 ))
    flock -w "$wait_limit" 9
    [[ -s $BDLM_AFTER_H0_SCREEN/H0/returned ]]
    "$py" -B - "$BDLM_AFTER_H0_SCREEN" "$reg" "$6" <<'PY'
import json, sys
from pathlib import Path
from tools.shared_screen_report import MANIFEST, audit
root, base, panel = map(Path, sys.argv[1:])
r = audit(root, json.loads(MANIFEST.read_bytes())["samples"],
          json.loads(base.read_bytes()), json.loads(panel.read_bytes()))
for model, ledger in r["models"].items():
    if ledger["by_board"]["n164:H0"] != 60 or any(e.startswith("n164:H0/") for e in ledger["deferred"]):
        raise ValueError(f"H0 predecessor incomplete: {model}")
PY
elif [[ -n ${BDLM_AFTER_H0_FORMAL:-} ]]; then
    [[ $# == 6 ]]
    wait_limit=$((end-$(date +%s)-180))
    (( wait_limit > 0 ))
    flock -w "$wait_limit" 9
    [[ -s $BDLM_AFTER_H0_FORMAL/returned ]]
    # Check the released shard, not just the existence of a supervisor marker.
    "$py" -B - "$BDLM_AFTER_H0_FORMAL" "$reg" <<'PY'
import json,sys
from pathlib import Path
from experiments.t2t.serving_slots.shared_fixed_shard import OFFERS, MANIFEST, identity, shard_cases
from experiments.t2t.serving_slots.shared_point_metrics import _receipt
root, reg = map(Path, sys.argv[1:])
version = identity(json.loads(reg.read_bytes()))
shard = shard_cases(json.loads(MANIFEST.read_bytes())["samples"], "n164:H0", "formal90")
for model, tiers in OFFERS.items():
    want={(r["index"], q):r for r in shard for q in tiers}
    with (root/(model+".ndjson")).open() as stream:
        for line in stream:
            r=json.loads(line); case=want.pop((r["index"],r["offer"]))
            assert (r["stage"],r["version"],r["board"],r["model"])==("formal90",version,"n164:H0",model)
            assert all(r[k]==case[k] for k in ("seed","dataset","task_id","prompt_sha256"))
            _receipt(r,model,r["offer"])
    assert not want
PY
else
    flock -n 9
fi
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
args=(--board n164:H0 --version "$version" --samples "$rt/t2t-native-1125/prefix90/samples.json")
expected=75
preflight=(run)
preflight_tail=(--preflight)
if [[ -n ${BDLM_AFTER_H0_MECHANISM:-} ]]; then
    module=tools.shared_successor_probe
    args+=(--base "$reg")
    expected=6
    preflight=(preflight)
    preflight_tail=()
elif [[ -n ${BDLM_AFTER_H0_SCREEN:-} ]]; then
    module=tools.shared_prefix_mechanism
    args+=(--base "$reg")
    expected=18
    preflight=(preflight)
    preflight_tail=()
elif [[ $# == 6 ]]; then
    module=tools.shared_screen_sweep
    args+=(--base "$reg" --panel "$(realpath "$6")")
    expected=60
elif [[ $# == 5 ]]; then
    args+=(--registration "$reg")
else
    exit 2
fi
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
    "$py" -Bm "$module" "${preflight[@]}" "${args[@]}" --model "$model" --checkpoint "$ckpt" "${preflight_tail[@]}"
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
    [[ $(wc -l <"$out/$model.ndjson") -eq $expected ]]
done
