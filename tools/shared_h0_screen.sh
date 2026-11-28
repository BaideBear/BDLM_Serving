#!/usr/bin/env bash
# Exclusive n164 H0 staging, followed by checked holder return on any exit.
set -euo pipefail
umask 077
out=${1:?unused n164 results directory}
export BDLM_WINDOW_END=${2:?authorized end epoch}
export BDLM_CODE_DIGEST=${3:?source digest}
export DG_SIGNAL=$out CUDA_VISIBLE_DEVICES=GPU-ceadb3cd-e63f-1446-a2d5-a291144f8efb
rt=/data1/project_data/gx-erqw-20261123/runtime
py=$rt/venv/bin/python
hold=/data1/project_data/gx-megrez-20261122/gpu-holders/gpu-hold
export PYTHONPATH="$PWD:$rt/diffusiongemma-native-20261125/deps:$rt/fixed-baseline-deps"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
case ${4:-screen} in
    screen) module=experiments.t2t.serving_slots.shared_h0_screen; args=() ;;
    ibs-bridge) module=tools.gemma_ibs_bridge_replay; args=(--staged-h0 --panel "${5:?panel}" --out "$out" --samples "$rt/t2t-native-1125/prefix90/samples.json") ;;
    *) exit 2 ;;
esac
pids(){ nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits | sort -n | xargs; }
owner(){ "$hold" status 0 | python3 -c 'import json,os,sys; h,=json.load(sys.stdin); i=h["instance"]; assert h["running"] and i["uid"]==os.getuid() and h["gpu_status"]["uuid"]==i["gpu_uuid"]==os.environ["CUDA_VISIBLE_DEVICES"] and i["pid"]==h["pid"] and i["boot_id"]==open("/proc/sys/kernel/random/boot_id").read().strip() and int(open("/proc/%d/stat"%h["pid"]).read().split(") ")[1].split()[19])==i["start_ticks"]; print(i["pid"])'; }
initial=$(owner)
test "$(pids)" = "$initial"
test "$(date +%s)" -lt "$((BDLM_WINDOW_END-60))"
mkdir "$out"
worker= released=0
restore(){
    trap - EXIT INT TERM HUP
    if [[ -n "$worker" ]] && kill -0 "$worker" 2>/dev/null; then
        kill -TERM "$worker" 2>/dev/null || true
        wait "$worker" 2>/dev/null || true
    fi
    if [[ "$released" == 1 ]]; then
        if [[ -z "$(pids)" ]]; then "$hold" start 0 --timeout 30 >&2; fi
        new=$(owner)
        test "$(pids)" = "$new"
        printf '%s\n' "$new" >"$out/returned"
    fi
}
trap restore EXIT
trap 'exit 130' INT TERM HUP
"$py" -Bu -m "$module" "${args[@]}" >"$out/receipts.ndjson" 2>"$out/stderr.log" &
worker=$!
printf '%s\n' "$worker" >"$out/worker.pid"
until [[ -e "$out/ready" ]]; do
    kill -0 "$worker"
    test "$(date +%s)" -lt "$((BDLM_WINDOW_END-30))"
    sleep 1
done
test "$(owner)" = "$initial"
test "$(pids)" = "$(printf '%s\n%s\n' "$initial" "$worker" | sort -n | xargs)"
released=1
"$hold" stop 0 >&2
test "$(pids)" = "$worker"
touch "$out/go"
wait "$worker"
worker=
