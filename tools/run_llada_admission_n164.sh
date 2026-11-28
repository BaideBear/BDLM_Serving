#!/usr/bin/env bash
# One H800 lane; only its identified holder is eligible for handoff.
set -euo pipefail
cd "$(dirname "$0")/.."
rt=/data1/project_data/gx-erqw-20261123/runtime
hold=/data1/project_data/gx-megrez-20261122/gpu-holders/gpu-hold
uuid=GPU-ceadb3cd-e63f-1446-a2d5-a291144f8efb
deadline=${1:?deadline}
if [[ ${2:-} == spare_waiting8 && $# == 3 ]]; then
  prev=$rt/llada-window-fallback-q8-1128
  pid=$3
  [[ $pid =~ ^[1-9][0-9]*$ ]]
  if [[ -r /proc/$pid/cmdline ]]; then
    tr '\0' '\n' <"/proc/$pid/cmdline" | grep -Fx "$prev/results" >/dev/null
    stamp=$(awk '{print $22}' "/proc/$pid/stat")
    while [[ -r /proc/$pid/stat ]]; do
      now=$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null) || break
      [[ $now == "$stamp" ]] || break
      (( $(date +%s)<deadline-180 ))
      sleep 4
    done
  fi
  (cd "$prev/repo" && /usr/bin/python3 -B -m experiments.t2t.serving_slots.llada_window_fallback \
      report "$prev/results" benchmarks/frozen90/manifest.json) |
    /usr/bin/python3 -c 'import json,sys; r=json.load(sys.stdin); assert r["finished"]==90 and r["cohorts"]["frozen90"]["n"]==90'
  # The preceding launcher's EXIT trap may still be restoring the holder.
  until "$hold" status 0 | /usr/bin/python3 -c 'import json,sys; r,=json.load(sys.stdin); sys.exit(not (r["running"] and r["phase"]=="running"))'; do
    (( $(date +%s)<deadline-180 ))
    sleep 2
  done
fi
left=$((deadline-$(date +%s)-5))
((left>0))
case "${2:-admission16}" in
  admission16) dir=$rt/llada-admission-1128
    module=experiments.t2t.serving_slots.llada_admission_factor
    opts=(run --offer 16 --history "$rt/layout-1128-v1/cross/results") ;;
  frontier4) dir=$rt/llada-frontier-q4-1128
    module=experiments.t2t.serving_slots.llada_frontier_seat
    opts=(--offer 4 --limit 90) ;;
  paid8) dir=$rt/llada-window-paid-q8-1128
    module=experiments.t2t.serving_slots.llada_frontier_seat
    opts=(--offer 8 --factor paid --limit 90) ;;
  fallback8) dir=$rt/llada-window-fallback-q8-1128
    module=experiments.t2t.serving_slots.llada_window_fallback
    opts=(run --parent "$rt/llada-window-paid-q8-1128/results") ;;
  spare_waiting8) dir=$rt/llada-window-spare-waiting-q8-1128
    module=experiments.t2t.serving_slots.llada_window_spare
    opts=(run --q 8 --factor waiting --parent "$rt/llada-window-paid-q8-1128/results") ;;
  spare_prefix8) dir=$rt/llada-window-spare-prefix-q8-1128
    module=experiments.t2t.serving_slots.llada_window_spare
    opts=(run --q 8 --factor prefix --parent "$rt/llada-window-paid-q8-1128/results") ;;
  *) exit 2 ;;
esac
[[ ! -e $dir/results ]]
exec 9>"$rt/llada-admission-1128/h0.lock"
control=$(mktemp -d "$dir/handoff.XXXXXX")
observe(){
  "$hold" status 0 | python3 -c 'import json,sys; r,=json.load(sys.stdin); assert r["running"] and r["phase"]=="running" and r["instance"]["gpu_uuid"]==sys.argv[1]; print(r["pid"])' "$uuid"
}
apps(){ nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader,nounits; }
pid=$(observe)
[[ $(apps) == "$pid" ]]
flock -n 9 || exit 1
released=no
worker=
cleanup(){
  if [[ -n $worker ]] && kill -0 "$worker" 2>/dev/null; then
    kill -TERM "$worker" 2>/dev/null || true
    wait "$worker" 2>/dev/null || true
  fi
  if [[ $released == yes ]]; then "$hold" start 0; fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT
export CUDA_VISIBLE_DEVICES="$uuid" PYTHONPATH="$PWD:$rt/fixed-baseline-deps" OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1
export REQUEST_STUDY="$module"
timeout -k 4 "$left" "$rt/venv/bin/python" -Bu -c \
'import os,runpy,sys;sys.path.insert(0,os.getcwd());runpy.run_module(os.environ["REQUEST_STUDY"],run_name="__main__")' \
 "${opts[@]}" --out "$dir/results" --samples "$rt/t2t-native-1125/prefix90/samples.json" \
 --checkpoint /data0/ysy/models/LLaDA2.1-mini --reference "$rt/t2t-native-1125/frozen90-native/llada21" \
 --handoff "$control" --deadline "$deadline" &
worker=$!
while [[ ! -f $control/ready ]]; do kill -0 "$worker"; sleep .25; done
[[ $(observe) == "$pid" && $(apps) == "$pid" ]]
released=yes
"$hold" stop 0
touch "$control/go"
wait "$worker"
