#!/usr/bin/env bash
# Does not signal the predecessor, allocate a holder, or wait for arbitrary GPU users.
set -euo pipefail
cd "$(dirname "$0")/.."
gpu=${1:?GPU index}; q=${2:?offer}; old=${3:?registered predecessor PID}
birth=${4:?registered predecessor start tick}; deadline=${5:?epoch}
case "$gpu/$q" in 4/32|6/64) ;; *) exit 2 ;; esac
[[ $old =~ ^[0-9]+$ && $birth =~ ^[0-9]+$ && $deadline =~ ^[0-9]+$ ]]
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
case "$gpu" in
  4) pattern=experiments.t2t.serving_slots.llada_close_factor ;;
  6) pattern=experiments.t2t.serving_slots.gemma_cold_pacing_eval ;;
esac
if [[ -r /proc/$old/stat && $(awk '{print $22}' "/proc/$old/stat") == "$birth" ]]; then
  tr '\0' ' ' < "/proc/$old/cmdline" | grep -F "$pattern" >/dev/null
  echo "Waiting without a GPU allocation for $pattern PID $old on $uuid"
  while [[ -r /proc/$old/stat && $(awk '{print $22}' "/proc/$old/stat" 2>/dev/null || true) == "$birth" ]]; do
    (( $(date +%s)<deadline )) || exit 3
    sleep 5
  done
fi
for _ in {1..12}; do
  live=$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z $live ]] && break
  [[ $live == "$old" ]] || { echo "foreign process; stopping"; exit 4; }
  sleep 2
done
[[ -z $live ]]
remaining=$((deadline-$(date +%s)))
(( remaining>0 ))
export CUDA_VISIBLE_DEVICES="$uuid" OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:/gs/env/fixed-baseline-deps" TRITON_CACHE_DIR="/gs/runtime/llada-admission-1128/triton-$q"
exec timeout -k 3 "$remaining" /gs/env/venv-llada/bin/python -B -c \
'import os,runpy,sys;sys.path.insert(0,os.getcwd());runpy.run_module("experiments.t2t.serving_slots.llada_admission_factor",run_name="__main__")' \
 run --out "/gs/results/llada-admission-c$q-1128" --offer "$q" --history /gs/results \
 --reference /gs/work/llada-refs/llada21 --checkpoint /gs/models/LLaDA2.1-mini \
 --samples /gs/code/samples.json --deadline "$deadline"
