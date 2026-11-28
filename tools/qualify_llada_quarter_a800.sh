#!/usr/bin/env bash
# A one-shot functional replay (no holder and no takeover).
set -euo pipefail
cd "$(dirname "$0")/.."
gpu=${1:?physical GPU index}
deadline=${2:?stop timestamp}
root=${3:?new receipt directory}
profile=${4:-quarter_fullwrite}
case "$profile" in quarter_fullwrite|quarter_edit_warm|quarter_edit_window) ;;
  quarter_window_repeat64)
    /usr/bin/python3 -B -m experiments.t2t.serving_slots.llada_agreement_eval \
      report /gs/results/llada-q64-paid-agree-1128 benchmarks/frozen90/manifest.json --check-transfer ;;
  *) exit 2 ;; esac
[[ $gpu =~ ^[0-7]$ && $deadline =~ ^[0-9]+$ && ! -e $root ]]
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
[[ -z $(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader,nounits) ]]
seconds=$((deadline-$(date +%s)))
((seconds>0))
export CUDA_VISIBLE_DEVICES="$uuid" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:/gs/env/fixed-baseline-deps" OMP_NUM_THREADS=2
exec timeout -k 5 "$seconds" /gs/env/venv-llada/bin/python -Bu -c \
 'import os,runpy,sys;sys.path.insert(0,os.getcwd());runpy.run_module("experiments.t2t.serving_slots.strict_handle_qualify",run_name="__main__")' \
 --model llada --profile "$profile" --deadline "$deadline" --out "$root" \
 --samples /gs/code/samples.json --checkpoint /gs/models/LLaDA2.1-mini \
 --reference /gs/work/llada-refs/llada21
