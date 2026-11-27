#!/usr/bin/env bash
# Same-device paired controls; leave the A800 free after the job.
set -euo pipefail
cd "$(dirname "$0")/.."
deadline=${1:?deadline epoch}
[[ $deadline =~ ^[0-9]+$ && $(date +%s) -lt $deadline ]]
[[ -z $(nvidia-smi -i 4 --query-compute-apps=pid --format=csv,noheader,nounits) ]]
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 4 --query-gpu=uuid --format=csv,noheader)"
export PYTHONPATH="$PWD:/gs/env/fixed-baseline-deps" OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/gs/runtime/llada-close-1128/triton PYTHONUNBUFFERED=1
exec timeout -k 3 "$((deadline-$(date +%s)))" /gs/env/venv-llada/bin/python -B -c \
'import os,runpy,sys; sys.path.insert(0,os.getcwd()); runpy.run_module("experiments.t2t.serving_slots.llada_close_factor",run_name="__main__")' \
    run --out /gs/results/llada-close-1128 --samples /gs/code/samples.json \
    --checkpoint /gs/models/LLaDA2.1-mini --reference /gs/work/llada-refs/llada21 \
    --anchors /gs/results/llada-low-dose-a800-gpu2/rows --deadline "$deadline"
