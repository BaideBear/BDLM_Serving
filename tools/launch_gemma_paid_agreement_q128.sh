#!/usr/bin/env bash
# A3-only Q128 certified paid-EOS + paid-agreement six-case pairing; no holder.
set -euo pipefail
deadline=${1:?deadline epoch}
root=${2:?/dev/shm output directory}
[[ $deadline =~ ^[0-9]+$ && $(date +%s) -lt $deadline && $root == /dev/shm/* ]]
uuid=GPU-94c61e79-d80b-cae2-d85f-6bd2a2c99a1c
[[ -z $(nvidia-smi -i 3 --query-compute-apps=pid --format=csv,noheader,nounits) ]]
mkdir -p "$root" /dev/shm/triton-a3 /dev/shm/tmp-a3 /dev/shm/hf-a3
exec 9>/dev/shm/gpu-a3.lock
flock -n 9 || { echo "GPU3 lock busy"; exit 1; }
export CUDA_VISIBLE_DEVICES="$uuid" TRITON_CACHE_DIR=/dev/shm/triton-a3
export TMPDIR=/dev/shm/tmp-a3 HF_HOME=/dev/shm/hf-a3 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
export PYTHONPATH=/gs/work/gemma-paid-agreement-q128:/gs/work/gemma-strict-ibs-shadow-sf:/gs/env/deps:/gs/env/fixed-baseline-deps
exec timeout -k 5 "$((deadline - $(date +%s)))" /root/miniconda3/envs/rl/bin/python -u -B \
  -m experiments.t2t.serving_slots.gemma_paid_agreement_eval \
  --root "$root" --deadline "$deadline" > "$root/stdout.log" 2> "$root/stderr.log"
