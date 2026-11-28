#!/usr/bin/env bash
# Run inside a pinned source checkout. Fresh bounded tmpfs; never reserve a card.
set -euo pipefail
root=${1:?new directory in /dev/shm}
study=${2:-equal_dose}
case "$study" in
  equal_dose|payment_observer) ;;
  *) exit 2 ;;
esac
[[ "$root" =~ ^/dev/shm/gemma-(equal-dose|payment-observer)-[a-zA-Z0-9-]+$ ]]
mkdir -m700 "$root"
exec 9>>/dev/shm/gpu-a3.lock
flock -n 9 || exit 75
[[ -z $(nvidia-smi -i 3 --query-compute-apps=pid --format=csv,noheader,nounits) ]]
export CUDA_VISIBLE_DEVICES=GPU-94c61e79-d80b-cae2-d85f-6bd2a2c99a1c
export TMPDIR="$root" HF_HOME="$root" TRITON_CACHE_DIR=/dev/shm/triton-a3
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD:/gs/work/gemma-strict-ibs-shadow-sf:/gs/env/deps:/gs/env/fixed-baseline-deps"
rc=0
timeout -k 10s 30m /root/miniconda3/envs/rl/bin/python -Bu \
  -m "experiments.t2t.serving_slots.gemma_$study" \
  --root "$root" --deadline "$(($(date +%s)+1800))" >"$root/stdout.log" 2>"$root/stderr.log" || rc=$?
printf '%s\n' "$rc" >"$root/runner.exit"
exit "$rc"
