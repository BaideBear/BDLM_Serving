#!/usr/bin/env bash
set -euo pipefail
rt=/gs/runtime/warm-accepted-pay-q32-a1-1128
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="$rt:/gs/runtime/cold-frontier-last-a1-1128:/gs/runtime/verified-eos-1128/repo:/gs/env/deps:/gs/env/fixed-baseline-deps"
cd "$rt"
exec /root/miniconda3/envs/rl/bin/python -B -m experiments.t2t.serving_slots.gemma_warm_accepted_pay_report
