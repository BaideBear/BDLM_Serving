#!/usr/bin/env bash
set -euo pipefail
rt=/gs/runtime/gemma-ibs-prefix-bridge-a1-1128
src=/gs/work/gemma-ibs512-1127
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="$rt:$src:$src/search-r1-deps:/gs/env/deps:/gs/env/fixed-baseline-deps:/gs/runtime/verified-eos-1128/repo"
cd "$rt"
exec /root/miniconda3/envs/search-r1/bin/python -B -m experiments.t2t.serving_slots.gemma_ibs_prefix_bridge_report
