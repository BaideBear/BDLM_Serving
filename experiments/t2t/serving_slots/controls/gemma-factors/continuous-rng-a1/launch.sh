#!/usr/bin/env bash
set -euo pipefail
umask 077
rt=/gs/runtime/gemma-continuous-rng-a1-1128
vis=/gs/runtime/gemma-native-visibility-a1-1128
cd "$rt"
export PYTHONPATH="$rt:$vis:/gs/runtime/eos-handle-1128/repo:/gs/runtime/cold-frontier-last-a1-1128:/gs/env/deps:/gs/env/fixed-baseline-deps"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="$rt/triton"
py=/root/miniconda3/envs/rl/bin/python
CUDA_VISIBLE_DEVICES= "$py" -B -m experiments.t2t.serving_slots.gemma_native_visibility_report "$rt/validated_visibility"
cd "$rt/validated_visibility"
sha256sum -c "$rt/visibility.sha256"
cd "$rt"
test ! -e /proc/240872
test ! -e /gs/results/gemma-continuous-rng-a1-1128
export CUDA_VISIBLE_DEVICES=GPU-861c9b5d-1e73-2550-f832-b73988bc64ff
test "$(nvidia-smi -i 1 --query-gpu=uuid,memory.used --format=csv,noheader,nounits)" = "$CUDA_VISIBLE_DEVICES, 0"
test -z "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits)"
left=$((1795843443-$(date +%s)))
test "$left" -gt 0
exec flock -n "$rt/run.lock" timeout -s KILL "$left" "$py" -Bu -m experiments.t2t.serving_slots.gemma_continuous_rng_eval
