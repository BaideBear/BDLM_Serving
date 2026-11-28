#!/usr/bin/env bash
set -euo pipefail
umask 077
rt=/gs/runtime/gemma-predicate-only-a1-1128
rng=/gs/runtime/gemma-continuous-rng-a1-1128
cd "$rt"
bash "$rng/report.sh" "$rt/validated_rng"
cd "$rt/validated_rng"
sha256sum -c "$rt/rng.sha256"
cd "$rt"
test ! -e /proc/258557
test ! -e /gs/results/gemma-predicate-only-a1-1128
export CUDA_VISIBLE_DEVICES=GPU-861c9b5d-1e73-2550-f832-b73988bc64ff
test "$(nvidia-smi -i 1 --query-gpu=uuid,memory.used --format=csv,noheader,nounits)" = "$CUDA_VISIBLE_DEVICES, 0"
test -z "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits)"
export PYTHONPATH="$rt:$rng:/gs/runtime/gemma-native-visibility-a1-1128:/gs/runtime/eos-handle-1128/repo:/gs/runtime/cold-frontier-last-a1-1128:/gs/env/deps:/gs/env/fixed-baseline-deps"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="$rt/triton"
left=$((1795843443-$(date +%s)))
test "$left" -gt 0
exec flock -n "$rt/run.lock" timeout -s KILL "$left" /root/miniconda3/envs/rl/bin/python -Bu -m experiments.t2t.serving_slots.gemma_predicate_only_eval
