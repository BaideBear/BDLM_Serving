#!/usr/bin/env bash
# Offline, no model or GPU. Output path is optional; never writes raw receipts.
set -euo pipefail
rt=/gs/runtime/gemma-continuous-rng-a1-1128
vis=/gs/runtime/gemma-native-visibility-a1-1128
export PYTHONPATH="$rt:$vis:/gs/runtime/eos-handle-1128/repo:/gs/runtime/cold-frontier-last-a1-1128:/gs/env/deps:/gs/env/fixed-baseline-deps"
cd "$rt"
export CUDA_VISIBLE_DEVICES=
/usr/bin/python3 -B -m unittest discover -s tests -p test_gemma_continuous_rng.py
(cd validated_visibility && sha256sum -c ../visibility.sha256)
exec /root/miniconda3/envs/rl/bin/python -B -m experiments.t2t.serving_slots.gemma_continuous_rng_report "${1:-$rt/compact}"
