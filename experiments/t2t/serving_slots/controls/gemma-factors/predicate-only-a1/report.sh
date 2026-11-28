#!/usr/bin/env bash
set -euo pipefail
rt=/gs/runtime/gemma-predicate-only-a1-1128
cd "$rt"
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="$rt:/gs/runtime/gemma-continuous-rng-a1-1128:/gs/runtime/gemma-native-visibility-a1-1128:/gs/runtime/eos-handle-1128/repo:/gs/runtime/cold-frontier-last-a1-1128:/gs/env/deps:/gs/env/fixed-baseline-deps"
/usr/bin/python3 -B -m unittest discover -s tests -p test_gemma_predicate_only.py
(cd validated_rng && sha256sum -c ../rng.sha256)
exec /root/miniconda3/envs/rl/bin/python -B -m experiments.t2t.serving_slots.gemma_predicate_only_report "${1:-$rt/compact}" "${@:2}"
