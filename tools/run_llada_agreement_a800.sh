#!/usr/bin/env bash
# Claim one empty A7 after the complete switch qualification. Never run a holder.
set -euo pipefail
cd "$(dirname "$0")/.."
deadline=${1:?epoch}
qualified=/gs/results/llada-switch-window-api-replay-1128
prior=/gs/results/llada-quarter-successor-q64-1128
out=/gs/results/llada-q64-paid-agree-1128
[[ $deadline =~ ^[0-9]+$ && ! -e "$out" ]]
/usr/bin/python3 -B -m experiments.t2t.serving_slots.strict_handle_report \
  llada "$qualified" benchmarks/frozen90/manifest.json >/dev/null
remaining=$((deadline-$(date +%s)))
(( remaining>120 ))
uuid=$(nvidia-smi -i 7 --query-gpu=uuid --format=csv,noheader)
[[ -z $(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader,nounits) ]]
export CUDA_VISIBLE_DEVICES="$uuid" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
export PYTHONPATH="$PWD:/gs/env/fixed-baseline-deps"
exec timeout -k 5 "$remaining" /gs/env/venv-llada/bin/python -Bu \
  -m experiments.t2t.serving_slots.llada_agreement_eval run \
  --checkpoint /gs/models/LLaDA2.1-mini --samples /gs/code/samples.json \
  --reference /gs/work/llada-refs/llada21 --out "$out" --control "$prior" \
  --control-sha f3fe1f2508342849e9c270d7cb54183a63869e852238098ad244c3016f95ef19 --deadline "$deadline"
