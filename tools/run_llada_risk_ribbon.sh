#!/usr/bin/env bash
# One GPU, one bounded, same-card fixed-offer paired pilot; no holder.
set -euo pipefail
q=${1:?offer}; gpu=${2:?physical board}; deadline=${3:?deadline epoch}; root=${4:?overlay root}
case "${5:-risk}" in
  risk) module=llada_risk_ribbon; prefix=llada-risk-ribbon-v1 ;;
  span) module=coverage_span_eval; prefix=llada-coverage-span-v1 ;;
  confirm) module=coverage_span_report ;;
  *) exit 2 ;;
esac
base=/gs/runtime/llada-frontier32-qualify-1128
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
[[ -z $(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader,nounits) ]]
[[ $(nvidia-smi -i "$uuid" --query-gpu=memory.used --format=csv,noheader,nounits) -lt 100 ]]
remain=$((deadline-$(date +%s)))
((remain>120))
export CUDA_VISIBLE_DEVICES="$uuid" OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$base:/gs/env/fixed-baseline-deps"
cd "$root"
if [[ "$module" == coverage_span_report ]]; then
  exec timeout -k 5 "$remain" /gs/env/venv-llada/bin/python -Bu -m "experiments.t2t.serving_slots.$module" \
    /gs/results benchmarks/frozen90/manifest.json --confirm "$q" --checkpoint /gs/models/LLaDA2.1-mini \
    --reference /gs/work/llada-refs/llada21 --samples /gs/code/samples.json
fi
exec timeout -k 5 "$remain" /gs/env/venv-llada/bin/python -Bu -m "experiments.t2t.serving_slots.$module" \
  --q "$q" --checkpoint /gs/models/LLaDA2.1-mini --reference /gs/work/llada-refs/llada21 \
  --samples /gs/code/samples.json --out "/gs/results/$prefix-q$q"
