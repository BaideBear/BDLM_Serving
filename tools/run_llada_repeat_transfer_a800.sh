#!/usr/bin/env bash
# Follow only our paid Q64 study. No holder and no retry on either outcome.
set -euo pipefail
cd "$(dirname "$0")/.."
deadline=${1:?epoch}
top=/gs/results/llada-q64-paid-agree-1128
q=16
parent="/gs/results/llada-window-q$q-extend-1128"
out="/gs/results/llada-window-paid-agree-q$q-1128"
test ! -e "$out"
uuid=$(nvidia-smi -i 7 --query-gpu=uuid --format=csv,noheader)
get_pids() { nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader,nounits; }
while [[ -n $(get_pids) ]]; do
    for pid in $(get_pids); do
        test -r "/proc/$pid/cmdline" || continue
        tr '\0' '\n' <"/proc/$pid/cmdline" | grep -Fx -- "$top" >/dev/null || exit 3
    done
    (( $(date +%s)<deadline-180 ))
    sleep 5
done
export PYTHONPATH="$PWD:/gs/env/fixed-baseline-deps" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
# The fixed-run report checks the canonical cohort, source registration,
# per-request caps and terminal bills before the transfer gate looks at scores.
/usr/bin/python3 -B -m experiments.t2t.serving_slots.llada_agreement_eval \
    report "$top" benchmarks/frozen90/manifest.json --check-transfer ||
    { echo 'Q64 transfer conditions not established; no inference started'; exit 4; }
/usr/bin/python3 -B -m experiments.t2t.serving_slots.llada_seat_report \
    "$parent" benchmarks/frozen90/manifest.json >/dev/null
remaining=$((deadline-$(date +%s)))
(( remaining>180 ))
test -z "$(get_pids)"
export CUDA_VISIBLE_DEVICES="$uuid" TRITON_CACHE_DIR=/gs/runtime/llada-admission-1128/triton-7
exec timeout -k 5 "$remaining" /gs/env/venv-llada/bin/python -Bu \
    -m experiments.t2t.serving_slots.llada_agreement_eval run --q "$q" \
    --checkpoint /gs/models/LLaDA2.1-mini --samples /gs/code/samples.json \
    --reference /gs/work/llada-refs/llada21 --control "$parent" \
    --control-sha "$(sha256sum "$parent/run.json" | cut -d' ' -f1)" --out "$out" --deadline "$deadline"
