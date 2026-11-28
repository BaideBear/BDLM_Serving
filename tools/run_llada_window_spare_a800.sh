#!/usr/bin/env bash
# Only follow the identified window process on the same GPU, never reserve it.
set -euo pipefail
cd "$(dirname "$0")/.."
q=${1:?offer}
deadline=${2:?epoch}
factor=${3:-interior}
case $factor in interior|contested) ;; *) exit 2;; esac
case $q in 8) gpu=6;; 16) gpu=5;; *) exit 2;; esac
parent="/gs/results/llada-window-q$q-extend-1128"
out="/gs/results/llada-window-spare-q$q-1128"
if [[ $factor == contested ]]; then out="/gs/results/llada-window-contested-q$q-1128"; fi
test ! -e "$out"
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
get_pids() { nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader,nounits; }
while [[ -n $(get_pids) ]]; do
    for pid in $(get_pids); do
        test -r "/proc/$pid/cmdline" || continue
        tr '\0' '\n' <"/proc/$pid/cmdline" | grep -Fx -- "$parent" >/dev/null || exit 3
    done
    (( $(date +%s) < deadline-180 ))
    sleep 5
done
export PYTHONPATH="$PWD:/gs/env/fixed-baseline-deps" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
# The parent must be sealed even if its process already disappeared.
/usr/bin/python3 -B -m experiments.t2t.serving_slots.llada_seat_report \
  "$parent" benchmarks/frozen90/manifest.json >/dev/null
remaining=$(( deadline-$(date +%s) ))
(( remaining>180 ))
test -z "$(get_pids)"
[[ $(nvidia-smi -i "$uuid" --query-gpu=memory.used --format=csv,noheader,nounits) -lt 100 ]]
export CUDA_VISIBLE_DEVICES="$uuid" TRITON_CACHE_DIR="/gs/runtime/llada-admission-1128/triton-$gpu"
exec timeout -k 3 "$remaining" /gs/env/venv-llada/bin/python -Bu -m experiments.t2t.serving_slots.llada_window_spare \
  run --q "$q" --factor "$factor" --parent "$parent" --out "$out" --deadline "$deadline" --samples /gs/code/samples.json \
  --reference /gs/work/llada-refs/llada21 --checkpoint /gs/models/LLaDA2.1-mini
