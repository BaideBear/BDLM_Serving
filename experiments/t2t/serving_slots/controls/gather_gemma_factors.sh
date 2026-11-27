#!/usr/bin/env bash
# Run the audited reader on the cluster. Only aggregates cross the network.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
kind=${1:?expected rank, close, profile-close or capacity}
case "$kind" in
    rank) name=prefix-rank-1128 ;;
    close) name=paid-close-1128 ;;
    profile-close) name=profile-close-1128 ;;
    capacity) name=capacity-head-1128 ;;
    density) name=phase-density-1128 ;;
    promotion) name=promotion-1128 ;;
    *) exit 2 ;;
esac
opts=(-o BatchMode=yes -o ConnectTimeout=8)
if [[ -n ${A800_MUX:-} ]]; then opts+=(-S "$A800_MUX"); fi
if [[ -n ${A800_PORT:-} ]]; then opts+=(-p "$A800_PORT"); fi
suite=experiments/t2t/serving_slots
draws=${BOOTSTRAP_DRAWS:-1000}
[[ $draws =~ ^[0-9]+$ ]] || exit 2
ssh "${opts[@]}" "${A800_HOST:-A800-80g-2}" \
    "cd /gs/runtime/$name/repo && /root/miniconda3/envs/rl/bin/python -B - \
     --manifest benchmarks/frozen90/manifest.json --schema $kind \
     --bootstrap $draws /gs/results/$name" < "$suite/cross_race_report.py" |
    "${PYTHON:-python3}" -B -m experiments.t2t.serving_slots.factor_tables "$kind" \
        "${2:-$suite/controls/gemma-factors/$kind}"
