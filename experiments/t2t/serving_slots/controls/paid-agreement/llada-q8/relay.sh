#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
attempt=${1:?fresh attempt}
stage=${2:?six or remaining}
[[ $attempt =~ ^[a-zA-Z0-9-]+$ && ( $stage == six || $stage == remaining ) ]]
r=/data1/project_data/gx-erqw-20261123
prior=$r/results/llada21-ibs-q8-tail-overlap-a1-v1
parent=$r/results/llada-ibs-q8-paid-agreement-a1-v1
d=$parent/$attempt
n=(ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=30 n164)
a=(ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=30 -S "${A800_MUX:?}" n1)
if [[ $stage == six ]]; then
    "${n[@]}" "umask 077; mkdir -p '$parent' && mkdir -m700 '$d' && mkdir '$d/frozen-q16'"
    COPYFILE_DISABLE=1 tar -cf - -C "$here" study.py a1.sh relay.sh |
        "${n[@]}" "umask 077; tar -xf - -C '$d'"
    COPYFILE_DISABLE=1 tar -cf - -C "$repo/tools" llada_ibs_eos_{stop,eval,q16_agreement,q16_eval,raw_audit}.py \
        -C "$repo/bdlm_serving/methods/t2t" paid_agreement.py |
        "${n[@]}" "umask 077; tar -xf - -C '$d/frozen-q16'"
    "${n[@]}" "umask 077; set -C; cd '$d' && python3 -B study.py prepare '$prior' >reference.json"
    wall=1800
else
    "${n[@]}" "cd '$d' && python3 -Bc 'import json; assert json.load(open(\"six-summary.json\"))[\"expand\"] is True' &&
        test \"\$(cat six.exit)\" = 0"
    wall=7100
fi
deadline=$(( $(date +%s) + wall ))
"${n[@]}" "umask 077; mkdir -m700 '$d/$stage'"
set +e
"${n[@]}" "cd '$d' && tar -cf - study.py a1.sh reference.json frozen-q16" |
    "${a[@]}" "umask 077; w=\$(mktemp -d /dev/shm/llada-q8-paid.XXXXXXXX) || exit 1
    trap 'rm -rf -- \"\$w\"' EXIT
    trap 'exit 143' HUP INT TERM
    cd \"\$w\" && tar -xf - && bash a1.sh '$stage' '$deadline'" \
    2> >("${n[@]}" "umask 077; set -C; cat >'$d/$stage/worker.stderr'") |
    "${n[@]}" "umask 077; set -C; cat >'$d/$stage/receipts.ndjson'"
pipe=("${PIPESTATUS[@]}")
set -e
"${n[@]}" "umask 077; set -C; printf '%s\n' '${pipe[1]}' >'$d/$stage.exit'"
printf 'attempt=%s stage=%s pipeline=(%s)\n' "$d" "$stage" "${pipe[*]}"
if (( pipe[0] || pipe[1] || pipe[2] )); then exit 1; fi
"${n[@]}" "umask 077; set -C; cd '$d' &&
    export PYTHONPATH='$r/runtime/formal90-7a38ffc/src:$r/runtime/fixed-baseline-deps' &&
    '$r/runtime/venv/bin/python' -B study.py report '$stage' '$stage/receipts.ndjson' \
      '$r/archive/a800-gs-20261128/snapshot/code/samples.json' '$prior' >'$stage-summary.json' &&
    cat '$stage-summary.json'"
