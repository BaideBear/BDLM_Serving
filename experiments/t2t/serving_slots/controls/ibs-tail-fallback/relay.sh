#!/usr/bin/env bash
# Run on the workstation; keep the old cohort and all raw new receipts on n164.
set -euo pipefail
cd -- "$(dirname -- "$0")"
base=$(pwd)
old=$(cd ../ibs-tail-overlap && pwd)
attempt=${1:?fresh attempt label}
[[ $attempt =~ ^[a-zA-Z0-9-]+$ ]]
r=/data1/project_data/gx-erqw-20261123
prior=$r/results/llada21-ibs-q8-tail-overlap-a1-v1
parent=$r/results/llada21-ibs-q8-tail-fallback-a1-v1
d=$parent/$attempt
n=(ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=30 n164)
a=(ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=30 -S "${A800_MUX:?existing authenticated A800 mux}" n1)
deadline=$(( $(date +%s) + 1800 ))
sha=$(shasum -a 256 study.py | cut -d' ' -f1)
"${n[@]}" "umask 077; mkdir -p '$parent' && mkdir -m700 '$d'"
tar -cf - -C "$base" study.py a1_guard.sh relay.sh -C "$old" runner_a1.py |
    "${n[@]}" "umask 077; tar -xf - -C '$d'"
"${n[@]}" "umask 077; set -C; cd '$d' &&
    python3 -B study.py prepare runner_a1.py '$prior' >reference.json &&
    printf 'deadline=%s\nstudy_sha256=%s\n' '$deadline' '$sha' >launch.txt"

# One physical forward per joined head/successor remains guaranteed by the
# pinned fused runner and the measured all-layer projection audit.
set +e
"${n[@]}" "cd '$d' && tar -cf - study.py a1_guard.sh reference.json" |
    "${a[@]}" "umask 077; w=\$(mktemp -d /dev/shm/ibs-tail-fallback.XXXXXXXX) || exit 1
    trap 'rm -rf -- \"\$w\"' EXIT
    trap 'exit 143' HUP INT TERM
    cd \"\$w\" && tar -xf - && bash a1_guard.sh '$deadline' '$sha'" \
    2> >("${n[@]}" "umask 077; set -C; cat >'$d/runner.stderr'") |
    "${n[@]}" "umask 077; set -C; cat >'$d/receipts.ndjson'"
pipe=("${PIPESTATUS[@]}")
set -e
"${n[@]}" "umask 077; set -C; printf '%s\n' '${pipe[1]}' >'$d/runner.exit'"
printf 'attempt=%s pipeline=(%s)\n' "$d" "${pipe[*]}"
[[ ${pipe[*]} == '0 0 0' ]]
"${n[@]}" "umask 077; set -C; cd '$d' &&
    PYTHONPATH='$r/runtime/formal90-7a38ffc/src:$r/runtime/fixed-baseline-deps'
    export PYTHONPATH
    '$r/runtime/venv/bin/python' -B study.py report '$prior' runner_a1.py receipts.ndjson \
        '$r/archive/a800-gs-20261128/snapshot/code/samples.json' >summary.json &&
    cat summary.json"
