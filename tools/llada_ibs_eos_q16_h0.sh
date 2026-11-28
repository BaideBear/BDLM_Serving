#!/usr/bin/env bash
set -Eeuo pipefail
trap 'printf "guard failure rc=%s line=%s: %s\n" "$?" "$LINENO" "$BASH_COMMAND" >&2' ERR
umask 077
src=$(realpath "${1:?frozen overlay}")
out=${2:?unused H0 results path}
end=${3:?deadline epoch}
export BDLM_Q16_SOURCE=${4:?overlay manifest digest}
base=/data1/project_data/gx-erqw-20261123
rt=$base/runtime
py=$rt/venv/bin/python
hold=/data1/project_data/gx-megrez-20261122/gpu-holders/gpu-hold
lock=$rt/shared-formal-H0.lock
prior=$base/results/llada-ibs-certified-eos-h0-v1
if [[ $# == 5 ]]; then
    prior=$(realpath "$5")
    [[ $prior == "$base/results/llada-ibs-q16-paid-agreement-h0-"* && $prior != "$out" ]]
fi
uuid=GPU-ceadb3cd-e63f-1446-a2d5-a291144f8efb
export CUDA_VISIBLE_DEVICES=$uuid PYTHONPATH="$src:$rt/fixed-baseline-deps"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$src"
panel=$src/experiments/t2t/serving_slots/controls/llada-ibs-certified-eos/q16-agreement-registration.json
[[ $(sha256sum q16-source.sha256 | cut -d ' ' -f1) == "$BDLM_Q16_SOURCE" ]]
sha256sum --quiet -c q16-source.sha256
CUDA_VISIBLE_DEVICES= "$py" -Bu -m tools.llada_ibs_eos_q16_eval check --panel "$panel"
exec 9<>"$lock"
flock -n 9
[[ $(stat -Lc '%d:%i' "/proc/$$/fd/9") == "$(stat -Lc '%d:%i' "$lock")" ]]
[[ $(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader) == "$uuid" ]]
[[ -s $prior/returned.identity && -s $base/results/llada-ibs-certified-eos-h0-v1/complete.json ]]
(( $(date +%s) < end-240 ))

apps() { nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader,nounits | sort -n | xargs; }
ticks() {
    [[ $(stat -c %u "/proc/$1") == "$(id -u)" ]] || return 1
    sed -E 's/^.*\) //' "/proc/$1/stat" | cut -d ' ' -f20
}
holder() {
    "$hold" status 0 9>&- | "$py" -c '
import json, os, pathlib, sys
h, = json.load(sys.stdin)
i = h["instance"]
p = pathlib.Path("/proc") / str(h["pid"])
hb = h["heartbeat"]
assert h["running"] and h["phase"] == "running" and h["gpu"] == h["gpu_status"]["index"] == 0
assert i["uid"] == p.stat().st_uid == os.getuid() and i["pid"] == h["pid"] == hb["pid"]
assert i["boot_id"] == pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
assert h["gpu_status"]["uuid"] == i["gpu_uuid"] == hb["gpu_uuid"] == os.environ["CUDA_VISIBLE_DEVICES"]
assert int((p/"stat").read_text().split(") ")[1].split()[19]) == i["start_ticks"]
argv = (p/"cmdline").read_bytes().split(b"\0")
assert argv[argv.index(b"_worker"):][:2] == [b"_worker", b"0"] and i["token"].encode() in argv
print(i["pid"], i["start_ticks"], i["uid"], i["gpu_uuid"], i["boot_id"])' 9>&-
}
initial=$(holder)
pid=${initial%% *}
[[ $initial == "$(<"$prior/returned.identity")" && $(apps) == "$pid" ]]
read -r old _ <"$prior/worker.identity"
[[ ! -e /proc/$old ]]
mkdir -m 700 "$out"
printf 'lock=%s inode=%s\nprior=%s\nholder=%s\n' "$lock" "$(stat -Lc '%d:%i' "$lock")" "$prior" "$initial" >"$out/handoff"
worker= worker_ticks= watchdog= released=0
restore() {
    rc=$?
    trap - EXIT ERR
    trap '' HUP INT TERM
    set +e
    if [[ -n $worker && -n $worker_ticks && $(ticks "$worker" 2>/dev/null) == "$worker_ticks" ]]; then
        kill -TERM "$worker" 2>/dev/null
        for ((i=0;i<10;i++)); do
            [[ $(ticks "$worker" 2>/dev/null) == "$worker_ticks" ]] || break
            sleep 1
        done
        if [[ $(ticks "$worker" 2>/dev/null) == "$worker_ticks" ]]; then kill -KILL "$worker" 2>/dev/null; fi
        wait "$worker"
    fi
    if [[ -n $watchdog ]]; then wait "$watchdog"; fi
    if ((released)); then
        active=$(apps) || exit 92
        if [[ -z $active ]]; then "$hold" start 0 --timeout 30 9>&- >"$out/holder-start.log" 2>&1 || exit 92; fi
        back=$(holder) || exit 92
        [[ $(apps) == "${back%% *}" ]] || exit 92
        printf '%s\n' "${back%% *}" >"$out/returned"
        printf '%s\n' "$back" >"$out/returned.identity"
    fi
    exit "$rc"
}
trap restore EXIT
trap 'exit 130' INT TERM HUP
[[ $(holder) == "$initial" && $(apps) == "$pid" ]]
released=1
"$hold" stop 0 9>&- >"$out/holder-stop.log" 2>&1
for ((i=0;i<10;i++)); do
    live=$(apps)
    if [[ -z $live ]]; then break; fi
    [[ $live == "$pid" ]]
    sleep 1
done
if [[ -e /proc/$pid/stat ]]; then
    [[ $(ticks "$pid") == "$(cut -d ' ' -f2 <<<"$initial")" ]]
    [[ $(sed -E 's/^.*\) //' "/proc/$pid/stat" | cut -d ' ' -f1) == Z ]]
fi
[[ -z $(apps) ]]
"$py" -Bu -m tools.llada_ibs_eos_q16_eval run --panel "$panel" --out "$out" 9>&- >"$out/receipts.ndjson" 2>"$out/worker.log" &
worker=$!
worker_ticks=$(ticks "$worker")
printf '%s %s %s\n' "$worker" "$worker_ticks" "$(id -u)" >"$out/worker.identity"
(
    trap - EXIT ERR
    while [[ $(ticks "$worker" 2>/dev/null) == "$worker_ticks" ]]; do
        if (( $(date +%s) >= end-45 )); then
            printf '%s %s\n' "$worker" "$worker_ticks" >"$out/time-limit"
            if [[ $(ticks "$worker" 2>/dev/null) == "$worker_ticks" ]]; then kill -TERM "$worker"; fi
            break
        fi
        sleep 2
    done
) 9>&- &
watchdog=$!
wait "$worker"
worker=
wait "$watchdog"
watchdog=
[[ -s $out/complete-six.json && $(wc -l <"$out/receipts.ndjson") -eq 6 ]]
