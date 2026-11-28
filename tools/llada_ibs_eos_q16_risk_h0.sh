#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
trap 'printf "guard failure rc=%s line=%s: %s\n" "$?" "$LINENO" "$BASH_COMMAND" >&2' ERR
base=/data1/project_data/gx-erqw-20261123
rt=$base/runtime
name=llada-ibs-q16-risk-cap-h0-six-v1
src=$rt/$name
out=$base/results/$name
prior=$base/results/llada-flow-successor-fresh-h0-six-v1
py=$rt/venv/bin/python
hold=/data1/project_data/gx-megrez-20261122/gpu-holders/gpu-hold
lock=$rt/shared-formal-H0.lock
uuid=GPU-ceadb3cd-e63f-1446-a2d5-a291144f8efb
end=${1:?epoch deadline}
export BDLM_Q16_RISK_SOURCE=${2:?risk manifest digest}
export PYTHONPATH="$src:$rt/fixed-baseline-deps" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$uuid
cd "$src"
panel=$src/experiments/t2t/serving_slots/controls/llada-ibs-certified-eos/q16-risk-registration.json
[[ $(sha256sum q16-risk-source.sha256 | cut -d ' ' -f1) == "$BDLM_Q16_RISK_SOURCE" ]]
sha256sum --quiet -c q16-risk-source.sha256
args=(-Bu -m tools.llada_ibs_eos_q16_risk)
CUDA_VISIBLE_DEVICES= "$py" "${args[@]}" check --panel "$panel" --exposure "$src/exposure.json" >/dev/null
exec 9<>"$lock"
flock -n 9
[[ $(stat -Lc '%d:%i' "/proc/$$/fd/9") == 66305:200303112 &&
   $(stat -Lc '%d:%i' "$lock") == 66305:200303112 &&
   $(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader) == "$uuid" ]]
(( $(date +%s) < end-300 ))

apps() { nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader,nounits | sort -n | xargs; }
ticks() {
    [[ $(stat -c %u "/proc/$1") == "$(id -u)" ]] || return 1
    sed -E 's/^.*\) //' "/proc/$1/stat" | cut -d ' ' -f20
}
holder() {
    "$hold" status 0 9>&- | "$py" -c '
import json, os, pathlib, sys
h, = json.load(sys.stdin)
i, hb = h["instance"], h["heartbeat"]
p = pathlib.Path("/proc") / str(h["pid"])
assert h["running"] and h["phase"] == "running" and h["gpu"] == h["gpu_status"]["index"] == 0
assert i["uid"] == p.stat().st_uid == os.getuid() and i["pid"] == h["pid"] == hb["pid"]
assert i["boot_id"] == pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
assert h["gpu_status"]["uuid"] == i["gpu_uuid"] == hb["gpu_uuid"] == os.environ["CUDA_VISIBLE_DEVICES"]
assert int((p/"stat").read_text().split(") ")[1].split()[19]) == i["start_ticks"]
argv = (p/"cmdline").read_bytes().split(b"\0")
assert argv[argv.index(b"_worker"):][:2] == [b"_worker", b"0"] and i["token"].encode() in argv
print(i["pid"], i["start_ticks"], i["uid"], i["gpu_uuid"], i["boot_id"])' 9>&-
}

old=$(holder)
pid=${old%% *}
read -r prevworker prevticks prevuid <"$prior/worker.identity"
[[ $prevuid == "$(id -u)" && $old == "$(<"$prior/returned.identity")" &&
   $(apps) == "$pid" && $(ticks "$prevworker" 2>/dev/null || true) != "$prevticks" ]]
mkdir -m 700 "$out"
printf 'lock=%s inode=%s\nprior=%s\nholder=%s\n' "$lock" "$(stat -Lc '%d:%i' "$lock")" "$prior" "$old" >"$out/handoff"
worker= worker_ticks= watchdog= released=0
same_worker() { [[ -n $worker && -n $worker_ticks && $(ticks "$worker" 2>/dev/null) == "$worker_ticks" ]]; }
stop_worker() {
    same_worker || return 0
    kill -TERM "$worker" || return 0
    for ((i=0;i<10;i++)); do
        same_worker || return 0
        sleep 1
    done
    if same_worker; then kill -KILL "$worker"; fi
}
restore() {
    rc=$?
    trap - EXIT ERR
    trap '' HUP INT TERM
    set +e
    if same_worker; then stop_worker; wait "$worker"; fi
    if [[ -n $watchdog ]]; then wait "$watchdog"; fi
    if ((released)); then
        active=$(apps) || exit 92
        if [[ -z $active ]]; then "$hold" start 0 --timeout 30 9>&- >"$out/holder-start.log" 2>&1 || exit 92; fi
        back=$(holder) || exit 92
        [[ $(apps) == "${back%% *}" && -z $(ticks "$worker" 2>/dev/null || true) ]] || exit 92
        printf '%s\n' "$back" >"$out/returned.identity"
    fi
    exit "$rc"
}
trap restore EXIT
trap 'exit 130' INT TERM HUP
[[ $(holder) == "$old" && $(apps) == "$pid" ]]
released=1
"$hold" stop 0 9>&- >"$out/holder-stop.log" 2>&1
for ((i=0;i<10;i++)); do
    active=$(apps)
    [[ -n $active ]] || break
    [[ $active == "$pid" ]]
    sleep 1
done
if [[ -e /proc/$pid/stat ]]; then
    [[ $(ticks "$pid") == "$(cut -d ' ' -f2 <<<"$old")" ]]
    [[ $(sed -E 's/^.*\) //' "/proc/$pid/stat" | cut -d ' ' -f1) == Z ]]
fi
[[ -z $(apps) ]]
"$py" "${args[@]}" run --panel "$panel" --exposure "$src/exposure.json" --out "$out" \
    9>&- >"$out/receipts.ndjson" 2>"$out/worker.log" &
worker=$!
worker_ticks=$(ticks "$worker")
printf '%s %s %s\n' "$worker" "$worker_ticks" "$(id -u)" >"$out/worker.identity"
(
    trap - EXIT ERR
    while same_worker; do
        if (( $(date +%s) >= end-60 )); then
            printf '%s %s\n' "$worker" "$worker_ticks" >"$out/time-limit"
            stop_worker
            break
        fi
        sleep 2
    done
) 9>&- &
watchdog=$!
wait "$worker"
wait "$watchdog"
watchdog=
[[ -s $out/complete-six.json && $(wc -l <"$out/receipts.ndjson") -eq 6 ]]
