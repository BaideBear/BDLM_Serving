#!/usr/bin/env bash
# Use the shared H0 lock and the established stage/ready/go loader.
set -Eeuo pipefail
trap 'printf "guard failure rc=%s line=%s: %s\n" "$?" "$LINENO" "$BASH_COMMAND" >&2' ERR
umask 077
src=$(realpath "${1:?registered source tree}")
out=${2:?new H0 result directory}
export BDLM_WINDOW_END=${3:?absolute deadline epoch}
export BDLM_SOURCE_TREE=${4:?source snapshot digest}
export DG_SIGNAL=$out CUDA_VISIBLE_DEVICES=GPU-ceadb3cd-e63f-1446-a2d5-a291144f8efb
base=/data1/project_data/gx-erqw-20261123
rt=$base/runtime
py=$rt/venv/bin/python
hold=/data1/project_data/gx-megrez-20261122/gpu-holders/gpu-hold
prior=$base/results/gemma-ibs-q32-frozen90-f6913c2
lock=$rt/shared-formal-H0.lock
export PYTHONPATH="$src:$rt/diffusiongemma-native-20261125/deps:$rt/fixed-baseline-deps"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
cd "$src"
test "$(sha256sum source.sha256 | cut -d ' ' -f1)" = "$BDLM_SOURCE_TREE"
sha256sum --quiet -c source.sha256
exec 9<>"$lock"
flock -n 9
test "$(stat -Lc '%d:%i' "/proc/$$/fd/9")" = "$(stat -Lc '%d:%i' "$lock")"
test "$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)" = "$CUDA_VISIBLE_DEVICES"
test "$(date +%s)" -lt "$((BDLM_WINDOW_END-240))"
test -s "$prior/returned"
test "$(wc -l <"$prior/receipts.ndjson")" -eq 90
prior_worker=$(<"$prior/worker.pid")
test ! -e "/proc/$prior_worker"

pids() {
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits |
        sort -n | xargs
}
identity() {
    local pid=$1
    test "$(stat -c '%u' "/proc/$pid")" = "$(id -u)" || return 1
    sed -E 's/^.*\) //' "/proc/$pid/stat" | cut -d ' ' -f20
}
holder() {
    "$hold" status 0 9>&- | "$py" -c '
import json, os, pathlib, sys
h, = json.load(sys.stdin)
i = h["instance"]
p = pathlib.Path("/proc") / str(h["pid"])
assert h["running"] and i["uid"] == p.stat().st_uid == os.getuid() and i["pid"] == h["pid"]
assert i["boot_id"] == pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
assert h["gpu_status"]["uuid"] == i["gpu_uuid"] == os.environ["CUDA_VISIBLE_DEVICES"]
assert int((p/"stat").read_text().split(") ")[1].split()[19]) == i["start_ticks"]
print(i["pid"], i["start_ticks"], i["uid"], i["gpu_uuid"], i["boot_id"])' 9>&-
}

initial=$(holder)
initial_pid=${initial%% *}
if [[ $# == 5 ]]; then
    previous=$(realpath "$5")
    [[ $previous == "$base/results/gemma-native-argmax-h0-"* && "$previous" != "$out" ]]
    test "$initial" = "$(<"$previous/returned.identity")"
    read -r old_worker _ <"$previous/worker.identity"
    test ! -e "/proc/$old_worker"
    test "$(head -n 1 "$previous/handoff")" = "lock=$lock inode=$(stat -Lc '%d:%i' "$lock")"
else
    test "$initial_pid" = "$(<"$prior/returned")"
fi
test "$(pids)" = "$initial_pid"
mkdir -m 700 "$out"
{
    printf 'lock=%s inode=%s\n' "$lock" "$(stat -Lc '%d:%i' "$lock")"
    printf 'pid ticks uid uuid boot_id\n%s\n' "$initial"
    printf 'prior=%s prior_worker=%s deadline=%s\n' "$prior" "$prior_worker" "$BDLM_WINDOW_END"
    printf 'previous_attempt=%s\n' "${previous:-none}"
} >"$out/handoff"
worker= worker_ticks= released=0
restore() {
    rc=$?
    trap - EXIT ERR
    trap '' INT TERM HUP
    set +e
    if [[ -n $worker && -n $worker_ticks ]] && [[ $(identity "$worker" 2>/dev/null) == "$worker_ticks" ]]; then
        kill -TERM "$worker" 2>/dev/null
        for ((i=0;i<10;i++)); do
            [[ $(identity "$worker" 2>/dev/null) == "$worker_ticks" ]] || break
            sleep 1
        done
        if [[ $(identity "$worker" 2>/dev/null) == "$worker_ticks" ]]; then kill -KILL "$worker" 2>/dev/null; fi
        wait "$worker"
    fi
    if ((released)); then
        current=$(pids) || exit 92
        if [[ -z $current ]]; then
            "$hold" start 0 --timeout 30 9>&- >"$out/holder-start.log" 2>&1 || exit 92
        fi
        returned=$(holder) || exit 92
        test "$(pids)" = "${returned%% *}" || exit 92
        printf '%s\n' "${returned%% *}" >"$out/returned"
        printf '%s\n' "$returned" >"$out/returned.identity"
    fi
    exit "$rc"
}
trap restore EXIT
trap 'exit 130' INT TERM HUP
"$py" -Bu -m experiments.t2t.serving_slots.gemma_native_argmax_eval --out "$out" 9>&- \
    >"$out/receipts.ndjson" 2>"$out/worker.log" &
worker=$!
worker_ticks=$(identity "$worker")
printf '%s %s %s\n' "$worker" "$worker_ticks" "$(id -u)" >"$out/worker.identity"
ready_end=$(( $(date +%s) + 300 ))
until [[ -e $out/ready ]]; do
    test "$(identity "$worker")" = "$worker_ticks"
    test "$(date +%s)" -lt "$ready_end"
    sleep 1
done
test "$(holder)" = "$initial"
test "$(identity "$worker")" = "$worker_ticks"
"$py" - "$out/registration.json" "$worker" "$worker_ticks" 9>&- <<'PY'
import json, os, pathlib, sys
r = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert (r["pid"], r["ticks"], r["uid"], r["gpu"]) == (
    int(sys.argv[2]), int(sys.argv[3]), os.getuid(), os.environ["CUDA_VISIBLE_DEVICES"])
PY
test "$(pids)" = "$(printf '%s\n%s\n' "$initial_pid" "$worker" | sort -n | xargs)"
released=1
"$hold" stop 0 9>&- >"$out/holder-stop.log" 2>&1
# Some stopped holders remain briefly as zombies. Neither an active previous
# identity nor an unexpected compute context may overlap actual evaluation.
for ((attempt=0; attempt<10; attempt++)); do
    live=$(pids)
    if [[ $live == "$worker" ]]; then break; fi
    [[ $live == "$(printf '%s\n%s\n' "$initial_pid" "$worker" | sort -n | xargs)" ]]
    sleep 1
done
if [[ -e /proc/$initial_pid/stat ]]; then
    test "$(identity "$initial_pid")" = "$(cut -d ' ' -f2 <<<"$initial")"
    test "$(sed -E 's/^.*\) //' "/proc/$initial_pid/stat" | cut -d ' ' -f1)" = Z
fi
test "$(pids)" = "$worker"
touch "$out/go"
wait "$worker"
worker=
test -s "$out/qualified-six.json" && test -s "$out/complete.json"
test "$(wc -l <"$out/receipts.ndjson")" -eq 90
