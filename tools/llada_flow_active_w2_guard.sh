#!/usr/bin/env bash
# n164 only. Keep the inherited H4 lock until the replacement holder is verified.
set -euo pipefail
umask 077
R=/data1/project_data/gx-erqw-20261123
D=$R/results/llada-flow-active-w2-h4-v1
T=$R/runtime
P=$T/venv/bin/python
H=/data1/project_data/gx-megrez-20261122/gpu-holders
U=GPU-d3bdc2bc-d19d-e98f-ac79-ab9c6e887e0e
exec 9>>"$T/gpu4-experiment.lock"
flock -n 9
cd "$D"
sha256sum -c source.sha256 >&2
test ! -e compact.ndjson
test ! -e worker.pid
cd "$T/formal90-7a38ffc/src"
export CUDA_VISIBLE_DEVICES=$U PYTHONPATH=$PWD:$T/fixed-baseline-deps
export BDLM_LLADA_REFERENCE=$T/t2t-native-1125/frozen90-native/llada21
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PACKED_FULL_CACHE=0
apps(){ nvidia-smi -i "$U" --query-compute-apps=pid --format=csv,noheader,nounits; }
holder(){
    "$P" -B - "$H" "$U" "$(apps)" "$1" <<'PY'
import json, os, sys
from pathlib import Path
h, uuid, gpu, phase = sys.argv[1:]
h = Path(h)
s, hb = [json.loads((h / "state" / f).read_bytes()) for f in ("gpu-4.json","gpu-4-heartbeat.json")]
p = Path("/proc") / str(s["pid"])
argv = (p/"cmdline").read_bytes().split(b"\0")
assert s["uid"] == p.stat().st_uid == os.getuid() and s["gpu"] == 4 and s["gpu_uuid"] == uuid
assert s["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
assert s["start_ticks"] == int((p/"stat").read_text().rsplit(")",1)[1].split()[19])
assert s["pid"] == hb["pid"] and hb["phase"] == "running" and str(s["pid"]) == gpu
assert os.fsencode(h/"gpu_hold.py") in argv and argv[argv.index(b"_worker"):][:2] == [b"_worker",b"4"] and s["token"].encode() in argv
print(json.dumps(dict(phase=phase,pid=s["pid"],start_ticks=s["start_ticks"],uid=s["uid"],uuid=uuid)),flush=True)
PY
}
test "$(nvidia-smi -i 4 --query-gpu=uuid --format=csv,noheader)" = "$U"
for m in gemma llada21; do
    test -f "$R/results/successor18-785aead4/H4/$m.done"
    test -f "$R/results/mechanism18-c6275852/H4/$m.done"
done
if ps -eo args | grep -Eiq '[h]4_(mech_wait|succ_wait|chain|mask_order)|[r]un_h4_|[w]ait_(gpu4|h4)|[h]4-mask-order'; then exit 16; fi
seconds=$((1795915200 - $(date +%s) - 90))
(( seconds >= 180 ))
(( seconds <= 14400 )) || seconds=14400
holder before
restore(){
    rc=$?
    trap - EXIT
    trap '' HUP INT TERM
    if [[ -n "${sup:-}" ]]; then
        kill -TERM "$sup" 2>/dev/null || true
        wait "$sup" 2>/dev/null || true
    fi
    if [[ -z "$(apps)" ]]; then "$H/gpu-hold" start 4 >/dev/null; fi
    holder returned
    exit "$rc"
}
trap restore EXIT
trap 'exit 143' HUP INT TERM
"$H/gpu-hold" stop 4 >/dev/null
test -z "$(apps)"
set -C
timeout -k 8 "$seconds" bash -c 'echo "$$" > "$1/worker.pid"; exec "$2" -Bu "$1/llada_flow_active_w2_study.py" run' \
    _ "$D" "$P" >"$D/compact.ndjson" 2>"$D/stderr" &
sup=$!
wait "$sup"
sup=
