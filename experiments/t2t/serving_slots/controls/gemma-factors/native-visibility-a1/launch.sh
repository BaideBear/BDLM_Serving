#!/usr/bin/env bash
set -euo pipefail
umask 077
rt=/gs/runtime/gemma-native-visibility-a1-1128
ll=/gs/runtime/llada-switch-repeat64-1128
prev=/gs/results/llada-switch-repeat64-1128
cd "$rt"
# Qualifier itself checks all fixed replays and the measured switching events.
CUDA_VISIBLE_DEVICES= PYTHONPATH="$ll" /usr/bin/python3 -B -m experiments.t2t.serving_slots.strict_handle_report \
  llada "$prev" "$ll/benchmarks/frozen90/manifest.json" > "$rt/prior-qualification.json"
/usr/bin/python3 - "$prev" "$rt/prior-qualification.json" <<'PY'
import json,sys
from pathlib import Path
d=Path(sys.argv[1]); r=json.loads(Path(sys.argv[2]).read_text())
assert r["profile"]=="quarter_window_repeat64" and len(r["arms"])==7 and r["indices"]==[0,10,20]
paths=[d/f"{s}-{i:02}.json" for s in r["arms"] for i in r["indices"]]
assert len(paths)==21 and all(p.is_file() for p in paths)
assert all(not Path("/proc",str(json.loads(p.read_bytes())["run_pid"])).exists() for p in paths)
PY
export CUDA_VISIBLE_DEVICES=GPU-861c9b5d-1e73-2550-f832-b73988bc64ff
test ! -e /gs/results/gemma-native-visibility-a1-1128
test "$(nvidia-smi -i 1 --query-gpu=uuid,memory.used --format=csv,noheader,nounits)" = "$CUDA_VISIBLE_DEVICES, 0"
test -z "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits)"
export PYTHONPATH="$rt:/gs/runtime/eos-handle-1128/repo:/gs/runtime/cold-frontier-last-a1-1128:/gs/env/deps:/gs/env/fixed-baseline-deps"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export TRITON_CACHE_DIR="$rt/triton"
left=$((1795843443-$(date +%s)))
test "$left" -gt 0
exec flock -n "$rt/run.lock" timeout -s KILL "$left" /root/miniconda3/envs/rl/bin/python -Bu \
  -m experiments.t2t.serving_slots.gemma_native_visibility_eval
