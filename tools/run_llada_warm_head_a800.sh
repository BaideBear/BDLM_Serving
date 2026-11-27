#!/usr/bin/env bash
# Continue a qualified same-device, two-arm fixed Q64 experiment. No GPU holder.
set -euo pipefail
qual=${1:?qualification result directory}
out=${2:?new result directory}
deadline=${3:?deadline epoch}
repo=/gs/work/llada-hfull-sf
py=/gs/env/venv-llada/bin/python
test ! -e "$out"
[[ $deadline =~ ^[0-9]+$ && $(date +%s) -lt $deadline ]]
[[ -z $(nvidia-smi -i 5 --query-compute-apps=pid --format=csv,noheader,nounits) ]]
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 5 --query-gpu=uuid --format=csv,noheader)"
export PYTHONPATH="$repo:/gs/env/fixed-baseline-deps" OMP_NUM_THREADS=2
export TRITON_CACHE_DIR=/gs/runtime/llada-hfull-1128/triton-g5
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
indices=$("$py" - "$qual" "$CUDA_VISIBLE_DEVICES" <<'PY'
import json, sys
from pathlib import Path
import math_verify, torch, transformers
root = Path(sys.argv[1])
r = json.loads((root/"run.json").read_bytes())
assert (r["indices"],r["device_uuid"],r["q"],r["mode"],r["law"],r["coverage"],r["old_source"]) == (
    [0,10,20],sys.argv[2],64,"half","1,8","head","run")
assert r["samples_sha256"]=="bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed"
assert (torch.__version__,transformers.__version__)==(r["torch"],r["transformers"])
for i in r["indices"]:
    v=json.loads((root/"rows"/f"{i:02}.json").read_bytes())
    assert v["index"]==i and v["compare"]["passed"] and v["compare"]["pre_state_verified"]
    assert "old_failure" not in v and "new_failure" not in v
print(",".join(str(i) for i in range(90) if i not in r["indices"]))
PY
)
cd "$repo"
exec timeout -k 3 "$((deadline - $(date +%s)))" "$py" -B -u \
    /gs/work/launch_llada_hfull_single_factor.py \
    --checkpoint /gs/models/LLaDA2.1-mini --reference /gs/work/llada-refs/llada21 \
    --samples /gs/code/samples.json --out "$out" --deadline "$deadline" --indices "$indices"
