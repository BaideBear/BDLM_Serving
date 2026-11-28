#!/usr/bin/env bash
# One-shot physical parent controls on two distinct extremes; no reservation.
set -euo pipefail
: "${A800_MUX:?active authenticated A800 SSH socket}"
q=${1:?32 or 512}
mode=${2:-pilot}
case "$mode" in pilot) extra= ;; remaining) extra=--remaining ;; *) exit 2 ;; esac
case "$q" in
    32) id=0; gpu=GPU-d3c3fc9c-b15d-9776-9094-7d3b85b705c1 ;;
    512) id=2; gpu=GPU-1d5b27ea-c99d-15a4-af2b-fd82c414a0c5 ;;
    *) exit 2 ;;
esac
src=7f160c54fa1fcd005aa92f9420c700f5df4cb47f
base=/data1/project_data/gx-erqw-20261123/results/gemma-parent-first-projection-7f160c5
sink=$(ssh -o BatchMode=yes n164 "umask 077; mkdir -p '$base'; mktemp -d '$base/Q$q-a$id-XXXXXXXX'")
printf 'Sink: n164:%s\n' "$sink"
local_tmp=$(mktemp -d)
trap 'rm -rf -- "$local_tmp"' EXIT
git ls-tree -r --name-only "$src" -- bdlm_serving experiments tools |
    awk '/\.py$/ {print}' > "$local_tmp/list"
git archive "$src" benchmarks $(cat "$local_tmp/list") | tar -x -C "$local_tmp"
cp "$(dirname "$0")/gemma_parent_first_projection.py" "$local_tmp/probe.py"
rm "$local_tmp/list"
set +e
tar -czf - -C "$local_tmp" . |
ssh -S "$A800_MUX" -o BatchMode=yes -o ServerAliveInterval=15 A800-80g-2 "set -euo pipefail
    umask 077
    exec 9>>/dev/shm/gpu-a$id.lock
    flock -n 9
    for pass in 1 2; do
      test \"\$(nvidia-smi -i '$id' --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)\" = '$gpu, 0, 0'
      test -z \"\$(nvidia-smi -i '$id' --query-compute-apps=pid --format=csv,noheader,nounits)\"
      sleep 1
    done
    d=\$(mktemp -d /dev/shm/gs-gemma-parent-XXXXXX)
    trap 'rm -rf -- \"\$d\"' EXIT
    cd \"\$d\"; tar -xzf -
    mkdir cache
    export PARENT_GIT_SHA='$src' CUDA_VISIBLE_DEVICES='$gpu' PYTHONPATH=\"\$PWD:/gs/env/deps:/gs/env/fixed-baseline-deps\"
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=\"\$PWD/cache\"
    timeout -k 10 1200 /root/miniconda3/envs/rl/bin/python -Bu probe.py '$q' $extra" \
     2> >(ssh -o BatchMode=yes n164 "umask 077; cat > '$sink/stderr'") |
ssh -o BatchMode=yes n164 "umask 077; cat > '$sink/events.ndjson'"
ex=("${PIPESTATUS[@]}")
set -e
ssh -o BatchMode=yes n164 "umask 077; printf '%s\n' '${ex[*]}' > '$sink/pipeline.exit'"
[[ ${ex[*]} == "0 0 0" ]]
