#!/usr/bin/env bash
# Historical launcher for the registered LLaDA Q32 A0/A1/A2 campaign only.
# "prepare" creates a fresh sink once. Each numeric rank streams back to n164.
set -euo pipefail
: "${A800_MUX:?existing authenticated A800 SSH control socket}"
n=(ssh -o BatchMode=yes -o ServerAliveInterval=15 n164)
a=(ssh -S "$A800_MUX" -o BatchMode=yes -o ServerAliveInterval=15 A800-80g-2)
base=/data1/project_data/gx-erqw-20261123/results/llada21-live-fullblock-fixed90-v1
src=$base/Q8/attempt-20261128T163400Z-a1/bundle
dst=$base/Q32
pysrc=$(cd "$(dirname "$0")" && pwd)/fixed90_partition.py
uuids=(GPU-d3c3fc9c-b15d-9776-9094-7d3b85b705c1 GPU-861c9b5d-1e73-2550-f832-b73988bc64ff GPU-1d5b27ea-c99d-15a4-af2b-fd82c414a0c5)
locks=(gpu-a0.lock gpu-a1.tail-overlap-v1.lock gpu-a2.lock)
if [[ $1 == prepare ]]; then
  "${n[@]}" "set -euo pipefail
    umask 077
    cd '$src'
    printf '%s  %s\n' 30e6ff4f9e55291fab31a99c57838509095202d0afe25ff4d8554656038e84d4 bundle.sha256 | sha256sum -c --quiet
    sha256sum -c --quiet bundle.sha256
    mkdir '$dst'
    cd '$dst'
    cp -r '$src'/src '$src'/helpers '$src'/reference '$src'/samples.json '$src'/fixed90.py .
    mkdir cases part-0 part-1 part-2
    cat >fixed90_partition.py" < "$pysrc"
  "${n[@]}" "set -euo pipefail
    cd '$dst'
    umask 077
    cp ../registration.json campaign-registration.json
    python3 -c 'import json;print(json.dumps(dict(offer=32,cap=2048,mode=\"full_block\",parent_sha=\"8ed0487998ff9b1aa12230d54bc0743b4b4cd435423ed082255f2384e1fa9db7\",cutoff=1795915200,uuids=[\"${uuids[0]}\",\"${uuids[1]}\",\"${uuids[2]}\"],id=\"llada-full-admission-q32-three-shards\",indices=\"rank, rank+3, ... <90\",predecessor_claim=\"llada21-live-fullblock-fixed90-q32-20261129T0100\")))' > registration.json
    find src helpers reference -type f -print0 |
      LC_ALL=C sort -z | xargs -0 sha256sum > source.sha256
    sha256sum fixed90.py fixed90_partition.py samples.json registration.json campaign-registration.json >> source.sha256
    sha256sum source.sha256"
  exit
fi
rank=$1
[[ $rank =~ ^[012]$ ]]
uuid=${uuids[rank]}
lock=${locks[rank]}
set +e
{
  "${n[@]}" "cd '$dst' && tar -czf - src helpers reference samples.json fixed90.py fixed90_partition.py registration.json campaign-registration.json source.sha256" |
  "${a[@]}" "set -euo pipefail
    umask 077
    exec 8>/dev/shm/$lock
    flock -n 8
    w=\$(mktemp -d /dev/shm/frozen90-q32-a${rank}.XXXXXXXX)
    trap 'rm -rf -- \"\$w\"' EXIT
    trap 'exit 143' TERM HUP INT
    mkdir \"\$w/run\"
    cd \"\$w/run\"
    tar -xzf -
    sha256sum -c --quiet source.sha256
    cp campaign-registration.json ../registration.json
    for i in 1 2; do
      readout=\$(nvidia-smi -i '$uuid' --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
      pids=\$(nvidia-smi -i '$uuid' --query-compute-apps=pid --format=csv,noheader,nounits)
      [[ \$readout == '$uuid, 0, 0' && -z \$pids ]] || exit 91
      sleep 1
    done
    export CUDA_VISIBLE_DEVICES='$uuid' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH=\"\$PWD/src:/gs/env/fixed-baseline-deps\" OMP_NUM_THREADS=2
    export BDLM_LLADA_REFERENCE=\"\$PWD/reference\" PACKED_FULL_CACHE=0 TMPDIR=/tmp
    timeout -k 10 3600 /gs/env/venv-llada/bin/python -Bu fixed90_partition.py run . --rank '$rank'" |
  "${n[@]}" "cd '$dst' && python3 -Bu fixed90_partition.py receive . --rank '$rank'"
  status=("${PIPESTATUS[@]}")
} 2> >("${n[@]}" "umask 077; tee -a '$dst/part-$rank/launcher.stderr' >/dev/null")
set -e
"${n[@]}" "umask 077; set -C; printf '%s\n' '${status[*]}' > '$dst/part-$rank/pipeline.exit'"
[[ ${status[*]} == "0 0 0" ]]
