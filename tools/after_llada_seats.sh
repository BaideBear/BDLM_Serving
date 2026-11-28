#!/usr/bin/env bash
# Wait for one identified model process; the launcher checks receipts and GPU.
set -euo pipefail
pid=${1:?previous model PID}
root=${2:?previous output directory}
deadline=${3:?deadline}
shift 3
[[ "$pid" =~ ^[1-9][0-9]*$ && "$root" == /gs/results/* ]]
if [[ -r /proc/$pid/cmdline ]]; then
  tr '\0' '\n' </proc/"$pid"/cmdline | grep -Fx "$root" >/dev/null
  # Bound a possible PID reuse; tail --pid alone cannot distinguish that case.
  read -r stamp < <(awk '{print $22}' /proc/"$pid"/stat)
  while [[ -r /proc/$pid/stat ]]; do
    now=$(awk '{print $22}' /proc/"$pid"/stat 2>/dev/null) || break
    [[ "$now" == "$stamp" ]] || break
    (( $(date +%s) < deadline-120 )) || exit 3
    sleep 5
  done
fi
exec bash "$(dirname "$0")/run_llada_small_geometry_a800.sh" "$1" "$deadline" "$2"
