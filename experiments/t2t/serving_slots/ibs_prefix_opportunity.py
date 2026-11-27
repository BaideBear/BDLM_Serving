"""CPU-only audit of strict-budget cached-prefix refresh opportunities.

The capped IBS90 arms in ``t2t-native-1125/prefix90`` execute every decoder
forward under a hard physical query cap (Q8 or Q32).  Some of those forwards
leave physical seats unused.  This module asks a narrow, offline question:

    At each decoder tick, could the unused seats have replayed an existing,
    previously seen, stable strict-B32 generated prefix position without
    exceeding the same per-forward cap?

The candidate pool is exactly

    [max(prompt, block*32), block*32 + cached_prefix_before)

intersected with positions seen and written in an earlier decoder forward,
minus positions that are dirty, edited, deferred for re-encoding, or selected
by the current tick.  Injection is limited to the remaining physical seats and
to ``floor(q/4)``.  Nothing here infers TPF or answer quality; those require a
new scored run.  This is only the offline evidence for whether a
same-budget refresh-only alternative has opportunities.

Only arms with a matching completion marker, all 90 frozen identities, and a
recognized physical-call ledger are analyzed.  Explicit
``optimus_progress_limit`` failures are retained and reported, never silently
dropped.  A missing marker or missing request files excludes the arm.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW = ROOT.parent / "t2t-native-1125" / "prefix90"
DEFAULT_OUT = ROOT / "experiments/t2t/serving_slots/controls" / (
    "llada21-ibs-prefix-slack-opportunity.csv")

IDENTITY = ("index", "dataset", "task_id", "seed")
COHORTS = (("original30", 0, 30), ("new60", 30, 90), ("full90", 0, 90))
SCOPES = (("all_terminal", ("success", "generation_failed")),
          ("success_only", ("success",)))
STAGES = ("all", "bootstrap", "denoise", "stall", "terminal")
PREFIXES = ("all", "nonempty", "empty")
TERMINAL = frozenset(("success", "generation_failed"))

# Candidate arms named by the request.  Q16 is included so its incompleteness
# is reported explicitly rather than being mistaken for a missing arm.
CANDIDATES = (
    {"q": 8, "folder": "llada21-optimus-hardcap-endpoints",
     "policy": "optcap_ibs_8"},
    {"q": 32, "folder": "llada21-optimus-hardcap-anchor",
     "policy": "optcap_ibs_32"},
    {"q": 16, "folder": "llada21-optimus-hardcap-endpoints",
     "policy": "optcap_ibs_16"},
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open("rb") as handle:
        return json.load(handle)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"),
                  allow_nan=False)
    temp.replace(path)


def load_manifest(raw: Path) -> Tuple[str, Dict[int, Dict[str, object]], dict]:
    path = raw / "samples.json"
    require(path.is_file(), "Missing frozen90 samples.json: %s" % path)
    digest = sha256_file(path)
    data = read_json(path)
    samples = data["samples"] if isinstance(data, dict) else data
    require(isinstance(samples, list) and len(samples) == 90,
            "Frozen manifest is not 90 samples")
    tasks: Dict[int, Dict[str, object]] = {}
    for expected, sample in enumerate(samples):
        require(isinstance(sample, dict) and sample.get("index") == expected,
                "Frozen manifest index mismatch at %d" % expected)
        tasks[expected] = {key: sample[key] for key in IDENTITY}
    require(sorted(tasks) == list(range(90)), "Frozen indices are not 0..89")
    return digest, tasks, data


def ownership(path: Path) -> Tuple[int, int]:
    require(path.exists(), "Missing path: %s" % path)
    require(not path.is_symlink(), "Refusing symlink: %s" % path)
    stat = path.lstat()
    return stat.st_uid, stat.st_gid


def verify_run(folder: Path, policy: str, manifest_sha: str) -> dict:
    path = folder / "run.json"
    require(path.is_file(), "Missing run.json in %s" % folder)
    run = read_json(path)
    require(run.get("model") == "llada21" and run.get("suite") == "frozen90",
            "Wrong run model or suite")
    require(run.get("samples_sha256") == manifest_sha,
            "run.json samples hash differs from frozen manifest")
    require(run.get("block") == 32 and run.get("batch_size") == 1,
            "run.json is not strict B32 batch 1")
    require(run.get("hard_q_cap") is True,
            "run.json does not assert a hard Q cap")
    require(run.get("cap_clock") == "all_decoder_forwards_request_local",
            "run.json cap clock is not decoder-local")
    require(run.get("cap_excludes") == "prompt_prefill_only",
            "run.json cap excludes more than prompt prefill")
    require(policy in (run.get("policies") or []),
            "run.json does not own policy %s" % policy)
    return run


def verify_completion_marker(folder: Path, policy: str) -> dict:
    path = folder / ("complete-%s.json" % policy)
    require(path.is_file(), "Missing completion marker: %s" % path)
    marker = read_json(path)
    require(marker == {"policy": policy, "requests": 90},
            "Completion marker differs from policy/90: %s" % marker)
    return marker


def validate_failure(row: Mapping[str, object]) -> None:
    """Match the registered frozen90 terminal-failure contract."""
    status = row.get("status")
    failure = row.get("request_failure")
    if status == "success":
        require(failure is None, "Successful request carries a failure record")
        return
    require(status == "generation_failed",
            "Unrecognized terminal status: %r" % (status,))
    require(isinstance(failure, dict), "Missing generation failure details")
    require(failure.get("type") == "optimus_progress_limit",
            "Unrecognized generation failure type")
    require(failure == row["extra"].get("request_failure"),
            "Failure record disagrees with extra.request_failure")
    require(failure.get("delivered_output_tokens") == 0,
            "Failed request delivered tokens")
    require(failure.get("steps_in_block") == failure.get("step_limit"),
            "Failure did not reach its step limit")
    events = row["events"]
    require(bool(events) and failure.get("step_limit")
            == 2 * events[-1]["active_width"] + 4,
            "Failure step limit disagrees with active width")
    require(0 <= failure.get("cached_prefix", -1) < 32,
            "Failure cached prefix is outside its block")
    require(row.get("output_tokens") == 0 and row.get("output_ids") == []
            and row.get("output_text") == "" and not row.get("eos")
            and not row.get("truncated"),
            "Failed request has a partial returned response")


def validate_request(row: Mapping[str, object], task: Mapping[str, object],
                     spec: Mapping[str, object]) -> dict:
    """Validate identity, frozen config, geometry and the physical call ledger."""
    policy, q = spec["policy"], spec["q"]
    require(row.get("model") == "llada21", "Wrong model")
    require(row.get("policy") == policy, "Wrong policy")
    for key in IDENTITY:
        require(row.get(key) == task[key],
                "%s mismatch at index %s" % (key, task["index"]))
    require(row.get("phase") == "formal"
            and row.get("generation_cap") == 2048,
            "Not the formal 2048-token frozen90 arm")

    extra = row.get("extra")
    require(isinstance(extra, dict), "Missing extra provenance")
    require(extra.get("mode") == "ibs", "Wrong mode")
    require(extra.get("fixed_chunk") == q,
            "fixed_chunk does not match arm q")
    require(extra.get("window_switches") == 0, "Window switched")
    if row.get("status") == "success":
        require(extra.get("physical_caps") == [q],
                "physical_caps does not match arm q")
    else:
        require(extra.get("physical_caps") in (None, [q]),
                "Failed arm record has a different physical cap")
    validate_failure(row)

    prompt = row.get("prompt_ids")
    require(isinstance(prompt, list)
            and all(type(token) is int for token in prompt),
            "Invalid prompt token ids")
    prefill_rows = len(prompt) // 32 * 32
    require(row.get("prefill_rows") == prefill_rows,
            "Prompt prefill ledger differs")

    events, calls = row.get("events"), row.get("calls")
    require(isinstance(events, list) and isinstance(calls, list)
            and bool(events) and len(events) == len(calls),
            "Events and physical calls are missing or misaligned")

    offset = 1 if events[0].get("phase") == "prefill" else 0
    if offset:
        prefill = events[0]
        require(prefill_rows > 0
                and prefill.get("q_tokens") == prefill_rows
                and prefill.get("physical_forwards") == 1
                and prefill.get("absolute_positions") == list(range(prefill_rows)),
                "Invalid prompt prefill event")
    else:
        require(prefill_rows == 0, "Missing prompt prefill event")

    gen = events[offset:]
    require(gen and row.get("generation_forwards") == len(gen),
            "Generation forward count differs")
    for event, call in zip(events, calls):
        q_rows = event.get("q_tokens")
        require(type(q_rows) is int and q_rows >= 1
                and event.get("physical_forwards") == 1
                and call.get("batch") == 1
                and call.get("rows") == q_rows
                and call.get("physical_rows") == q_rows,
                "Physical call disagrees with its event")

    dirty = set()
    previous_block = previous_step = previous_after = None
    widths: List[int] = []
    for event in gen:
        require(event.get("phase") == "optimus"
                and event.get("mode") == "ibs"
                and event.get("fixed_chunk") == q
                and event.get("active_width") == 32,
                "Generation event is not strict B32 IBS")
        require(event.get("physical_cap") == q,
                "Physical cap is not arm q")
        width = event.get("q_tokens")
        require(type(width) is int and 0 < width <= q,
                "Invalid physical query width")
        positions = event.get("absolute_positions")
        require(isinstance(positions, list) and len(positions) == width
                and positions == sorted(set(positions)),
                "Invalid absolute positions")
        block, step = event.get("block"), event.get("step")
        require(type(block) is int and type(step) is int and step >= 0,
                "Invalid block or step")
        if previous_block is None:
            require(block == prefill_rows // 32 and step == 0,
                    "First generation block is not the prompt block")
        elif step == 0:
            require(block == previous_block + 1,
                    "Block did not advance by one")
        else:
            require(block == previous_block and step == previous_step + 1,
                    "Step did not advance by one in its block")
        start = block * 32
        require(positions[0] >= start and positions[-1] < start + 32,
                "Positions escape their B32 block")
        require(event.get("cross_boundary_rows") == 0,
                "Cross-boundary rows present")

        selected_local = event.get("selected_local")
        require(isinstance(selected_local, list)
                and [start + value for value in selected_local] == positions,
                "selected_local disagrees with absolute positions")
        candidate_local = event.get("candidate_local")
        require(isinstance(candidate_local, list)
                and candidate_local == sorted(set(candidate_local))
                and all(0 <= value < 32 for value in candidate_local),
                "Invalid candidate_local")
        candidate = {start + value for value in candidate_local}
        picked = set(positions)
        require(picked <= candidate, "Picked positions escape the candidate set")
        if len(candidate) <= q:
            require(picked == candidate,
                    "Under-filled candidate set was not projected")

        before = event.get("cached_prefix_before")
        after = event.get("cached_prefix_after")
        require(type(before) is int and type(after) is int
                and 0 <= before <= after <= 32,
                "Invalid cached prefix bounds")
        if step == 0:
            require(before == 0, "Block start has a non-zero cached prefix")
        else:
            require(before == previous_after,
                    "cached_prefix_before does not continue the block")

        writes = set(event.get("written_absolute") or [])
        require(writes <= picked, "Writes escape the picked positions")
        require(set(event.get("dirty_before") or []) == dirty,
                "dirty_before is not the carried dirty set")
        accepted, edited = event.get("accepted"), event.get("edited")
        require(type(accepted) is int and type(edited) is int
                and accepted >= 0 and edited >= 0 and accepted + edited <= width,
                "Invalid accepted/edited count")
        edited_positions = set(event.get("edited_positions") or [])
        deferred = event.get("cache_deferred_t2t_positions") or []
        require(edited_positions <= picked
                and set(deferred) == edited_positions
                and len(deferred) == len(edited_positions),
                "Edited/deferred positions disagree with the picked rows")
        require(edited == len(deferred),
                "Edited count disagrees with deferred positions")

        dirty = (dirty - picked) | writes
        require(all(position >= start + after for position in dirty),
                "A cached prefix position still has an outstanding write")
        widths.append(width)
        previous_block, previous_step, previous_after = block, step, after

    require(row.get("generation_query_rows") == sum(widths)
            == row.get("generation_physical_padded_rows"),
            "Generation Q ledger does not reconstruct")
    require(row.get("max_q") == max(widths) == q,
            "Observed maximum Q is not the arm cap")
    require(Counter(str(width) for width in widths) == row.get("q_histogram"),
            "q_histogram does not match generation events")
    if "logical_iterations" in row:
        require(row["logical_iterations"] == len(gen),
                "logical_iterations differs from generation forwards")

    if row.get("status") == "success":
        output_ids = row.get("output_ids")
        require(isinstance(output_ids, list) and output_ids
                and type(row.get("output_tokens")) is int
                and 0 <= row["output_tokens"] <= len(output_ids)
                and isinstance(row.get("output_text"), str),
                "Successful request has no usable output")

    return dict(policy=policy, q=q, status=row["status"],
                prefill_q=prefill_rows, generation_forwards=len(gen),
                generation_q=sum(widths), calls=len(calls),
                commit=extra.get("commit"),
                source_hashes=dict(extra.get("source_hashes") or {}))


@dataclass(frozen=True)
class TickRecord:
    index: int
    status: str
    block: int
    step: int
    stage: str
    q_tokens: int
    physical_cap: int
    cached_prefix_before: int
    cached_prefix_after: int
    accepted: int
    pool: int
    added: int
    remaining: int
    quota: int


def tick_stage(step: int, accepted: int, cached_prefix_after: int) -> str:
    if step == 0:
        return "bootstrap"
    if cached_prefix_after >= 32:
        return "terminal"
    if accepted == 0:
        return "stall"
    return "denoise"


def candidate_pool(event: Mapping[str, object], prompt: int,
                   seen: set, written: set) -> List[int]:
    """Stable generated prefix seen/written before, excluding dirty/selected."""
    start = int(event["block"]) * 32
    end = start + int(event["cached_prefix_before"])
    low = max(prompt, start)
    dirty = (set(event.get("dirty_before") or [])
             | set(event.get("edited_positions") or [])
             | set(event.get("cache_deferred_t2t_positions") or []))
    selected = set(event["absolute_positions"])
    return [position for position in range(low, end)
            if position in seen and position in written
            and position not in dirty and position not in selected]


def tick_records(row: Mapping[str, object],
                 spec: Mapping[str, object]) -> List[TickRecord]:
    """Replay one validated request under the refresh-only candidate rule."""
    prompt = len(row["prompt_ids"])
    records: List[TickRecord] = []
    seen, written = set(), set()
    for event in row["events"]:
        if event.get("phase") != "optimus":
            continue
        cap = int(event["physical_cap"])
        width = int(event["q_tokens"])
        pool = candidate_pool(event, prompt, seen, written)
        remaining = max(0, cap - width)
        quota = cap // 4
        added = min(len(pool), remaining, quota)
        records.append(TickRecord(
            index=int(row["index"]), status=row["status"],
            block=int(event["block"]), step=int(event["step"]),
            stage=tick_stage(int(event["step"]), int(event["accepted"]),
                             int(event["cached_prefix_after"])),
            q_tokens=width, physical_cap=cap,
            cached_prefix_before=int(event["cached_prefix_before"]),
            cached_prefix_after=int(event["cached_prefix_after"]),
            accepted=int(event["accepted"]), pool=len(pool), added=added,
            remaining=remaining, quota=quota))
        seen.update(event["absolute_positions"])
        written.update(event.get("written_absolute") or [])
    return records


def inspect_arm(raw: Path, spec: Mapping[str, object], manifest_sha: str,
                tasks: Mapping[int, Mapping[str, object]]) -> dict:
    """Return a complete-arm audit or an explicit exclusion reason."""
    folder = raw / spec["folder"]
    policy, q = spec["policy"], spec["q"]
    info = dict(policy=policy, q=q, folder=str(folder))
    try:
        require(folder.is_dir(), "Missing arm folder")
        info["owner"] = ownership(folder)
        info["request_files_present"] = sum(
            (folder / ("%02d-%s.json" % (index, policy))).is_file()
            for index in range(90))
        verify_run(folder, policy, manifest_sha)
        marker = verify_completion_marker(folder, policy)
        info["marker"] = marker
        info["run_sha256"] = sha256_file(folder / "run.json")
        info["marker_sha256"] = sha256_file(
            folder / ("complete-%s.json" % policy))
        files = []
        for index in range(90):
            path = folder / ("%02d-%s.json" % (index, policy))
            require(path.is_file(), "Missing request file %s" % path.name)
            require(ownership(path) == info["owner"],
                    "Request file ownership differs from arm folder")
            files.append(path)
    except ValueError as error:
        present = info.get("request_files_present")
        reason = str(error)
        if present is not None:
            reason += " (%d/90 request files present)" % present
        info.update(complete=False, reason=reason)
        return info

    file_digests = [sha256_file(path) for path in files]
    info["request_files_sha256"] = hashlib.sha256(
        "".join(file_digests).encode("ascii")).hexdigest()
    rows = []
    failures = 0
    commits, hashes = set(), set()
    for index, path in enumerate(files):
        row = read_json(path)
        if row.get("status") == "generation_failed":
            failures += 1
        summary = validate_request(row, tasks[index], spec)
        rows.append(row)
        if summary["commit"] is not None:
            commits.add(summary["commit"])
        hashes.add(json.dumps(summary["source_hashes"], sort_keys=True))
    require(len(commits) <= 1 and len(hashes) == 1,
            "Arm does not share one frozen source identity")
    physical_calls = sum(len(row["calls"]) for row in rows)
    call_rows = sum(call["rows"] for row in rows for call in row["calls"])
    prefill_calls = sum(row["prefill_rows"] > 0 for row in rows)
    generation_forwards = sum(row["generation_forwards"] for row in rows)
    generation_q = sum(row["generation_physical_padded_rows"] for row in rows)
    prefill_q = sum(row["prefill_rows"] for row in rows)
    require(physical_calls == generation_forwards + prefill_calls,
            "Physical call count is not generation forwards plus prefill calls")
    require(call_rows == generation_q + prefill_q,
            "Physical call rows do not reconstruct prefill plus generation Q")
    info.update(complete=True, requests=len(rows), success=len(rows) - failures,
                generation_failed=failures, rows=rows,
                completion=("complete_success90" if failures == 0
                            else "complete_terminal_failed%d" % failures),
                commit=(next(iter(commits)) if commits else None),
                source_hashes=json.loads(next(iter(hashes))),
                physical_calls=physical_calls, call_rows=call_rows,
                prefill_calls=prefill_calls,
                generation_forwards=generation_forwards,
                generation_q=generation_q, prefill_q=prefill_q)
    return info


def aggregate(records: Sequence[TickRecord], request_status: Mapping[int, str],
              cohort_range: range, scope_statuses: Iterable[str],
              stage: str, prefix: str) -> dict:
    statuses = set(scope_statuses)
    base_requests = {index for index in cohort_range
                     if request_status.get(index) in statuses}
    excluded_failed = {index for index in cohort_range
                       if request_status.get(index) == "generation_failed"
                       and "generation_failed" not in statuses}
    subset = [record for record in records
              if record.index in base_requests
              and (stage == "all" or record.stage == stage)
              and (prefix == "all"
                   or (prefix == "nonempty" and record.pool > 0)
                   or (prefix == "empty" and record.pool == 0))]
    eligible = [record for record in subset if record.pool > 0]
    slots_added = sum(record.added for record in subset)
    pool_slots = sum(record.pool for record in subset)
    seat_slots = sum(min(record.pool, record.remaining) for record in subset)
    previous_q = sum(record.q_tokens for record in eligible)
    group_q = sum(record.q_tokens for record in subset)
    return dict(
        requests=len(base_requests),
        success_requests=sum(request_status.get(index) == "success"
                             for index in base_requests),
        generation_failed_requests=sum(
            request_status.get(index) == "generation_failed"
            for index in base_requests),
        excluded_generation_failed=len(excluded_failed),
        requests_with_pool=len({record.index for record in eligible}),
        requests_with_opportunity=len({record.index for record in subset
                                       if record.added > 0}),
        ticks=len(subset),
        eligible_ticks=len(eligible),
        added_ticks=sum(record.added > 0 for record in subset),
        pool_slots=pool_slots,
        max_pool=max((record.pool for record in subset), default=0),
        slots_available_within_seats=seat_slots,
        slots_added=slots_added,
        prev_physical_q=previous_q,
        prev_forwards=len(eligible),
        group_physical_q=group_q,
        group_forwards=len(subset),
        slots_added_pct_of_prev_q=(
            round(100.0 * slots_added / previous_q, 4) if previous_q else 0.0),
        slots_added_pct_of_group_q=(
            round(100.0 * slots_added / group_q, 4) if group_q else 0.0),
    )


FIELDS = (
    "arm", "q", "completion", "scope", "cohort", "stage", "prefix",
    "requests", "success_requests", "generation_failed_requests",
    "excluded_generation_failed", "requests_with_pool",
    "requests_with_opportunity", "ticks", "eligible_ticks", "added_ticks",
    "pool_slots", "max_pool", "slots_available_within_seats", "slots_added",
    "prev_physical_q", "prev_forwards", "group_physical_q", "group_forwards",
    "slots_added_pct_of_prev_q", "slots_added_pct_of_group_q",
)


def build_rows(records: Sequence[TickRecord],
               request_status: Mapping[int, str], spec: Mapping[str, object],
               completion: str) -> List[dict]:
    rows = []
    for scope_name, scope_statuses in SCOPES:
        for cohort_name, low, high in COHORTS:
            for stage in STAGES:
                for prefix in PREFIXES:
                    values = aggregate(records, request_status,
                                       range(low, high), scope_statuses,
                                       stage, prefix)
                    rows.append(dict(arm=spec["policy"], q=spec["q"],
                                     completion=completion,
                                     scope=scope_name, cohort=cohort_name,
                                     stage=stage, prefix=prefix, **values))
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import csv
    require(bool(rows), "No aggregate rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW,
                        help="prefix90 raw root")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit-json", type=Path, default=None,
                        help="optional sidecar with identities and arm audits")
    parser.add_argument("--only-q", type=int, action="append", default=None,
                        help="limit to these caps (default: Q8, Q32, Q16)")
    args = parser.parse_args(argv)

    manifest_sha, tasks, manifest = load_manifest(args.raw)
    selected = [spec for spec in CANDIDATES
                if args.only_q is None or spec["q"] in args.only_q]
    require(bool(selected), "No candidate arms selected")

    arms, excluded = [], []
    records_by_arm: Dict[str, List[TickRecord]] = {}
    for spec in selected:
        info = inspect_arm(args.raw, spec, manifest_sha, tasks)
        if not info["complete"]:
            excluded.append(info)
            print("EXCLUDED q%d %s: %s" %
                  (spec["q"], spec["policy"], info["reason"]))
            continue
        arms.append(info)
        records_by_arm[info["policy"]] = []
        for row in info["rows"]:
            records_by_arm[info["policy"]].extend(tick_records(row, spec))

    require(bool(arms), "No complete capped IBS90 arm")
    all_rows = []
    for info in arms:
        spec = {"policy": info["policy"], "q": info["q"]}
        status = {row["index"]: row["status"] for row in info["rows"]}
        all_rows.extend(build_rows(records_by_arm[info["policy"]], status, spec,
                                   info["completion"]))

    write_csv(args.out, all_rows)
    audit = dict(raw=str(args.raw), out=str(args.out),
                 manifest_sha256=manifest_sha,
                 manifest_selection_salt=(
                     manifest.get("selection_salt")
                     if isinstance(manifest, dict) else None),
                 arms=[{key: value for key, value in info.items()
                        if key != "rows"} for info in arms],
                 excluded=[{key: value for key, value in info.items()
                            if key != "rows"} for info in excluded],
                 rows=len(all_rows))
    if args.audit_json:
        atomic_json(args.audit_json, audit)
    print("AUDIT " + json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
