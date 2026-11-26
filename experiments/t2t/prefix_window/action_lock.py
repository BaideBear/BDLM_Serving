"""Audit Gemma prefix action locking against both a physical repeat and zero.

Raw per-request files and intermediate scores stay under the private data root.
Invoke with --data /path/to/prefix90 --q {32,128} --out /path/to/aggregate.csv.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

from . import exposure
from .score_run import snapshot
from tools.prefix_cohorts import score_files


AUDIT_KEYS = {"prefix_actions", "prefix_canvas_sha256", "prefix_draft_sha256"}
OUTCOME_KEYS = {"edited", "revoked", "resolved", "reserved_strict_edit_positions", "hard_strict_edits"}
CONTROLS = {32: (8, "right", "gemma-q32"), 128: (32, "random", "gemma-grid-k1")}


def drop(obj, keys):
    return {k: v for k, v in obj.items() if k not in keys}


def check(ok, why):
    if not ok:
        raise ValueError(why)


def proposals(row, normal_policy, index, locked):
    for k in ("prefix_canvas_sha256", "prefix_draft_sha256"):
        check(re.fullmatch("[0-9a-f]{64}", row.get(k, "")), "Missing after-state digest")
    if row["stage"] != "selected":
        check("prefix_actions" not in row, "Lock on a non-selected forward")
        return [], []
    exposure.classify("gemma", 256, normal_policy, index, row)
    added = row["prefix_plan"]["added_rows"]
    info = row["prefix_actions"]
    strict, revoked = info["proposed_strict"], info["proposed_revoked"]
    check(info["locked"] is locked and added == row["reserved_extra_edit_positions"],
          "Wrong reserved rows or lock label")
    check(all(isinstance(v, list) and v == sorted(set(v)) for v in (strict, revoked))
          and not set(strict) & set(revoked) and set(strict + revoked) <= set(added),
          "Invalid proposal row sets")
    check(row["reserved_strict_edit_positions"] == ([] if locked else strict),
          "Wrong committed strict edits on added rows")
    if not locked:
        check(row["revoked"] >= len(revoked), "Additional revocations exceed all revocations")
    return strict, revoked


def audit_one(old, normal, lock, index):
    """Require full repeat, common history and a localized first suppression."""
    check(normal["policy"] + "_action_lock" == lock["policy"], "Wrong policy pair")
    check(drop(old, {"events"}) == drop(normal, {"events"})
          and old["events"] == [drop(e, AUDIT_KEYS) for e in normal["events"]],
          f"Historical normal repeat mismatch on {index}")
    proposal_sets = []
    counts = Counter()
    for name, request, locked in (("normal", normal, False), ("lock", lock, True)):
        values = [proposals(e, normal["policy"], index, locked) for e in request["events"]]
        proposal_sets.append(values)
        for key, slot in (("strict", 0), ("revoked", 1)):
            n = sum(len(pair[slot]) for pair in values)
            counts[name + "_" + key] = n
            counts[name + "_" + key + "_requests"] = int(n > 0)
    first = None
    for i, (a, b) in enumerate(zip(normal["events"], lock["events"])):
        x, y = (v[i] for v in proposal_sets)
        check(x == y, f"Proposal mismatch on common state {index}:{i}")
        if x == ([], []):
            check(drop(a, {"prefix_actions"}) == drop(b, {"prefix_actions"}),
                  f"Pre-suppression state diverged at {index}:{i}")
            continue
        strict, revoked = x
        first = i
        check(drop(a, AUDIT_KEYS | OUTCOME_KEYS) == drop(b, AUDIT_KEYS | OUTCOME_KEYS),
              f"First suppression also changed the plan or other actions at {index}:{i}")
        for k, delta in (("edited", len(strict)), ("hard_strict_edits", len(strict)),
                         ("revoked", len(revoked)), ("resolved", -len(revoked))):
            check(a[k] - b[k] == delta, f"First suppression: wrong {k} at {index}:{i}")
        check(a["prefix_canvas_sha256"] != b["prefix_canvas_sha256"],
              "Directly suppressed action did not change the canvas/acceptance digest")
        if not revoked:
            check(a["prefix_draft_sha256"] == b["prefix_draft_sha256"],
                  "Unchanged acceptance but draft changed on first strict edit")
        break
    if first is None:
        check(len(normal["events"]) == len(lock["events"])
              and drop(normal, {"events", "policy"}) == drop(lock, {"events", "policy"}),
              f"Unexposed request changed output, length or work: {index}")
    counts["requests_directly_suppressed"] = int(first is not None)
    return counts


def run(data, q, out):
    k, selector, zero_dir = CONTROLS[q]
    p = f"t2t_q{q}_f0_m{q-k}_hard_prefix_slack_{selector}"
    locked, zero = p + "_action_lock", f"t2t_q{q}_f0_m{q}_hard_prefix_zero"
    run_dir, cases = data / "gemma-prefix-action-lock", data / "samples.json"
    digest = exposure.sha256(cases)
    cohort = exposure.load_cases(cases)
    working = data / ".action-lock-scores"
    sets = [snapshot(d, policy, digest, working / ".inputs") for d, policy in
            ((data / f"gemma-ratio-q{q}-part2", p), (run_dir, p), (run_dir, locked),
             (data / zero_dir, zero))]
    counts = {}
    for i in range(90):
        records = [json.loads((directory / f"{i:02d}-{policy}.json").read_text())
                   for directory, policy in zip(sets[:3], (p, p, locked), strict=True)]
        check(all(all(r[field] == cohort[i][field] for field in ("index", "dataset", "seed", "task_id"))
                  for r in records), "Task identity mismatch")
        counts[i] = audit_one(*records, i)
    rows = []
    for label, baseline, b in (("normal", sets[1], p), ("zero", sets[3], zero)):
        scores, paired, _ = score_files(baseline, sets[2], cases, working / f"q{q}-{label}.csv",
                                        baseline_policy=b, arm_policy=locked)
        for score in scores:
            if score["scope"] != "all":
                continue
            n = score["window"]
            items = [r for r in paired if r["window"] == n]
            check(len(items) == n, "Incomplete paired score")
            events = Counter()
            for r in items:
                events.update(counts[r["index"]])
            for field in ("rescued", "lost"):
                events["unexposed_" + field] = sum(
                    int(r[field]) for r in items if not counts[r["index"]]["requests_directly_suppressed"])
            rows.append(dict(score, control=label, **dict(events)))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--q", required=True, choices=CONTROLS, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    run(args.data, args.q, args.out)
