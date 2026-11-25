"""CPU-only same-state diagnostic for the preregistered LLaDA B32 ranker."""

import argparse
from collections import Counter
import csv
from fractions import Fraction
import gzip
import json
from pathlib import Path

from bdlm_serving.evaluation.frozen import MANIFEST
from bdlm_serving.methods.t2t.frontier import select_hard
from bdlm_serving.methods.t2t.geometry import rank_old


WIDTH = 32
ARMS = ((4, "slack", 1), (8, "slack", 1), (16, "slack", 2),
        (16, "fixed", 2), (32, "dense", 0))
QS = tuple(sorted({q for q, _, _ in ARMS}))


def _pick(old, masks, fresh, cap, rank, slots, slack):
    take = min(slots, len(old), max(0, cap - len(fresh) - 1))
    if slack:
        take = min(take, max(0, cap - len(fresh) - len(masks)))
    if cap == len(fresh):
        return set(fresh), set()
    reserved = set(rank[:take])
    picked = fresh | {masks[0]} | reserved
    remaining = (i for i in (*masks, *old) if i not in picked)
    return picked | set(i for _, i in zip(range(cap - len(picked)), remaining)), reserved


def _decide(editable, resolved, fresh, write, queries, clock, strict, seed, full):
    old = [i for i in editable if resolved[i] and i not in fresh]
    masks = [i for i in editable if not resolved[i]]
    dense = set(range(WIDTH))
    baseline = {}
    decisions, all_selected = {}, {}
    for q, allocation, slots in ARMS:
        arm = q, allocation, slots
        selected = not full and q != WIDTH
        all_selected[arm] = selected
        if not selected:
            baseline[q] = dense
            decisions[arm] = dense, set(), set()
            continue
        query = queries[q]
        if q not in baseline:
            picked, _ = select_hard(editable, resolved, fresh, write, query,
                                    clock, q, 0, "hard_noedit", seed)
            baseline[q] = set(picked)
        cap = len(baseline[q])
        slack = allocation == "slack"
        ranked = rank_old(old, editable, resolved, fresh, write, query, clock, strict)
        picked, reserved = _pick(old, masks, fresh, cap, ranked, slots, slack)
        frontier_picked, frontier_reserved = select_hard(
            editable, resolved, fresh, write, query, clock, q, slots,
            "hard_slack" if slack else "hard_frontier", seed)
        mask_set = set(masks)
        if (len(reserved) != len(frontier_reserved) or len(picked) != cap
                or not fresh <= picked or picked & mask_set != set(frontier_picked) & mask_set
                or (slack and picked & mask_set != baseline[q] & mask_set)):
            raise ValueError(f"Allocation mismatch for {arm} at tick {clock}")
        decisions[arm] = picked, reserved, set(frontier_reserved)
    return set(old), set(masks), baseline, decisions, all_selected


def replay(raw, task, counts):
    path = raw / f"{task['index']:02d}.trace.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = map(json.loads, f)
        meta, header = next(rows), next(rows)
        if meta.get("schema") != 1 or any(
                meta[k] != task[k] for k in ("index", "task_id", "dataset", "seed")):
            raise ValueError(f"Trace identity mismatch: {path}")
        plen, mask_id = header["prompt_length"], header["mask_id"]
        block = -WIDTH
        for row in rows:
            start, clock, before = row["block"], row["clock"], row["before"]
            first = start != block
            if first:
                if start <= block or start % WIDTH or clock != 1:
                    raise ValueError(f"Invalid block boundary: {path}, {start}, {clock}")
                block = start
                editable = tuple(range(max(0, plen - start), WIDTH))
                resolved = [token != mask_id for token in before]
                write = [-1] * WIDTH
                queries = {q: [-1] * WIDTH for q in QS}
                fresh, strict = set(), set()
            elif before != prior_after or clock != prior_clock + 1:
                raise ValueError(f"Broken native trajectory: {path}, {start}, {clock}")

            pool, masks, baseline, decisions, selected = _decide(
                editable, resolved, fresh, write, queries, clock, strict, task["seed"],
                full=first or all(resolved))

            # Only this point and later may use the current outcomes.
            after, accepted = row["after"], row["accepted"]
            if (len(after) != WIDTH or len(before) != WIDTH
                    or accepted != [token != mask_id for token in after]):
                raise ValueError(f"Invalid LLaDA trace frame: {path}, {start}, {clock}")
            changed = {i for i in editable if before[i] != after[i]}
            strict = {i for i in changed if resolved[i] and accepted[i]}
            old_targets = strict & pool
            for arm, (picked, reserved, frontier_reserved) in decisions.items():
                q, _, _ = arm
                zero = baseline[q]
                lazy = selected[arm]
                counts[arm].update({
                    "selected_steps": int(lazy),
                    "dense_steps": int(not lazy),
                    "strict_targets_all": len(strict),
                    "old_targets_all": len(old_targets),
                    "old_targets_selected": len(old_targets) if lazy else 0,
                    "reserved_rows": len(reserved),
                    "geometry_reserved_hits": len(old_targets & reserved),
                    "frontier_reserved_hits": len(old_targets & frontier_reserved),
                    "uniform_expected_reserved_hits": (
                        Fraction(len(reserved) * len(old_targets), len(pool)) if pool else 0),
                    "geometry_old_hits_selected": len(old_targets & picked) if lazy else 0,
                    "zero_old_hits_selected": len(old_targets & zero) if lazy else 0,
                    "geometry_old_hits_all": len(old_targets & picked),
                    "geometry_total_hits_all": len(strict & picked),
                    "different_query_set_steps": int(picked != zero),
                    "mask_rows_lost_vs_zero": len((zero - picked) & masks),
                })
            for q, picked in baseline.items():
                for i in picked:
                    queries[q][i] = clock
            for i in changed:
                write[i] = clock
            fresh = {i for i in changed if accepted[i]}
            resolved, prior_after, prior_clock = accepted, after, clock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True, type=Path, help="Verified llada21 trace directory")
    p.add_argument("--out", required=True, type=Path, help="New summary CSV path")
    args = p.parse_args()
    tasks = json.loads(MANIFEST.read_text())["samples"]
    verified = json.loads((args.raw / "verified.json").read_text())
    if len(tasks) != 30 or verified != {"model": "llada21", "matched_requests": 30}:
        raise ValueError("Expected the verified LLaDA frozen30 native cohort")
    counts = {arm: Counter() for arm in ARMS}
    for task in tasks:
        replay(args.raw, task, counts)
    results = []
    for (q, allocation, slots), c in counts.items():
        if (c["strict_targets_all"], c["old_targets_all"]) != (754, 311):
            raise ValueError("Native edit totals do not match the existing trace audit")
        c["uniform_expected_reserved_hits"] = f"{float(c['uniform_expected_reserved_hits']):.6f}"
        results.append(dict(q=q, allocation=allocation, edit_cap=slots, **c))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
