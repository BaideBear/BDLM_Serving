"""Repeat / audit report for the Gemma Q32 conditional one-seat swap.

The report reads three measured frozen30 runs:

* ``--original``: the archived ``one-seat-v2-gemma`` run with the historical
  ``hard_once_base`` / ``hard_once_rank`` arms at the original seeds.
* ``--bridge``: the new original-seed direct run with ``hard_once_base`` and
  ``hard_once_mask``.  This is the bridge between the archived run and the
  repeat.
* ``--repeat``: the new repeat-seed direct run with ``fullwidth`` plus the same
  two one-seat arms, at every original seed plus ``--seed-offset``.

The report never reads ``marks-*.json``.  Every request is re-scored with the
actual scorer.  It verifies task and seed pairing, checks the conditional
splice against the archived rank/base arms, audits the paired pre-state and
victim admissions, and writes per-request, paired and summary tables.

This is a CPU-only post-hoc audit.  It does not run a model and does not use a
GPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

from bdlm_serving.evaluation.frozen import load_tasks
from bdlm_serving.evaluation.scoring import score
from bdlm_serving.methods.t2t.protocol import verify_bridge, verify_work


DEFAULT_SEED_OFFSET = 1_000_003
BASE = "base"
MASK = "mask"
RANK = "rank"
FULLWIDTH = "fullwidth"
ARMS = (FULLWIDTH, BASE, MASK)
PLAN_KEYS = (
    "base", "targeted", "random", "victim", "candidate", "random_candidate",
    "stratum", "fresh", "masks", "clock", "pre_sha256", "post_rng_sha256",
)
COMPARE_KEYS = (
    "output_ids", "output_tokens", "generation_forwards",
    "generation_physical_padded_rows",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_table(path, rows):
    """Write one CSV atomically, preserving the column order of the first row."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def add_check(checks, name, passed, detail=""):
    checks.append(dict(check=name, passed=int(bool(passed)), detail=detail))


def policy_for(meta, suffix):
    """Return the unique policy ending in ``_<suffix>``."""
    matches = [p for p in meta["policies"] if p.endswith("_" + suffix)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one policy ending in _{suffix} in {meta['policies']}")
    return matches[0]


def first_intervention(row):
    """Return ``(event_index, plan)`` for the first intervention, or ``(-1, None)``."""
    for index, event in enumerate(row.get("events", [])):
        if "intervention" in event:
            return index, event["intervention"]
    return -1, None


def plan_view(plan):
    return {key: plan.get(key) for key in PLAN_KEYS}


def compare_request(a, b):
    """Compare the measured output and physical work, not diagnostic events."""
    return all(a[key] == b[key] for key in COMPARE_KEYS)


def request_record(row, task, arm, seed_group, correct, anchor, plan):
    n = row["output_tokens"]
    f = row["generation_forwards"]
    q = row["generation_query_rows"]
    physical_q = row["generation_physical_padded_rows"]
    ids_sha = hashlib.sha256(json.dumps(row["output_ids"]).encode()).hexdigest()
    return dict(
        model=row["model"], seed_group=seed_group, arm=arm, policy=row["policy"],
        index=row["index"], dataset=row["dataset"], task_id=row["task_id"],
        seed=row["seed"], status=row["status"],
        N=n, F=f, Q=q, physical_Q=physical_q, max_q=row["max_q"],
        tpf=n / f if f else 0.0,
        token_efficiency=n / physical_q if physical_q else 0.0,
        q_per_f=q / f if f else 0.0,
        correct=int(correct), failed=int(row["status"] != "success"),
        truncated=int(bool(row["truncated"])),
        output_ids_sha256=ids_sha,
        anchor=anchor,
        victim_kind=plan.get("stratum", "") if plan else "",
        stratum=plan.get("stratum", "") if plan else "",
        victim=plan.get("victim", "") if plan else "",
        candidate=plan.get("candidate", "") if plan else "",
        executed=json.dumps(plan.get("executed", [])) if plan else "",
        pre_sha256=plan.get("pre_sha256", "") if plan else "",
        after_sha256=plan.get("after_sha256", "") if plan else "",
        victim_queried=plan.get("victim_queried", "") if plan else "",
        victim_unresolved_before=plan.get("victim_unresolved_before", "") if plan else "",
        victim_accepted=plan.get("victim_accepted", "") if plan else "",
    )


def score_rows(rows, tasks):
    """Score with the real scorer; failed requests remain in the denominator."""
    return [
        int(row["status"] == "success" and score(row["output_text"], task)["correct"])
        for row, task in zip(rows, tasks, strict=True)
    ]


def summarize(records, seed_group, arm):
    rows = [r for r in records if r["seed_group"] == seed_group and r["arm"] == arm]
    n = sum(r["N"] for r in rows)
    f = sum(r["F"] for r in rows)
    q = sum(r["Q"] for r in rows)
    physical_q = sum(r["physical_Q"] for r in rows)
    return dict(
        seed_group=seed_group, arm=arm, requests=len(rows),
        correct=sum(r["correct"] for r in rows),
        failed=sum(r["failed"] for r in rows),
        truncated=sum(r["truncated"] for r in rows),
        N=n, F=f, Q=q, physical_Q=physical_q,
        tpf=mean(r["N"] / r["F"] for r in rows if r["F"]) if any(r["F"] for r in rows) else 0.0,
        token_efficiency=n / physical_q if physical_q else 0.0,
        q_per_f=q / f if f else 0.0,
        anchors=sum(r["anchor"] >= 0 for r in rows),
        matching_anchors=sum(r["anchor"] >= 0 and r["victim_kind"] == "mask" for r in rows),
        nonmatching_anchors=sum(r["anchor"] >= 0 and r["victim_kind"] != "mask" for r in rows),
        no_anchor=sum(r["anchor"] < 0 for r in rows),
        victim_queried=sum(int(r["victim_queried"]) for r in rows if r["victim_queried"] != ""),
        victim_unresolved_before=sum(
            int(r["victim_unresolved_before"]) for r in rows if r["victim_unresolved_before"] != ""),
        victim_accepted=sum(int(r["victim_accepted"]) for r in rows if r["victim_accepted"] != ""),
    )


def add_paired(summaries, records, seed_group):
    """Attach mask-vs-base paired wins/losses and the no-swap victim-admission count."""
    base = [r for r in records if r["seed_group"] == seed_group and r["arm"] == BASE]
    mask = [r for r in records if r["seed_group"] == seed_group and r["arm"] == MASK]
    wins = sum(int(m["correct"] and not b["correct"]) for b, m in zip(base, mask, strict=True))
    losses = sum(int(b["correct"] and not m["correct"]) for b, m in zip(base, mask, strict=True))
    matching = [r for r in base if r["anchor"] >= 0 and r["victim_kind"] == "mask"]
    accepts = sum(int(r["victim_accepted"]) for r in matching if r["victim_accepted"] != "")
    for row in summaries:
        if row["seed_group"] != seed_group:
            continue
        row["paired_wins"] = wins if row["arm"] == MASK else ""
        row["paired_losses"] = losses if row["arm"] == MASK else ""
        row["paired_delta_correct"] = (
            sum(m["correct"] for m in mask) - sum(b["correct"] for b in base)
            if row["arm"] == MASK else "")
        row["matching_victim_accepts_base"] = accepts


def paired_rows(records, seed_group):
    base = [r for r in records if r["seed_group"] == seed_group and r["arm"] == BASE]
    mask = [r for r in records if r["seed_group"] == seed_group and r["arm"] == MASK]
    rows = []
    for b, m in zip(base, mask, strict=True):
        nonmatching_identical = int(
            b["victim_kind"] != "mask"
            and b["output_ids_sha256"] == m["output_ids_sha256"]
            and all(b[k] == m[k] for k in ("N", "F", "Q", "physical_Q")))
        rows.append(dict(
            seed_group=seed_group, index=b["index"], dataset=b["dataset"],
            task_id=b["task_id"], seed=b["seed"], anchor=b["anchor"],
            victim_kind=b["victim_kind"], stratum=b["stratum"],
            base_correct=b["correct"], mask_correct=m["correct"],
            win=int(m["correct"] and not b["correct"]),
            loss=int(b["correct"] and not m["correct"]),
            delta_correct=m["correct"] - b["correct"],
            base_N=b["N"], mask_N=m["N"], delta_N=m["N"] - b["N"],
            base_F=b["F"], mask_F=m["F"], delta_F=m["F"] - b["F"],
            base_Q=b["Q"], mask_Q=m["Q"], delta_Q=m["Q"] - b["Q"],
            base_physical_Q=b["physical_Q"], mask_physical_Q=m["physical_Q"],
            delta_physical_Q=m["physical_Q"] - b["physical_Q"],
            base_victim_queried=b["victim_queried"],
            mask_victim_queried=m["victim_queried"],
            base_victim_unresolved_before=b["victim_unresolved_before"],
            mask_victim_unresolved_before=m["victim_unresolved_before"],
            base_victim_accepted=b["victim_accepted"],
            mask_victim_accepted=m["victim_accepted"],
            pre_sha256_equal=int(b["pre_sha256"] == m["pre_sha256"]),
            victim_equal=int(b["victim"] == m["victim"]),
            candidate_equal=int(b["candidate"] == m["candidate"]),
            nonmatching_identical=nonmatching_identical,
        ))
    return rows


def check_intervention_fields(checks, label, plan):
    """Validate the new per-row acceptance fields against the actual executed Q."""
    required = ("victim_queried", "victim_unresolved_before", "victim_accepted")
    missing = [key for key in required if key not in plan]
    if missing:
        add_check(checks, f"{label}/intervention_fields", False,
                  "missing " + ",".join(missing))
        return
    detail = []
    if plan["victim_queried"] != (plan["victim"] in plan["executed"]):
        detail.append("victim_queried differs from executed membership")
    if plan["victim_unresolved_before"] != (plan["stratum"] == "mask"):
        detail.append("victim_unresolved_before differs from stratum")
    if plan["victim_accepted"] and not plan["victim_unresolved_before"]:
        detail.append("accepted without being unresolved before")
    if plan["victim_accepted"] and not plan["victim_queried"]:
        detail.append("accepted without being queried")
    add_check(checks, f"{label}/intervention_fields", not detail, "; ".join(detail))


def anchor_audit_row(seed_group, index, task, base_row, mask_row, base_plan, mask_plan,
                     *, old_base_row=None, old_rank_row=None):
    base_anchor, _ = first_intervention(base_row)
    mask_anchor, _ = first_intervention(mask_row)
    kind = base_plan.get("stratum", "") if base_plan else ""
    old_base_match = old_rank_match = ""
    if old_base_row is not None:
        old_base_match = int(compare_request(mask_row, old_base_row))
    if old_rank_row is not None:
        old_rank_match = int(compare_request(mask_row, old_rank_row))
    return dict(
        seed_group=seed_group, index=index, dataset=task["dataset"],
        task_id=task["task_id"], seed=task["seed"],
        base_anchor=base_anchor, mask_anchor=mask_anchor,
        anchor_equal=int(base_anchor == mask_anchor),
        victim_kind=kind, stratum=kind,
        pre_sha256=base_plan.get("pre_sha256", "") if base_plan else "",
        mask_pre_sha256=mask_plan.get("pre_sha256", "") if mask_plan else "",
        pre_sha256_equal=int(bool(base_plan and mask_plan)
                             and base_plan.get("pre_sha256") == mask_plan.get("pre_sha256")),
        victim=base_plan.get("victim", "") if base_plan else "",
        mask_victim=mask_plan.get("victim", "") if mask_plan else "",
        victim_equal=int(bool(base_plan and mask_plan)
                         and base_plan.get("victim") == mask_plan.get("victim")),
        candidate=base_plan.get("candidate", "") if base_plan else "",
        mask_candidate=mask_plan.get("candidate", "") if mask_plan else "",
        candidate_equal=int(bool(base_plan and mask_plan)
                            and base_plan.get("candidate") == mask_plan.get("candidate")),
        base_executed=json.dumps(base_plan.get("executed", [])) if base_plan else "",
        mask_executed=json.dumps(mask_plan.get("executed", [])) if mask_plan else "",
        base_victim_queried=base_plan.get("victim_queried", "") if base_plan else "",
        mask_victim_queried=mask_plan.get("victim_queried", "") if mask_plan else "",
        base_victim_unresolved_before=(
            base_plan.get("victim_unresolved_before", "") if base_plan else ""),
        mask_victim_unresolved_before=(
            mask_plan.get("victim_unresolved_before", "") if mask_plan else ""),
        base_victim_accepted=base_plan.get("victim_accepted", "") if base_plan else "",
        mask_victim_accepted=mask_plan.get("victim_accepted", "") if mask_plan else "",
        base_after_sha256=base_plan.get("after_sha256", "") if base_plan else "",
        mask_after_sha256=mask_plan.get("after_sha256", "") if mask_plan else "",
        paired_plan_identical=int(bool(base_plan and mask_plan)
                                  and plan_view(base_plan) == plan_view(mask_plan)),
        mask_matches_old_base=old_base_match,
        mask_matches_old_rank=old_rank_match,
    )


def audit_original(original, bridge, tasks, checks):
    """Check the bridge against the archived base/rank splice."""
    rows = []
    old_base = original["rows"][BASE]
    old_rank = original["rows"][RANK]
    old_full = original["rows"].get(FULLWIDTH)
    new_base = bridge["rows"][BASE]
    new_mask = bridge["rows"][MASK]
    new_full = bridge["rows"].get(FULLWIDTH)
    for index, task in enumerate(tasks):
        label = f"original/{index:02d}"
        add_check(checks, f"{label}/base_matches_old_base",
                  compare_request(new_base[index], old_base[index]))
        if old_full is not None and new_full is not None:
            add_check(checks, f"{label}/fullwidth_matches_old_fullwidth",
                      compare_request(new_full[index], old_full[index]))
        base_anchor, base_plan = first_intervention(new_base[index])
        mask_anchor, mask_plan = first_intervention(new_mask[index])
        add_check(checks, f"{label}/anchor_index", base_anchor == mask_anchor)
        if base_plan is None:
            add_check(checks, f"{label}/no_anchor_mask_matches_old_base",
                      compare_request(new_mask[index], old_base[index]))
            identical = (compare_request(new_mask[index], new_base[index])
                         and new_mask[index].get("events") == new_base[index].get("events"))
            add_check(checks, f"{label}/no_anchor_identical", identical)
            rows.append(anchor_audit_row(
                "original", index, task, new_base[index], new_mask[index],
                base_plan, mask_plan, old_base_row=old_base[index],
                old_rank_row=old_rank[index]))
            continue
        add_check(checks, f"{label}/paired_plan",
                  plan_view(base_plan) == plan_view(mask_plan))
        check_intervention_fields(checks, f"{label}/base", base_plan)
        check_intervention_fields(checks, f"{label}/mask", mask_plan)
        kind = base_plan.get("stratum", "")
        if kind == "mask":
            add_check(checks, f"{label}/mask_anchor_matches_old_rank",
                      compare_request(new_mask[index], old_rank[index]))
            swapped = (set(mask_plan.get("executed", [])) == set(mask_plan.get("targeted", []))
                       and mask_plan.get("victim") not in mask_plan.get("executed", [])
                       and mask_plan.get("candidate") in mask_plan.get("executed", []))
            add_check(checks, f"{label}/mask_swap_executed", swapped)
        else:
            add_check(checks, f"{label}/slack_anchor_matches_old_base",
                      compare_request(new_mask[index], old_base[index]))
            identical = (compare_request(new_mask[index], new_base[index])
                         and new_mask[index].get("events") == new_base[index].get("events"))
            add_check(checks, f"{label}/slack_no_swap_identical", identical)
        rows.append(anchor_audit_row(
            "original", index, task, new_base[index], new_mask[index],
            base_plan, mask_plan, old_base_row=old_base[index],
            old_rank_row=old_rank[index]))
    return rows


def audit_repeat(repeat, tasks, checks, native=None):
    """Check the repeat's paired pre-state, conditional swap and no-swap identity."""
    rows = []
    base_rows = repeat["rows"][BASE]
    mask_rows = repeat["rows"][MASK]
    full_rows = repeat["rows"][FULLWIDTH]
    for index, task in enumerate(tasks):
        label = f"repeat/{index:02d}"
        if native is not None:
            try:
                verify_bridge(full_rows[index], native[index])
                add_check(checks, f"{label}/fullwidth_matches_native", True)
            except ValueError as error:
                add_check(checks, f"{label}/fullwidth_matches_native", False, str(error))
        base_anchor, base_plan = first_intervention(base_rows[index])
        mask_anchor, mask_plan = first_intervention(mask_rows[index])
        add_check(checks, f"{label}/anchor_index", base_anchor == mask_anchor)
        if base_plan is None:
            identical = (compare_request(mask_rows[index], base_rows[index])
                         and mask_rows[index].get("events") == base_rows[index].get("events"))
            add_check(checks, f"{label}/no_anchor_identical", identical)
            rows.append(anchor_audit_row(
                "repeat", index, task, base_rows[index], mask_rows[index],
                base_plan, mask_plan))
            continue
        add_check(checks, f"{label}/paired_plan",
                  plan_view(base_plan) == plan_view(mask_plan))
        check_intervention_fields(checks, f"{label}/base", base_plan)
        check_intervention_fields(checks, f"{label}/mask", mask_plan)
        kind = base_plan.get("stratum", "")
        if kind == "mask":
            swapped = (set(mask_plan.get("executed", [])) == set(mask_plan.get("targeted", []))
                       and mask_plan.get("victim") not in mask_plan.get("executed", [])
                       and mask_plan.get("candidate") in mask_plan.get("executed", []))
            add_check(checks, f"{label}/mask_swap_executed", swapped)
        else:
            identical = (compare_request(mask_rows[index], base_rows[index])
                         and mask_rows[index].get("events") == base_rows[index].get("events"))
            add_check(checks, f"{label}/slack_no_swap_identical", identical)
        rows.append(anchor_audit_row(
            "repeat", index, task, base_rows[index], mask_rows[index],
            base_plan, mask_plan))
    return rows


def load_native(folder, tasks):
    """Load optional ``native256`` reference rows for the repeat fullwidth bridge."""
    rows = []
    for task in tasks:
        row = read_json(Path(folder) / f"{task['index']:02d}-native256.json")
        if row["policy"] != "native256":
            raise ValueError(f"Wrong native policy on request {task['index']}")
        if any(row[key] != task[key] for key in ("index", "dataset", "task_id", "seed")):
            raise ValueError(f"Native reference identity mismatch on request {task['index']}")
        rows.append(row)
    return rows


def load_run(folder, suffixes, tasks, expected_offset, label, *, fullwidth=True):
    """Load one measured run and enforce task, policy and seed identity."""
    folder = Path(folder)
    meta = read_json(folder / "run.json")
    if meta.get("model") != "gemma":
        raise ValueError(f"{label}: expected gemma, got {meta.get('model')!r}")
    if meta.get("block") != 256:
        raise ValueError(f"{label}: expected B256, got {meta.get('block')!r}")
    if meta.get("seed_offset", 0) != expected_offset:
        raise ValueError(
            f"{label}: seed_offset {meta.get('seed_offset')!r} != {expected_offset}")
    arm_for_suffix = {"hard_once_base": BASE, "hard_once_mask": MASK,
                      "hard_once_rank": RANK}
    policies = {arm_for_suffix[suffix]: policy_for(meta, suffix) for suffix in suffixes}
    if fullwidth:
        policies[FULLWIDTH] = FULLWIDTH
    rows = {}
    for arm, policy in policies.items():
        if arm != FULLWIDTH and not policy.startswith("t2t_q32_f0_m32_"):
            raise ValueError(f"{label}/{arm}: expected a Gemma Q32 policy, got {policy!r}")
        loaded = []
        try:
            for task in tasks:
                row = read_json(folder / f"{task['index']:02d}-{policy}.json")
                if any(row[key] != task[key] for key in ("index", "dataset", "task_id", "seed")):
                    raise ValueError(f"{label}/{arm}: request identity mismatch on {task['index']}")
                if row["model"] != meta["model"] or row["policy"] != policy:
                    raise ValueError(f"{label}/{arm}: wrong model or policy on {task['index']}")
                if row["status"] == "success":
                    verify_work(row)
                loaded.append(row)
        except FileNotFoundError:
            if arm == FULLWIDTH and fullwidth != "require":
                continue
            raise
        rows[arm] = loaded
    return dict(meta=meta, rows=rows, folder=folder, label=label)


def plot_comparison(out, summaries):
    """Draw a small comparison if matplotlib is installed; never fail the report."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping comparison plot")
        return False
    groups = ("original", "repeat")
    arms = (BASE, MASK, FULLWIDTH)
    metrics = (("correct", "Correct / requests"), ("tpf", "Mean N/F"),
               ("token_efficiency", "Sum N / sum physical Q"))
    by_key = {(r["seed_group"], r["arm"]): r for r in summaries}
    figure, axes = plt.subplots(1, len(metrics), figsize=(12, 4))
    for axis, (key, title) in zip(axes, metrics):
        labels = [f"{g}\n{a}" for g in groups for a in arms if (g, a) in by_key]
        values = [by_key[(g, a)][key] for g in groups for a in arms if (g, a) in by_key]
        axis.bar(labels, values)
        axis.set_title(title)
        axis.tick_params(axis="x", labelsize=7)
    figure.tight_layout()
    path = Path(out) / "seat-repeat-comparison.svg"
    figure.savefig(path)
    plt.close(figure)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("samples", "bridge", "original", "repeat", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--seed-offset", type=int, default=DEFAULT_SEED_OFFSET,
                        help="predeclared additive seed offset (default 1000003)")
    parser.add_argument("--native-reference", type=Path,
                        help="optional native256 folder for the repeat fullwidth bridge")
    parser.add_argument("--plot", action="store_true", help="draw a small comparison SVG")
    args = parser.parse_args(argv)

    tasks_original = load_tasks(args.samples)
    tasks_repeat = load_tasks(args.samples, seed_offset=args.seed_offset)
    if len(tasks_original) != len(tasks_repeat):
        raise ValueError("Repeat task count differs from the original")
    for old, new in zip(tasks_original, tasks_repeat, strict=True):
        if any(old[key] != new[key] for key in ("index", "dataset", "task_id")):
            raise ValueError("Repeat task identity differs from the original")
        if new["seed"] != old["seed"] + args.seed_offset:
            raise ValueError("Repeat seed is not original seed plus the offset")

    original = load_run(args.original, ("hard_once_base", "hard_once_rank"),
                        tasks_original, 0, "original")
    bridge = load_run(args.bridge, ("hard_once_base", "hard_once_mask"),
                      tasks_original, 0, "bridge")
    repeat = load_run(args.repeat, ("hard_once_base", "hard_once_mask"),
                      tasks_repeat, args.seed_offset, "repeat", fullwidth="require")
    if len({(run["meta"]["model"], run["meta"]["block"]) for run in (original, bridge, repeat)}) != 1:
        raise ValueError("Runs do not share one model and block size")

    native_folder = args.native_reference
    if native_folder is None and (args.repeat.parent / "seat-repeat-native-v1").exists():
        native_folder = args.repeat.parent / "seat-repeat-native-v1"
    native = load_native(native_folder, tasks_repeat) if native_folder else None

    checks = []
    scored = {}
    for label, run, tasks in (("original", original, tasks_original),
                              ("bridge", bridge, tasks_original),
                              ("repeat", repeat, tasks_repeat)):
        for arm, rows in run["rows"].items():
            scored[(label, arm)] = score_rows(rows, tasks)

    records = []
    for arm in ARMS:
        if arm not in bridge["rows"]:
            continue
        for row, task, correct in zip(bridge["rows"][arm], tasks_original,
                                      scored[("bridge", arm)], strict=True):
            anchor, plan = first_intervention(row)
            records.append(request_record(row, task, arm, "original", correct, anchor, plan))
    for arm in ARMS:
        if arm not in repeat["rows"]:
            continue
        for row, task, correct in zip(repeat["rows"][arm], tasks_repeat,
                                      scored[("repeat", arm)], strict=True):
            anchor, plan = first_intervention(row)
            records.append(request_record(row, task, arm, "repeat", correct, anchor, plan))

    anchor_rows = []
    anchor_rows.extend(audit_original(original, bridge, tasks_original, checks))
    anchor_rows.extend(audit_repeat(repeat, tasks_repeat, checks, native=native))

    summaries = []
    for group in ("original", "repeat"):
        for arm in ARMS:
            if any(r["seed_group"] == group and r["arm"] == arm for r in records):
                summaries.append(summarize(records, group, arm))
        add_paired(summaries, records, group)
    paired = paired_rows(records, "original") + paired_rows(records, "repeat")

    args.out.mkdir(parents=True, exist_ok=True)
    write_table(args.out / "seat-repeat-requests.csv", records)
    write_table(args.out / "seat-repeat-summary.csv", summaries)
    write_table(args.out / "seat-repeat-paired.csv", paired)
    write_table(args.out / "seat-repeat-anchor-audit.csv", anchor_rows)
    write_table(args.out / "seat-repeat-checks.csv", checks)
    (args.out / "seat-repeat-summary.json").write_text(json.dumps(dict(
        seed_offset=args.seed_offset, runs=dict(
            original=str(args.original), bridge=str(args.bridge), repeat=str(args.repeat)),
        native_reference=str(native_folder) if native_folder else None,
        summaries=summaries), indent=2) + "\n")
    if args.plot:
        plot_comparison(args.out, summaries)

    failed = [c for c in checks if not c["passed"]]
    if failed:
        raise ValueError("Repeat audit checks failed: " + "; ".join(
            f"{c['check']}: {c['detail']}" for c in failed))
    print(f"wrote {args.out}")
    for row in summaries:
        if row["arm"] == MASK:
            print(f"{row['seed_group']:8s} {row['arm']:9s} correct={row['correct']:2d} "
                  f"w/l={row['paired_wins']}/{row['paired_losses']} "
                  f"F={row['F']:5d} Q={row['Q']:6d} TPF={row['tpf']:.4f} "
                  f"tokEff={row['token_efficiency']:.5f} "
                  f"victimAccepts={row['matching_victim_accepts_base']}")


if __name__ == "__main__":
    main()
