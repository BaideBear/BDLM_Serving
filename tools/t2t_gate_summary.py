"""Summarize completed frozen30 LLaDA gate runs as aggregate JSON on stdout.

Usage: python3 tools/t2t_gate_summary.py --run RUN_DIR [--run OTHER_DIR]
Add --only-complete to skip arms whose completion marker is not yet present.
No model dependencies, scoring, writes, or native/teacher-forced trace inputs.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import NamedTuple


MODEL, WIDTH, CASES = "llada21", 32, 30
GATES = ("near_miss", "gate_random")
CATEGORIES = ("prompt", "mask", "fresh", "near_miss", "other_resolved")
STAGE_PHASE = dict(prefill="prefill", bootstrap="denoise", selected="denoise",
                   verify="denoise", commit="commit")


class Trajectory(NamedTuple):
    output_ids: tuple
    denoise_checks: list
    strict_edits: list
    reservations: list


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def count(value, name):
    require(type(value) is int and value >= 0, f"Invalid count: {name}")
    return value


def positions(value, name):
    require(isinstance(value, list), f"Expected position list: {name}")
    result = {count(i, name) for i in value}
    require(len(result) == len(value), f"Duplicate position: {name}")
    return result


def read_json(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        require(isinstance(data, dict), "Expected a JSON object")
        return data
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def gated_arms(folders, only_complete):
    seen = set()
    for folder in folders:
        path = folder / "run.json"
        meta = read_json(path)
        require((meta.get("model"), meta.get("block"), meta.get("batch_size"))
                == (MODEL, WIDTH, 1), f"{path}: expected batch-one LLaDA B32")
        arms, names = meta["arms"], meta["policies"]
        require(isinstance(arms, list) and isinstance(names, list)
                and len(arms) == len(names), f"{path}: invalid arms/policies")
        qualified = False
        for arm, policy in zip(arms, names):
            mode = arm["selection"]
            if mode not in GATES:
                continue
            q, fresh, mask = (count(arm[k], k) for k in ("budget", "fresh", "mask"))
            split = (q, fresh, mask)
            require(fresh + mask < q < WIDTH
                    and policy == f"t2t_q{q}_f{fresh}_m{mask}_{mode}",
                    f"{path}: invalid gate arm or policy name")
            marker = folder / f"complete-{policy}.json"
            try:
                complete = read_json(marker)
            except FileNotFoundError:
                if only_complete:
                    continue
                raise
            require(complete == dict(policy=policy, requests=CASES),
                    f"{marker}: invalid completion marker")
            if not qualified:
                qualification = folder / "qualified.json"
                require(read_json(qualification) == dict(model=MODEL, full_width=WIDTH, matched=CASES),
                        f"{qualification}: expected full-width qualification for all 30 cases")
                qualified = True
            require(policy not in seen, f"Duplicate completed policy: {policy}")
            seen.add(policy)
            yield folder, policy, mode, split


def summarize_case(row, index, policy, mode, split):
    require(type(row["index"]) is int and row["index"] == index
            and (row["model"], row["policy"], row["status"]) == (MODEL, policy, "success"),
            "Wrong case, model, or policy, or unsuccessful request")
    require(all(isinstance(row[k], str) and row[k] for k in ("dataset", "task_id")),
            "Invalid case identity")
    identity = (index, row["dataset"], row["task_id"], count(row["seed"], "seed"))
    require(isinstance(row["output_ids"], list), "Expected output_ids list")
    output = tuple(count(i, "output_ids") for i in row["output_ids"])
    require(isinstance(row["events"], list) and row["events"], "Missing events")

    q, fresh, mask = split
    nominal = q - fresh - mask
    totals = Counter()
    work = defaultdict(Counter)
    selected_categories = Counter()
    path = Trajectory(output, [], [], [])
    previous = {}
    for idx, e in enumerate(row["events"]):
        try:
            stage, phase = e["stage"], e["phase"]
            physical = count(e["physical_q"], "physical_q")
            require(STAGE_PHASE.get(stage) == phase and stage in STAGE_PHASE, "Unexpected stage/phase")
            layers = e["layer_physical_q"]
            require(physical > 0 and e["F"] == 1 and physical == e["actual_q"]
                    and isinstance(layers, list) and layers and all(n == physical for n in layers),
                    "Physical query audit disagrees")
            work[stage].update(forwards=1, physical_query_rows=physical)
            if phase == "prefill":
                require("gate" not in e, "Unexpected gate on prefill")
                continue
            totals.update(generation_forwards=1, generation_physical_query_rows=physical)
            if phase == "commit":
                require("gate" not in e and physical == WIDTH, "Unexpected commit gate or width")
                continue

            selected = stage == "selected"
            require(e["q_budget"] == q and (physical <= q if selected else physical == WIDTH),
                    "Unexpected denoise query budget or width")
            block = count(e["block"], "block")
            step = count(e["step_in_block"], "step_in_block")
            checked = positions(e["absolute_positions"], "absolute_positions")
            require(step > 0 and len(checked) == physical
                    and all(block * WIDTH <= i < (block + 1) * WIDTH for i in checked),
                    "Invalid denoise clock or checked positions")
            g = e["gate"]
            cats = g["selected_category_counts"]
            require(isinstance(cats, dict) and cats.keys() == set(CATEGORIES)
                    and sum(count(n, k) for k, n in cats.items()) == physical,
                    "Category counts do not cover all physical checks")
            ages = g["selected_evidence_age"]
            require(isinstance(ages, list) and len(ages) == physical
                    and all(a is None or (type(a) is int and 0 < a < step) for a in ages),
                    "Invalid selected evidence ages")
            reserved = positions(g["reserved_positions"], "reserved_positions")
            strict = positions(g["strict_edit_positions"], "strict_edit_positions")
            hits = positions(g["reserved_strict_edit_positions"], "reserved_strict_edit_positions")
            prior = positions(g["previous_strict_edit_positions"], "previous_strict_edit_positions")
            previous_step, previous_strict = previous.get(block, (0, set()))
            require(step == previous_step + 1 and prior == previous_strict,
                    "Previous strict edits or denoise clock disagree")
            previous[block] = (step, strict)
            edited = count(e["edited"], "edited")
            require(reserved <= checked and strict <= checked and hits == reserved & strict
                    and len(strict) <= min(edited, physical - cats["mask"] - cats["prompt"]),
                    "Invalid reservation, strict edit, or reserved hit")
            available = count(g["available_near_miss_count"], "available_near_miss_count")
            nonfresh = count(g["available_nonfresh_count"], "available_nonfresh_count")
            reserved_near = count(g["reserved_near_miss_count"], "reserved_near_miss_count")
            released = count(g["released_edit_slots_to_mask"], "released_edit_slots_to_mask")
            other = cats["other_resolved"]
            require(cats["near_miss"] <= available <= nonfresh <= WIDTH
                    and other <= nonfresh - available and reserved_near <= min(len(reserved), cats["near_miss"])
                    and len(reserved) - reserved_near <= other,
                    "Candidate, selected, and reserved counts disagree")
            totals.update(denoise_passes=1, denoise_physical_checked=physical,
                          strict_edits=len(strict), native_edited=edited)
            work[stage].update(strict_edits=len(strict))
            # A block-local bitmask preserves positions without retaining the raw event.
            start = block * WIDTH
            path.denoise_checks.append((block, step, stage, sum(1 << (i-start) for i in checked)))
            if strict:
                path.strict_edits.append((block, step, sum(1 << (i-start) for i in strict)))
            if selected:
                require(cats["prompt"] == 0 and len(reserved) + released == nominal
                        and len(reserved) <= available
                        and (mode != "near_miss" or reserved_near == len(reserved)),
                        "Invalid selected-pass gate budget, eligibility, or reservation")
                for i, age in zip(e["absolute_positions"], ages):
                    if i in reserved:
                        group = ("unknown" if age is None else "previous" if age == 1
                                 else "two_steps" if age == 2 else "older")
                        totals[f"age_{group}_seats"] += 1
                        totals[f"age_{group}_hits"] += i in hits
                path.reservations.append((block, step, sum(1 << (i-start) for i in reserved), released))
                totals.update(selected_passes=1, selected_physical_checked=physical,
                              selected_strict_edits=len(strict), reserved_seats=len(reserved),
                              reserved_near_miss_seats=reserved_near, reserved_strict_hits=len(hits),
                              released_edit_slots_to_mask=released,
                              available_near_miss_exposures=available,
                              available_nonfresh_exposures=nonfresh)
                selected_categories.update(cats)
            else:
                require(not reserved and not released, "Reservations/releases on a non-selected pass")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"event {idx}: {exc}") from exc

    require(totals["denoise_passes"] > 0 and work["commit"]["forwards"] > 0,
            "Missing denoise or commit work")
    for field, actual in (
        ("generation_forwards", totals["generation_forwards"]),
        ("generation_query_rows", totals["generation_physical_query_rows"]),
        ("generation_physical_padded_rows", totals["generation_physical_query_rows"]),
        ("prefill_rows", work.get("prefill", {}).get("physical_query_rows", 0)),
        ("t2t_edits", totals["native_edited"]),
    ):
        require(count(row[field], field) == actual, f"Events disagree with {field}")
    return identity, totals, work, selected_categories, path


def ratio(n, d):
    return n / d if d else None


def summarize_arm(folder, policy, mode, split):
    identities, cases = [], []
    totals, categories = Counter(), Counter(dict.fromkeys(CATEGORIES, 0))
    stages = defaultdict(Counter)
    for i in range(CASES):
        path = folder / f"{i:02d}-{policy}.json"
        row = read_json(path)
        try:
            identity, counts, work, checked_cats, trajectory = summarize_case(row, i, policy, mode, split)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        identities.append(identity)
        cases.append((counts, trajectory))
        totals.update(counts)
        categories.update(checked_cats)
        for stage, stage_counts in work.items():
            stages[stage].update(stage_counts)
    # Keep zero-denominator arms and their counts explicit (e.g., no selected passes).
    for key in ("selected_passes", "selected_physical_checked", "selected_strict_edits",
                "reserved_seats", "reserved_near_miss_seats", "reserved_strict_hits",
                "released_edit_slots_to_mask", "available_near_miss_exposures",
                "available_nonfresh_exposures"):
        totals.setdefault(key, 0)
    q, fresh, mask = split
    by_age = {}
    for group in ("previous", "two_steps", "older", "unknown"):
        seats, hits = (totals.pop(f"age_{group}_{k}", 0) for k in ("seats", "hits"))
        by_age[group] = dict(seats=seats, hits=hits, precision=ratio(hits, seats))
    require(sum(v["seats"] for v in by_age.values()) == totals["reserved_seats"]
            and sum(v["hits"] for v in by_age.values()) == totals["reserved_strict_hits"],
            "Evidence age totals disagree")
    report = dict(policy=policy, selection=mode, requests=CASES, nominal_q=q,
                  fresh_slots=fresh, mask_slots=mask, nominal_edit_slots=q-fresh-mask,
                  reserved_by_observation_age=by_age,
                  **totals, work_by_stage=dict(stages), selected_checked_by_category=dict(categories),
                  reserved_hit_rate=ratio(totals["reserved_strict_hits"], totals["reserved_seats"]),
                  reserved_share_of_observed_selected_strict_edits=ratio(
                      totals["reserved_strict_hits"], totals["selected_strict_edits"]))
    require(len({identity[1:3] for identity in identities}) == CASES,
            f"{policy}: duplicate (dataset, task_id)")
    return identities, report, cases


def compare(near, random):
    near_arm, near_cases = near
    random_arm, random_cases = random
    keys = ("generation_physical_query_rows", "denoise_physical_checked", "selected_physical_checked",
            "selected_passes", "reserved_seats", "strict_edits", "selected_strict_edits",
            "reserved_strict_hits", "released_edit_slots_to_mask")
    paired = list(zip(near_cases, random_cases))
    return dict(
        near_miss_policy=near_arm["policy"], gate_random_policy=random_arm["policy"],
        requests=CASES,
        totals_near_miss_minus_gate_random={k: near_arm[k] - random_arm[k] for k in keys},
        cases_with_different_counts={k: sum(a[0][k] != b[0][k] for a, b in paired) for k in keys},
        cases_with_different_output_ids=sum(a[1].output_ids != b[1].output_ids for a, b in paired),
        cases_with_different_denoise_check_path=sum(a[1].denoise_checks != b[1].denoise_checks for a, b in paired),
        cases_with_different_strict_edit_path=sum(a[1].strict_edits != b[1].strict_edits for a, b in paired),
        cases_with_different_reservation_path=sum(a[1].reservations != b[1].reservations for a, b in paired),
    )


def summarize_runs(folders, only_complete=False):
    reference, arms, groups = None, [], defaultdict(dict)
    for folder, policy, mode, split in gated_arms(folders, only_complete):
        identities, report, cases = summarize_arm(folder, policy, mode, split)
        require(reference is None or identities == reference,
                f"{policy}: case identities or seeds differ across completed arms")
        reference = identities
        arms.append(report)
        groups[split][mode] = (report, cases)
    require(arms, "No complete frozen30 LLaDA gate arms")
    return dict(
        model=MODEL, block=WIDTH, scope="online_own_trajectories",
        interpretation=(
            "Hits are strict edits on reserved positions actually checked by that arm. "
            "Released slots include force-MASK redirects and count quota, not actual MASK checks. "
            "Same-split deltas pair case identities, not model states or seat counts; "
            "they do not measure offline recall or edits on unchecked rows."
        ),
        arms=sorted(arms, key=lambda a: a["policy"]),
        online_own_trajectory_comparisons=[
            compare(group[GATES[0]], group[GATES[1]])
            for _, group in sorted(groups.items()) if all(mode in group for mode in GATES)
        ],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", type=Path, required=True,
                        help="run_t2t.py output directory; repeat for disjoint arms")
    parser.add_argument("--only-complete", action="store_true",
                        help="Skip gate arms without a completion marker")
    args = parser.parse_args(argv)
    try:
        report = summarize_runs(args.run, args.only_complete)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
