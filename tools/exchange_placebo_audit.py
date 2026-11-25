"""Independent CPU audit of paired LLaDA candidate-placebo exchange runs.

The audit compares the historical ranked exchange archive with the two
matched-victim candidate placebos (``random_a`` and ``random_b``) at Q4 and
Q8.  It needs only JSON run directories: no torch, checkpoint, GPU, scoring,
network, or results copy.  ``--run`` is repeatable so the two placebo arms may
live in separate directories.

The backend records, on a selected step that actually swaps::

    event["placebo_exchange"] = {
        "base": ...,            # block-local zero-reserve Q set
        "ranked": ...,          # block-local ranked-exchange Q set
        "executed": ...,        # block-local placebo Q set
        "victims": ...,         # block-local victims removed from base
        "pool": ...,            # block-local eligible old candidates
        "fresh": ...,           # block-local mandatory-fresh seats
        "masked_victims": ...,  # victims that were MASK before the step
    }

All seven arrays are LOCAL to the block.  ``event["absolute_positions"]`` and
``event["reserved_extra_edit_positions"]`` are ABSOLUTE, with offset
``event["block"] * width``.

Per paired request the audit:

* strips diagnostic events and diagnostic-only fields, then requires identical
  histories before the first difference;
* at the first physical Q split, replays the ranked exchange from the recorded
  locals and checks it against the archived absolute positions and reserved
  rows (same victims, gate count, pool size, and width);
* if nothing diverges, requires identical outputs and physical F/Q;
* after a fork, does not require a shared trajectory;
* independently checks every logged swap: the executed set reconstructs the
  physical rows, fresh seats are protected, candidates come from the pool,
  masked victims and reserved strict hits are consistent, and reserved rows do
  not overlap base or victims;
* at the first differing common-history step, records MASK revocations and
  admissions for both arms and checks
  ``0 <= admissions <= accepted <= selected_masked_before``.

The report is one CSV with ``row_type=request`` rows and ``row_type=total``
rows (by budget/arm and overall).  Counts cover swaps by victim type,
candidate collisions with the ranked replacement, direct strict hits, and
first-fork categories.  The audit only verifies observable event metadata, not
layer caches or logits.

Strict mode (the default) requires the frozen30 manifest and each arm's
``qualified.json`` plus ``complete-<policy>.json`` marker.  Expected request
indices come from the manifest, so a request missing from both sides is still
reported.  ``--allow-missing`` relaxes the completion marker for a live run,
but a missing placebo arm is always an error.  The public CSV is written with
LF line endings, and the ``run`` column uses directory basenames only.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from random import Random


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks/frozen30/manifest.json"

POLICY = re.compile(r"^t2t_q(\d+)_f(\d+)_m(\d+)_(.+)$")
REQUEST = re.compile(r"^(\d+)-(.+)\.json$")

SELECTION_A = "hard_exchange_random_a"
SELECTION_B = "hard_exchange_random_b"
SELECTION_OLD_A = "hard_exchange_oldhalf_a"
SELECTION_OLD_B = "hard_exchange_oldhalf_b"
OLDHALF_BUDGETS = (4, 8)

DOC_KEYS = ("base", "ranked", "executed", "victims", "pool", "fresh",
            "masked_victims")

DIAGNOSTIC_KEYS = frozenset((
    "diagnostic",
    "screened_candidates",
    "screen_hits",
    "trajectory_sha256",
    "accepted_sha256",
    "intervention",
))

STRUCTURAL_FIELDS = (
    "stage", "block", "q_tokens", "q_budget", "physical_q", "actual_q",
    "layer_physical_q", "F", "prefix_kv_length", "visible_kv",
    "context_version", "step_in_block", "hard_clock", "mandatory_fresh",
    "masked_before", "selected_masked_before", "overflow", "store_kv",
    "approximate_same_block_kv",
)

OUTPUT_FIELDS = ("output_ids", "output_tokens", "eos", "truncated")
WORK_FIELDS = (
    "generation_forwards", "generation_query_rows",
    "generation_physical_padded_rows", "prefill_rows", "max_q",
)
PHASE_FIELDS = ("forwards_by_phase", "q_by_phase", "q_histogram")

FIELDS = (
    "row_type", "group", "run", "policy", "kind", "budget", "index",
    "dataset", "task_id", "status",
    "requests", "forked", "nondivergent", "outputs_identical",
    "physical_fq_identical",
    "output_tokens", "generation_forwards", "generation_query_rows",
    "generation_physical_padded_rows", "prefill_rows", "max_q",
    "first_fork", "first_q_split", "history_mismatch", "first_swap",
    "first_fork_block", "first_fork_clock", "first_fork_victim",
    "first_fork_swaps", "first_fork_gate", "first_fork_base_size",
    "first_fork_pool_size", "first_fork_width", "first_fork_ranked_replay",
    "first_fork_draw_pool_size", "first_fork_all_tied",
    "first_fork_new_reserved_rows", "first_fork_ranked_reserved_rows",
    "first_fork_new_strict_hits", "first_fork_ranked_strict_hits",
    "first_fork_new_strict_rows", "first_fork_ranked_strict_rows",
    "fork_new_revocations", "fork_new_admissions", "fork_new_accepted",
    "fork_new_selected_masked_before",
    "fork_ranked_revocations", "fork_ranked_admissions",
    "fork_ranked_accepted", "fork_ranked_selected_masked_before",
    "swap_events", "swaps_total", "swaps_mask", "swaps_old", "collisions",
    "direct_strict_hits", "reserved_rows", "collision_rows",
    "direct_strict_rows",
    "age_swaps", "age_all_tied_swaps", "age_tie_sampled_swaps",
    "age_draw_pool_rows",
    "first_fork_mask", "first_fork_old", "first_fork_mixed",
    "first_fork_unknown", "first_fork_none", "fork_at_first_swap",
    "first_fork_ranked_ok",
    "violation_requests", "violations",
)

MISSING = object()


def parse_policy(policy):
    match = POLICY.match(policy or "")
    if match is None:
        return None
    budget, fresh, mask, selection = match.groups()
    return dict(budget=int(budget), fresh=int(fresh), mask=int(mask),
                selection=selection)


def arm_kind(selection):
    if SELECTION_OLD_A in selection or selection.endswith("_oldhalf_a"):
        return "oldhalf_a"
    if SELECTION_OLD_B in selection or selection.endswith("_oldhalf_b"):
        return "oldhalf_b"
    if SELECTION_A in selection or selection.endswith("_random_a"):
        return "random_a"
    if SELECTION_B in selection or selection.endswith("_random_b"):
        return "random_b"
    if "exchange" in selection:
        return "ranked"
    return None


def oldhalf_salt(kind):
    if kind == "oldhalf_a":
        return "a"
    if kind == "oldhalf_b":
        return "b"
    raise ValueError("Not an old-half arm: %s" % kind)


def rank_tag(selection):
    if "edit4" in selection:
        return "edit4"
    if "equal" in selection:
        return "equal"
    return None


def _is_request_name(name):
    match = REQUEST.match(name)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def _read_json(path):
    return json.loads(Path(path).read_text())


def _load_manifest(source):
    if source is None:
        if not DEFAULT_MANIFEST.exists():
            return None
        data = _read_json(DEFAULT_MANIFEST)
    elif isinstance(source, dict):
        data = source
    else:
        data = _read_json(source)
    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Manifest has no samples list")
    expected = {}
    for sample in samples:
        index = _as_int(sample.get("index"))
        expected[index] = dict(
            index=index, task_id=sample.get("task_id"),
            dataset=sample.get("dataset"), seed=_as_int(sample.get("seed")))
    return expected


def _source_label(arm):
    names = sorted({Path(source).name for source in arm.sources})
    return ";".join(names)


def _work_row(result):
    output = result.get("output_ids", [])
    return dict(
        output_tokens=len(output) if isinstance(output, list) else "",
        generation_forwards=result.get("generation_forwards", ""),
        generation_query_rows=result.get("generation_query_rows", ""),
        generation_physical_padded_rows=result.get(
            "generation_physical_padded_rows", ""),
        prefill_rows=result.get("prefill_rows", ""),
        max_q=result.get("max_q", ""),
    )


@dataclass
class ArmData:
    policy: str
    budget: int
    fresh: int
    mask: int
    selection: str
    kind: str
    width: int
    model: str
    files: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)
    rates: str = ""


@dataclass
class RunSpec:
    path: Path
    model: str
    width: int
    arms: list


def _request_files(folder, policy):
    found = {}
    if not folder.is_dir():
        return found
    for path in sorted(folder.iterdir()):
        parsed = _is_request_name(path.name)
        if parsed is None or parsed[1] != policy or not path.is_file():
            continue
        index = parsed[0]
        if index in found:
            raise ValueError("%s: duplicate request index %d for %s"
                             % (folder, index, policy))
        found[index] = path
    return found


def _looks_like_run(folder):
    if not folder.is_dir():
        return False
    if (folder / "run.json").exists():
        return True
    for path in folder.iterdir():
        parsed = _is_request_name(path.name)
        if parsed is not None and parse_policy(parsed[1]) is not None:
            return True
    return False


def expand_run_paths(paths):
    """Yield run roots; a parent of run directories is accepted for convenience."""
    out = []
    for path in map(Path, paths):
        if _looks_like_run(path):
            out.append(path)
            continue
        children = [child for child in sorted(path.iterdir()) if child.is_dir()
                    and _looks_like_run(child)] if path.is_dir() else []
        if not children:
            raise ValueError("No run.json or per-request JSON under %s" % path)
        out.extend(children)
    return out


def load_run(folder):
    folder = Path(folder)
    meta_path = folder / "run.json"
    meta = _read_json(meta_path) if meta_path.exists() else {}
    model = meta.get("model", "llada21")
    width = int(meta.get("block", 32))
    rates = meta.get("exchange_rates_sha256", "")
    specs = []
    policies = list(meta.get("policies", []))
    arms = list(meta.get("arms", []))
    if policies:
        if arms and len(arms) != len(policies):
            raise ValueError("%s: policies/arms length mismatch" % folder)
        for position, policy in enumerate(policies):
            parsed = parse_policy(policy)
            if parsed is None:
                continue
            if arms:
                arm = arms[position]
                for key in ("budget", "fresh", "mask", "selection"):
                    if key in arm and arm[key] != parsed[key]:
                        raise ValueError("%s: arm/policy disagree for %s"
                                         % (folder, policy))
            specs.append(parsed)
    elif arms:
        for arm in arms:
            policy = "t2t_q%d_f%d_m%d_%s" % (
                int(arm["budget"]), int(arm["fresh"]), int(arm["mask"]),
                arm["selection"])
            parsed = parse_policy(policy)
            if parsed is None:
                raise ValueError("%s: invalid arm %r" % (folder, arm))
            specs.append(parsed)
    else:
        seen = {}
        for path in sorted(folder.iterdir()):
            parsed = _is_request_name(path.name)
            if parsed is None or not path.is_file():
                continue
            spec = parse_policy(parsed[1])
            if spec is not None:
                seen[parsed[1]] = spec
        specs = list(seen.values())
    known = {"t2t_q%d_f%d_m%d_%s" % (
        spec["budget"], spec["fresh"], spec["mask"], spec["selection"])
        for spec in specs}
    for path in sorted(folder.iterdir()):
        parsed = _is_request_name(path.name)
        if parsed is None or not path.is_file() or parsed[1] in known:
            continue
        spec = parse_policy(parsed[1])
        if spec is not None:
            specs.append(spec)
            known.add(parsed[1])
    out = []
    for spec in specs:
        kind = arm_kind(spec["selection"])
        if kind is None:
            continue
        out.append(ArmData(
            policy="t2t_q%d_f%d_m%d_%s" % (
                spec["budget"], spec["fresh"], spec["mask"], spec["selection"]),
            budget=spec["budget"], fresh=spec["fresh"], mask=spec["mask"],
            selection=spec["selection"], kind=kind, width=width, model=model,
            rates=rates,
            files=_request_files(folder, "t2t_q%d_f%d_m%d_%s" % (
                spec["budget"], spec["fresh"], spec["mask"], spec["selection"])),
            sources=[str(folder)],
        ))
    if not out:
        raise ValueError("%s: no exchange arms" % folder)
    return RunSpec(folder, model, width, out)


def merge_arms(specs):
    merged = {}
    for spec in specs:
        for arm in spec.arms:
            key = arm.policy
            if key not in merged:
                merged[key] = ArmData(
                    policy=arm.policy, budget=arm.budget, fresh=arm.fresh,
                    mask=arm.mask, selection=arm.selection, kind=arm.kind,
                    width=arm.width, model=arm.model, files=dict(arm.files),
                    sources=list(arm.sources), rates=arm.rates)
                continue
            current = merged[key]
            if (current.budget, current.fresh, current.mask, current.kind,
                    current.width, current.model) != (
                    arm.budget, arm.fresh, arm.mask, arm.kind, arm.width,
                    arm.model):
                raise ValueError("Inconsistent duplicate arm %s" % key)
            if (current.rates and arm.rates
                    and current.rates != arm.rates):
                raise ValueError("Inconsistent exchange rates for %s" % key)
            current.rates = current.rates or arm.rates
            for index, path in arm.files.items():
                if index in current.files and current.files[index] != path:
                    raise ValueError("Duplicate request %02d for %s" % (index, key))
                current.files[index] = path
            current.sources.extend(arm.sources)
    return merged


def discover(ranked_paths, run_paths, budgets, mode="placebo"):
    archived_specs = [load_run(path) for path in expand_run_paths(ranked_paths)]
    new_specs = [load_run(path) for path in expand_run_paths(run_paths)]
    archived = {}
    if mode == "oldhalf":
        wanted = ("random_a", "random_b")
        new_kinds = ("oldhalf_a", "oldhalf_b")
        label = "old-half"
    else:
        wanted = ("ranked",)
        new_kinds = ("random_a", "random_b")
        label = "candidate-placebo"
    for arm in merge_arms(archived_specs).values():
        if arm.kind in wanted and arm.budget in budgets:
            archived.setdefault(arm.budget, []).append(arm)
    new = []
    for arm in merge_arms(new_specs).values():
        if arm.kind in new_kinds and arm.budget in budgets:
            new.append(arm)
    if not new:
        raise ValueError("No %s arms at budgets %s" % (label, budgets))
    for budget in sorted({arm.budget for arm in new}):
        if budget not in archived:
            raise ValueError("No %s reference at Q%d" % (label, budget))
    return archived, new


def select_reference(archived, arm, mode="placebo"):
    candidates = archived[arm.budget]
    if mode == "oldhalf":
        salt = oldhalf_salt(arm.kind)
        matching = [candidate for candidate in candidates
                    if candidate.kind == "random_" + salt]
        if len(matching) != 1:
            raise ValueError(
                "Expected one uniform random_%s reference at Q%d for %s"
                % (salt, arm.budget, arm.policy))
        return matching[0]
    tag = rank_tag(arm.selection) or "edit4"
    matching = [candidate for candidate in candidates
                if rank_tag(candidate.selection) == tag]
    if len(matching) == 1:
        return matching[0]
    edit4 = [candidate for candidate in candidates
             if rank_tag(candidate.selection) == "edit4"]
    if len(edit4) == 1:
        return edit4[0]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError("Ambiguous archived ranked reference at Q%d: %s"
                     % (arm.budget, [candidate.policy for candidate in candidates]))


def _as_int(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("not an integer")
    return value


def _position_list(value, label, violations, width=None):
    if not isinstance(value, (list, tuple)):
        violations.append("%s: not a position list" % label)
        return []
    out = []
    for item in value:
        try:
            position = _as_int(item)
        except ValueError:
            violations.append("%s: non-integer position %r" % (label, item))
            return []
        if width is not None and not 0 <= position < width:
            violations.append("%s: position %d outside width %d"
                              % (label, position, width))
        out.append(position)
    if len(out) != len(set(out)):
        violations.append("%s: duplicate positions" % label)
    if out != sorted(out):
        violations.append("%s: positions are not sorted" % label)
    return out


def _field_int(event, key, label, violations):
    if key not in event or event[key] is None:
        violations.append("%s: missing %s" % (label, key))
        return None
    try:
        return _as_int(event[key])
    except ValueError:
        violations.append("%s: %s is not an integer" % (label, key))
        return None


def _admission_metrics(event, label, violations):
    values = {}
    for key in ("edited", "hard_strict_edits", "masked_before", "masked_after",
                "accepted", "selected_masked_before"):
        values[key] = _field_int(event, key, label, violations)
    if any(value is None for value in values.values()):
        return None
    revocations = values["edited"] - values["hard_strict_edits"]
    admissions = values["masked_before"] - values["masked_after"] + revocations
    accepted = values["accepted"]
    selected = values["selected_masked_before"]
    if revocations < 0:
        violations.append("%s: negative revocations %d" % (label, revocations))
    if not 0 <= admissions <= accepted <= selected:
        violations.append(
            "%s: admission chain 0<=%d<=%d<=%d failed"
            % (label, admissions, accepted, selected))
    return dict(revocations=revocations, admissions=admissions,
                accepted=accepted, selected_masked_before=selected)


def _doc_metrics(event, width, label, violations):
    block = _field_int(event, "block", label, violations)
    if block is None:
        return None
    offset = block * width
    doc = event.get("placebo_exchange")
    if not isinstance(doc, dict):
        violations.append("%s: missing placebo_exchange document" % label)
        return None
    missing = [key for key in DOC_KEYS if key not in doc]
    if missing:
        violations.append("%s: placebo_exchange missing %s"
                          % (label, ", ".join(missing)))
        return None
    arrays = {}
    for key in DOC_KEYS:
        arrays[key] = _position_list(doc[key], "%s.%s" % (label, key),
                                     violations, width)
    base = arrays["base"]
    ranked = arrays["ranked"]
    executed = arrays["executed"]
    victims = arrays["victims"]
    pool = arrays["pool"]
    fresh = arrays["fresh"]
    masked_victims = arrays["masked_victims"]
    base_set, ranked_set = set(base), set(ranked)
    executed_set, victim_set = set(executed), set(victims)
    pool_set, fresh_set = set(pool), set(fresh)
    masked_set = set(masked_victims)

    if event.get("stage") != "selected":
        violations.append("%s: placebo_exchange outside selected stage" % label)
    if not (len(base) == len(ranked) == len(executed)):
        violations.append("%s: base/ranked/executed widths differ" % label)
    if len(base) > width:
        violations.append("%s: base width %d exceeds %d"
                          % (label, len(base), width))
    if not victim_set <= base_set:
        violations.append("%s: victims outside base" % label)
    if base_set - ranked_set != victim_set:
        violations.append("%s: base-ranked victims differ from victims" % label)
    if len(ranked_set - base_set) != len(victims):
        violations.append("%s: victim and replacement counts differ" % label)
    if not fresh_set <= base_set:
        violations.append("%s: fresh outside base" % label)
    if not fresh_set <= ranked_set:
        violations.append("%s: fresh outside ranked" % label)
    if not fresh_set <= executed_set:
        violations.append("%s: fresh outside executed" % label)
    if victim_set & fresh_set:
        violations.append("%s: fresh displaced by a victim" % label)
    if not masked_set <= victim_set:
        violations.append("%s: masked_victims outside victims" % label)
    if pool_set & base_set:
        violations.append("%s: pool overlaps base" % label)
    if pool_set & fresh_set:
        violations.append("%s: pool overlaps fresh" % label)
    if victim_set & pool_set:
        violations.append("%s: pool overlaps victims" % label)
    if not ranked_set - base_set <= pool_set:
        violations.append("%s: ranked replacements outside pool" % label)
    replacements = executed_set - base_set
    if not replacements <= pool_set:
        violations.append("%s: executed replacements outside pool" % label)
    if len(replacements) != len(victims):
        violations.append("%s: executed replacement count differs from victims"
                          % label)
    if executed_set & base_set != base_set - victim_set:
        violations.append("%s: executed retained rows differ from base-victims"
                          % label)
    if len(pool) < len(victims):
        violations.append("%s: pool smaller than victim count" % label)
    if (event.get("masked_before") is not None
            and len(masked_victims) > event["masked_before"]):
        violations.append("%s: masked victims exceed masked_before" % label)
    if (event.get("selected_masked_before") is not None
            and event.get("masked_before") is not None
            and event["selected_masked_before"] > event["masked_before"]):
        violations.append("%s: selected_masked_before exceeds masked_before"
                          % label)
    if (event.get("mandatory_fresh") is not None
            and len(fresh) > event["mandatory_fresh"]):
        violations.append("%s: recorded fresh exceeds mandatory_fresh" % label)

    absolute = _position_list(event.get("absolute_positions", []),
                              "%s.absolute_positions" % label, violations)
    expected_absolute = sorted(offset + position for position in executed)
    if sorted(absolute) != expected_absolute:
        violations.append("%s: executed locals do not reconstruct physical rows"
                          % label)
    for position in absolute:
        if not offset <= position < offset + width:
            violations.append("%s: absolute position %d outside block %d"
                              % (label, position, block))

    reserved = _position_list(event.get("reserved_extra_edit_positions", []),
                              "%s.reserved_extra_edit_positions" % label,
                              violations)
    expected_reserved = sorted(offset + position for position in replacements)
    if sorted(reserved) != expected_reserved:
        violations.append("%s: reserved rows are not the executed replacements"
                          % label)
    reserved_set = set(reserved)
    if reserved_set & {offset + position for position in base}:
        violations.append("%s: reserved rows overlap base" % label)
    if reserved_set & {offset + position for position in victims}:
        violations.append("%s: reserved rows overlap victims" % label)

    strict = _position_list(event.get("reserved_strict_edit_positions", []),
                            "%s.reserved_strict_edit_positions" % label,
                            violations)
    if not set(strict) <= reserved_set:
        violations.append("%s: strict reserved rows outside reserved rows"
                          % label)
    if set(strict) & {offset + position for position in masked_victims}:
        violations.append("%s: strict reserved rows overlap masked victims"
                          % label)
    hard_strict = event.get("hard_strict_edits")
    if hard_strict is not None:
        try:
            if len(strict) > _as_int(hard_strict):
                violations.append(
                    "%s: strict reserved hits exceed hard_strict_edits" % label)
        except ValueError:
            violations.append("%s: hard_strict_edits is not an integer" % label)

    collisions = sorted((replacements & (ranked_set - base_set)))
    return dict(
        offset=offset, base=base, ranked=ranked, executed=executed,
        victims=victims, pool=pool, fresh=fresh,
        masked_victims=masked_victims, reserved=reserved, strict=strict,
        collisions=sorted(offset + position for position in collisions),
        direct_strict=sorted(strict),
        swaps=len(victims), masked_swaps=len(masked_victims),
    )


def _normalized_event(event):
    return {key: value for key, value in event.items()
            if key not in DIAGNOSTIC_KEYS and key != "placebo_exchange"}


def _common_events(result):
    events = result.get("events")
    if not isinstance(events, list):
        return []
    return [event for event in events
            if isinstance(event, dict) and not event.get("diagnostic")]


def _event_positions(event):
    positions = event.get("absolute_positions", [])
    try:
        return sorted(_as_int(position) for position in positions)
    except (TypeError, ValueError):
        return []


def _plain_absolute(value):
    try:
        return sorted(_as_int(position) for position in (value or []))
    except (TypeError, ValueError):
        return []


def _result_comparison(new, archived):
    mismatches = []
    output_ok = True
    for key in OUTPUT_FIELDS:
        left, right = new.get(key, MISSING), archived.get(key, MISSING)
        if left != right:
            output_ok = False
            mismatches.append(key)
    physical_ok = True
    for key in WORK_FIELDS + PHASE_FIELDS:
        left, right = new.get(key, MISSING), archived.get(key, MISSING)
        if left != right:
            physical_ok = False
            mismatches.append(key)
    return output_ok, physical_ok, mismatches


def _metadata_violations(new, archived, arm, reference):
    violations = []
    for key in ("index", "task_id", "dataset", "seed", "model", "prefill_rows"):
        if new.get(key) != archived.get(key):
            violations.append("metadata mismatch: %s" % key)
    if new.get("policy") != arm.policy:
        violations.append("new policy does not match filename")
    if archived.get("policy") != reference.policy:
        violations.append("archived policy does not match filename")
    if new.get("model") != "llada21" or archived.get("model") != "llada21":
        violations.append("not a llada21 pair")
    if arm.width != reference.width:
        violations.append("paired runs use different block widths")
    if arm.model != reference.model:
        violations.append("paired runs use different models")
    if arm.rates and reference.rates and arm.rates != reference.rates:
        violations.append("paired runs use different exchange-rate tables")
    if new.get("status") != "success":
        violations.append("new status is not success")
    if archived.get("status") != "success":
        violations.append("archived status is not success")
    return violations


def _manifest_violations(result, sample, label):
    if sample is None:
        return ["%s: request index is not in the frozen manifest" % label]
    violations = []
    for key in ("task_id", "dataset", "seed"):
        if result.get(key) != sample[key]:
            violations.append("%s: manifest %s mismatch" % (label, key))
    return violations


def _environment_violations(arm, require_completion):
    violations = []
    qualified = False
    completed = False
    for source in arm.sources:
        folder = Path(source)
        meta_path = folder / "run.json"
        if meta_path.exists():
            meta = _read_json(meta_path)
            if meta.get("model") != "llada21":
                violations.append("%s: run.json model is not llada21"
                                  % folder.name)
            if int(meta.get("block", arm.width)) != arm.width:
                violations.append("%s: run.json block differs" % folder.name)
            policies = list(meta.get("policies", []))
            if policies and arm.policy not in policies:
                violations.append("%s: run.json does not list %s"
                                  % (folder.name, arm.policy))
            rates = meta.get("exchange_rates_sha256")
            if arm.rates and rates and rates != arm.rates:
                violations.append("%s: run.json exchange rates differ"
                                  % folder.name)
        qualified_path = folder / "qualified.json"
        if qualified_path.exists():
            qualified_row = _read_json(qualified_path)
            if (qualified_row.get("model") == "llada21"
                    and int(qualified_row.get("full_width", -1)) == arm.width
                    and int(qualified_row.get("matched", -1)) == 30):
                qualified = True
            else:
                violations.append("%s: invalid full-width qualification"
                                  % folder.name)
        complete_path = folder / ("complete-%s.json" % arm.policy)
        if complete_path.exists():
            complete_row = _read_json(complete_path)
            if (complete_row.get("policy") == arm.policy
                    and int(complete_row.get("requests", -1)) == 30):
                completed = True
            else:
                violations.append("%s: invalid completion marker for %s"
                                  % (folder.name, arm.policy))
    if require_completion:
        if not qualified:
            violations.append("unqualified run: no valid qualified.json")
        if not completed:
            violations.append("incomplete arm: no valid complete-%s.json"
                              % arm.policy)
    return violations


def _victim_type(metrics):
    if metrics is None:
        return "unknown"
    masked, total = metrics["masked_swaps"], metrics["swaps"]
    if total == 0:
        return "none"
    if masked == total:
        return "mask"
    if masked == 0:
        return "old"
    return "mixed"


def _audit_fork(new_event, archived_event, metrics, width, violations, row):
    if metrics is None:
        row["first_fork_ranked_replay"] = "missing"
        return
    for key in STRUCTURAL_FIELDS:
        if new_event.get(key, MISSING) != archived_event.get(key, MISSING):
            violations.append("fork structural mismatch: %s" % key)
    offset = metrics["offset"]
    ranked = metrics["ranked"]
    base = metrics["base"]
    victims = metrics["victims"]
    pool = metrics["pool"]
    ranked_reserved = sorted(set(ranked) - set(base))
    expected_ranked = sorted(offset + position for position in ranked)
    archived_positions = _position_list(
        archived_event.get("absolute_positions", []),
        "archived.absolute_positions", violations)
    if sorted(archived_positions) != expected_ranked:
        violations.append("fork: ranked locals do not replay archived positions")
        row["first_fork_ranked_replay"] = "mismatch"
    else:
        row["first_fork_ranked_replay"] = "ok"
    archived_reserved = _position_list(
        archived_event.get("reserved_extra_edit_positions", []),
        "archived.reserved_extra_edit_positions", violations)
    expected_reserved = sorted(offset + position for position in ranked_reserved)
    if sorted(archived_reserved) != expected_reserved:
        violations.append("fork: ranked locals do not replay archived reserved rows")
    if len(archived_reserved) != len(victims):
        violations.append("fork: archived gate count differs from victims")
    derived_victims = sorted(set(base) - set(ranked))
    if derived_victims != sorted(victims):
        violations.append("fork: base-ranked victims differ from logged victims")
    if not set(ranked_reserved) <= set(pool):
        violations.append("fork: ranked replacement outside pool")
    if len(pool) < len(victims):
        violations.append("fork: pool smaller than victims")
    if len(base) > width:
        violations.append("fork: width overflow")
    row["first_fork_swaps"] = len(victims)
    row["first_fork_gate"] = len(victims)
    row["first_fork_base_size"] = len(base)
    row["first_fork_pool_size"] = len(pool)
    row["first_fork_width"] = width


def audit_request(new, archived, arm, reference, index, label, sample=None):
    violations = _metadata_violations(new, archived, arm, reference)
    if sample is not None:
        violations.extend(_manifest_violations(new, sample, label + " new"))
        violations.extend(_manifest_violations(
            archived, sample, label + " ranked"))
    row = dict(row_type="request", run=_source_label(arm), policy=arm.policy,
               kind=arm.kind, budget=arm.budget, index=index,
               dataset=new.get("dataset", ""), task_id=new.get("task_id", ""),
               status=new.get("status", ""))
    width = arm.width
    new_events = _common_events(new)
    old_events = _common_events(archived)
    if not new_events or not old_events:
        violations.append("missing event history")

    doc_events = {}
    first_swap = None
    totals = Counter()
    collision_rows = []
    direct_strict_rows = []
    reserved_rows = []
    for tick, event in enumerate(new_events):
        if "placebo_exchange" not in event:
            continue
        if first_swap is None:
            first_swap = tick
        metrics = _doc_metrics(event, width, label + " tick %d" % tick,
                               violations)
        if metrics is None:
            continue
        doc_events[tick] = metrics
        totals["swap_events"] += 1
        totals["swaps_total"] += metrics["swaps"]
        totals["swaps_mask"] += metrics["masked_swaps"]
        totals["swaps_old"] += metrics["swaps"] - metrics["masked_swaps"]
        totals["collisions"] += len(metrics["collisions"])
        totals["direct_strict_hits"] += len(metrics["direct_strict"])
        reserved_rows.extend(metrics["reserved"])
        collision_rows.extend(metrics["collisions"])
        direct_strict_rows.extend(metrics["direct_strict"])

    common = min(len(new_events), len(old_events))
    q_split = None
    for tick in range(common):
        if _event_positions(new_events[tick]) != _event_positions(old_events[tick]):
            q_split = tick
            break
    stop = q_split if q_split is not None else common
    history_mismatch = None
    for tick in range(stop):
        if _normalized_event(new_events[tick]) != _normalized_event(old_events[tick]):
            history_mismatch = tick
            break
    length_mismatch = q_split is None and len(new_events) != len(old_events)
    if history_mismatch is not None:
        violations.append("pre-divergence history mismatch at tick %d"
                          % history_mismatch)
    if length_mismatch:
        violations.append("event-count divergence without a physical Q split")

    candidates = [tick for tick in (history_mismatch, q_split)
                  if tick is not None]
    if length_mismatch:
        candidates.append(common)
    fork_tick = min(candidates) if candidates else None
    forked = fork_tick is not None
    output_ok, physical_ok, result_mismatch = _result_comparison(new, archived)
    row["outputs_identical"] = int(output_ok)
    row["physical_fq_identical"] = int(physical_ok)
    row["forked"] = int(forked)
    row["nondivergent"] = int(not forked)
    row["first_fork"] = "" if fork_tick is None else fork_tick
    row["first_q_split"] = "" if q_split is None else q_split
    row["history_mismatch"] = "" if history_mismatch is None else history_mismatch
    row["first_swap"] = "" if first_swap is None else first_swap
    row["swap_events"] = totals["swap_events"]
    row["swaps_total"] = totals["swaps_total"]
    row["swaps_mask"] = totals["swaps_mask"]
    row["swaps_old"] = totals["swaps_old"]
    row["collisions"] = totals["collisions"]
    row["direct_strict_hits"] = totals["direct_strict_hits"]
    row["reserved_rows"] = ";".join(map(str, sorted(reserved_rows)))
    row["collision_rows"] = ";".join(map(str, sorted(collision_rows)))
    row["direct_strict_rows"] = ";".join(map(str, sorted(direct_strict_rows)))

    if q_split is not None:
        _audit_fork(new_events[q_split], old_events[q_split],
                    doc_events.get(q_split), width, violations, row)
    else:
        row["first_fork_ranked_replay"] = "" if not forked else "missing"

    if not forked:
        if not output_ok:
            violations.append("nondivergent output mismatch: %s"
                              % ", ".join(result_mismatch))
        if not physical_ok:
            violations.append("nondivergent physical F/Q mismatch: %s"
                              % ", ".join(result_mismatch))
        row["first_fork_victim"] = "none"
    elif fork_tick < len(new_events) and fork_tick < len(old_events):
        metrics = doc_events.get(fork_tick)
        row["first_fork_victim"] = _victim_type(metrics)
        new_event = new_events[fork_tick]
        old_event = old_events[fork_tick]
        new_reserved = (metrics["reserved"] if metrics is not None
                        else _plain_absolute(
                            new_event.get("reserved_extra_edit_positions")))
        new_strict = (metrics["direct_strict"] if metrics is not None
                      else _plain_absolute(
                          new_event.get("reserved_strict_edit_positions")))
        old_reserved = _plain_absolute(
            old_event.get("reserved_extra_edit_positions"))
        old_strict = _plain_absolute(
            old_event.get("reserved_strict_edit_positions"))
        row["first_fork_new_reserved_rows"] = ";".join(
            map(str, new_reserved))
        row["first_fork_ranked_reserved_rows"] = ";".join(
            map(str, old_reserved))
        row["first_fork_new_strict_hits"] = len(new_strict)
        row["first_fork_ranked_strict_hits"] = len(old_strict)
        row["first_fork_new_strict_rows"] = ";".join(map(str, new_strict))
        row["first_fork_ranked_strict_rows"] = ";".join(map(str, old_strict))
        if metrics is not None:
            row["first_fork_swaps"] = len(metrics["victims"])
            row["first_fork_gate"] = len(metrics["victims"])
            row["first_fork_base_size"] = len(metrics["base"])
            row["first_fork_pool_size"] = len(metrics["pool"])
            row["first_fork_width"] = width
        else:
            for key in ("first_fork_swaps", "first_fork_gate",
                        "first_fork_base_size", "first_fork_pool_size",
                        "first_fork_width"):
                row[key] = ""
        row["first_fork_block"] = new_event.get("block", "")
        row["first_fork_clock"] = new_event.get("hard_clock", "")
        new_adm = _admission_metrics(
            new_events[fork_tick], label + " fork new", violations)
        old_adm = _admission_metrics(
            old_events[fork_tick], label + " fork ranked", violations)
        if new_adm is not None:
            row["fork_new_revocations"] = new_adm["revocations"]
            row["fork_new_admissions"] = new_adm["admissions"]
            row["fork_new_accepted"] = new_adm["accepted"]
            row["fork_new_selected_masked_before"] = new_adm["selected_masked_before"]
        if old_adm is not None:
            row["fork_ranked_revocations"] = old_adm["revocations"]
            row["fork_ranked_admissions"] = old_adm["admissions"]
            row["fork_ranked_accepted"] = old_adm["accepted"]
            row["fork_ranked_selected_masked_before"] = old_adm["selected_masked_before"]
        if (new_adm is not None and old_adm is not None
                and new_adm["selected_masked_before"]
                != old_adm["selected_masked_before"]):
            violations.append("fork: selected MASK counts differ between arms")
        if q_split is not None and fork_tick == q_split and doc_events.get(q_split) is None:
            violations.append("first Q split without placebo_exchange document")
    else:
        row["first_fork_victim"] = "unknown"
    return row, violations


def _base_row(arm, kind=None):
    return dict(row_type="request", run=_source_label(arm),
                policy=arm.policy, kind=kind or arm.kind, budget=arm.budget)


def _missing_row(arm, index, side):
    row = _base_row(arm)
    row.update(index=index, status="missing")
    row["violations"] = "missing %s request %02d" % (side, index)
    return row, [row["violations"]]


def _totals(rows):
    groups = {}

    def bucket(name, kind, budget):
        key = (name, kind, budget)
        if key not in groups:
            groups[key] = dict(
                row_type="total", group=name, kind=kind, budget=budget,
                requests=0, forked=0, nondivergent=0,
                swap_events=0, swaps_total=0, swaps_mask=0, swaps_old=0,
                collisions=0, direct_strict_hits=0,
                first_fork_mask=0, first_fork_old=0, first_fork_mixed=0,
                first_fork_unknown=0, first_fork_none=0,
                first_fork_new_strict_hits=0,
                first_fork_ranked_strict_hits=0,
                fork_at_first_swap=0, violation_requests=0, violations=0)
        return groups[key]

    for row in rows:
        if row.get("row_type") != "request":
            continue
        budget = row.get("budget")
        kind = row.get("kind", "")
        targets = [bucket("all", "all", ""),
                   bucket("budget=%s" % budget, "all", budget),
                   bucket("budget=%s/%s" % (budget, kind), kind, budget)]
        for total in targets:
            total["requests"] += 1
            total["forked"] += int(row.get("forked") or 0)
            total["nondivergent"] += int(row.get("nondivergent") or 0)
            for key in ("swap_events", "swaps_total", "swaps_mask", "swaps_old",
                        "collisions", "direct_strict_hits",
                        "first_fork_new_strict_hits",
                        "first_fork_ranked_strict_hits"):
                total[key] += int(row.get(key) or 0)
            victim = row.get("first_fork_victim", "")
            if victim in ("mask", "old", "mixed", "unknown", "none"):
                total["first_fork_" + victim] += 1
            first_fork = row.get("first_fork")
            first_swap = row.get("first_swap")
            if (first_fork not in ("", None) and first_swap not in ("", None)
                    and first_fork == first_swap):
                total["fork_at_first_swap"] += 1
            if row.get("violations"):
                total["violation_requests"] += 1
                total["violations"] += int(row.get("_violation_count") or 0)
    return [groups[key] for key in sorted(
        groups, key=lambda key: (str(key[0]), str(key[1]), str(key[2])))]


def _require_arms(new, archived, budgets, mode="placebo"):
    by_budget = {}
    for arm in new:
        if arm.budget in budgets:
            by_budget.setdefault(arm.budget, set()).add(arm.kind)
    required = ({"oldhalf_a", "oldhalf_b"} if mode == "oldhalf"
                else {"random_a", "random_b"})
    for budget in budgets:
        if budget not in archived or not archived[budget]:
            raise ValueError("No %s reference at Q%d"
                             % ("old-half" if mode == "oldhalf"
                                else "archived ranked", budget))
        missing = required - by_budget.get(budget, set())
        if missing:
            raise ValueError("Missing %s arm(s) at Q%d: %s"
                             % ("old-half" if mode == "oldhalf" else "placebo",
                                budget, ", ".join(sorted(missing))))


def audit_paths(ranked_paths, run_paths, budgets=(4, 8), allow_missing=False,
                manifest=None):
    budgets = tuple(sorted(set(int(budget) for budget in budgets)))
    expected = _load_manifest(manifest)
    strict = not allow_missing
    if strict and expected is None:
        raise ValueError(
            "strict audit requires benchmarks/frozen30/manifest.json "
            "or an explicit --manifest")
    archived, new = discover(ranked_paths, run_paths, budgets)
    _require_arms(new, archived, budgets)
    records = []
    for arm in sorted(new, key=lambda item: (item.budget, item.kind, item.policy)):
        reference = select_reference(archived, arm)
        if rank_tag(reference.selection) != "edit4":
            raise ValueError(
                "Placebo arm %s requires an archived edit4 reference"
                % arm.policy)
        if not reference.files:
            raise ValueError("Archived reference %s has no requests"
                             % reference.policy)
        reference_errors = _environment_violations(
            reference, require_completion=strict)
        arm_errors = ["reference %s: %s" % (reference.policy, message)
                      for message in reference_errors]
        arm_errors.extend(_environment_violations(
            arm, require_completion=strict))
        if strict and not arm.files:
            arm_errors.append("arm has no request files")
        if arm_errors:
            row = _base_row(arm)
            row.update(row_type="arm", index="", status="arm")
            row["violations"] = "; ".join(arm_errors)
            records.append((row, arm_errors))
        indices = set(arm.files) | set(reference.files)
        if expected is not None:
            indices |= set(expected)
        for index in sorted(indices):
            label = "%s[%02d]" % (arm.policy, index)
            sample = expected.get(index) if expected is not None else None
            if index not in arm.files and index not in reference.files:
                if not allow_missing:
                    records.append(_missing_row(
                        arm, index, "placebo and archived"))
                continue
            if index not in arm.files:
                if not allow_missing:
                    records.append(_missing_row(arm, index, "placebo"))
                continue
            if index not in reference.files:
                if not allow_missing:
                    records.append(_missing_row(arm, index, "archived"))
                continue
            try:
                current = _read_json(arm.files[index])
                base = _read_json(reference.files[index])
            except (OSError, ValueError) as error:
                message = "%s: cannot read request (%s)" % (
                    label, error.__class__.__name__)
                row = _base_row(arm)
                row.update(index=index, status="error")
                row["violations"] = message
                records.append((row, [message]))
                continue
            row, violations = audit_request(
                current, base, arm, reference, index, label, sample=sample)
            records.append((row, violations))

    rows = []
    messages = 0
    for row, violations in records:
        row["violations"] = "; ".join(violations)
        row["_violation_count"] = len(violations)
        messages += len(violations)
        rows.append(row)
    totals = _totals(rows)
    request_rows = [row for row in rows if row.get("row_type") == "request"]
    summary = dict(
        requests=len(request_rows),
        forks=sum(int(row.get("forked") or 0) for row in request_rows),
        nondivergent=sum(int(row.get("nondivergent") or 0)
                         for row in request_rows),
        missing=sum(1 for row in request_rows
                    if row.get("status") == "missing"),
        arm_rows=sum(1 for row in rows if row.get("row_type") == "arm"),
        violations=messages,
        violation_requests=sum(1 for row in rows if row.get("violations")),
        strict=strict,
        budgets=list(budgets),
        arms=sorted({(row.get("budget"), row.get("kind"))
                     for row in request_rows}),
    )
    return rows, totals, summary


def write_csv(path, rows, totals):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, restval="",
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        for row in totals:
            writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked", type=Path, action="append", required=True,
                        help="historical ranked raw run directory; repeatable")
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="candidate-placebo raw run directory; repeatable")
    parser.add_argument("--out", type=Path, required=True,
                        help="CSV output path")
    parser.add_argument("--budget", type=int, action="append",
                        help="nominal Q to audit; defaults to 4 and 8")
    parser.add_argument("--allow-missing", action="store_true",
                        help="skip missing requests instead of reporting them")
    parser.add_argument("--manifest", type=Path,
                        help="frozen30 manifest path (default: repository manifest)")
    args = parser.parse_args(argv)
    budgets = tuple(args.budget) if args.budget else (4, 8)
    try:
        rows, totals, summary = audit_paths(
            args.ranked, args.run, budgets=budgets,
            allow_missing=args.allow_missing, manifest=args.manifest)
    except (OSError, ValueError) as error:
        print("exchange placebo audit: %s" % error, file=sys.stderr)
        return 2
    write_csv(args.out, rows, totals)
    summary["out"] = str(args.out)
    print(json.dumps(summary, sort_keys=True))
    return 1 if summary["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
