"""CPU-only authenticity audit for hard-refresh T2T arms.

Reads 30-request arms from one or more run directories, verifies the
hard-policy invariants recorded by the LLaDA and Gemma backends, and writes one
compact CSV row per (model, policy). No torch, checkpoint, GPU, scoring, or
network access is used.

The two backends expose different event schemas. The audit keeps those
differences visible instead of inventing common fields:

* llada21 events carry absolute positions, per-step MASK counts, and the
  top-level hard fields ``hard_clock``, ``mandatory_fresh``,
  ``reserved_extra_edit_positions``, ``reserved_strict_edit_positions``,
  ``overflow`` and ``hard_strict_edits``. Fresh positions are not logged, so
  per-position fresh verification is impossible. The audit checks the count
  chain with the net MASK change plus native edits, including mask writebacks,
  and a predecessor intersection lower bound.
* gemma events carry block-local positions and explicit ``available_fresh`` /
  ``selected_fresh`` / ``revoked`` / ``resolved`` counts, so the audit checks
  the count chain and the resolved chain exactly.
* llada21 has commit forwards but no reencode stage; gemma has reencode encoder
  forwards but no separate commit forward.
* An arm with fewer than 30 request files is reported as unverified, never as
  a completed arm. A budget larger than the block width is a saturation label;
  the row reports ``saturation=1`` and the actual selected Q, not the label.
* ``selected_max_q`` is a maximum across requests, not a sum.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARD = ("hard_frontier", "hard_random", "hard_noedit")
POLICY = re.compile(r"^t2t_q(\d+)_f(\d+)_m(\d+)_([a-z_]+)$")

COLUMNS = (
    "model", "policy", "selection", "budget", "fresh", "mask", "edit", "requests",
    "prefill_f", "prefill_q",
    "bootstrap_f", "bootstrap_q",
    "selected_steps", "selected_q", "selected_max_q",
    "selected_over_steps", "selected_over_rows", "selected_overflow_rows",
    "verify_f", "verify_q",
    "commit_f", "commit_q",
    "reencode_f", "reencode_q",
    "dense_f", "dense_q",
    "reserved_seats", "reserved_hits", "reserved_hit_rate",
    "strict_edits", "fresh_edits", "nonfresh_edits",
    "selected_fresh_required", "selected_fresh_hit", "fresh_exact",
    "fresh_evidence", "fresh_chain_ok", "fresh_verified",
    "saturation", "ok", "violations", "first_violation",
)


def parse_policy(text):
    match = POLICY.match(text)
    if match is None:
        raise ValueError("Invalid T2T policy: %s" % text)
    budget, fresh, mask, selection = match.groups()
    budget, fresh, mask = int(budget), int(fresh), int(mask)
    return dict(policy=text, budget=budget, fresh=fresh, mask=mask,
                edit=budget - fresh - mask, selection=selection)


def load_manifest(path):
    if path is None:
        return {}
    data = json.loads(Path(path).read_text())
    return {int(sample["index"]): sample for sample in data["samples"]}


def read_policies(folder):
    meta = json.loads((folder / "run.json").read_text())
    model, width = meta["model"], int(meta["block"])
    policies = list(meta.get("policies", []))
    arms = list(meta.get("arms", []))
    if not policies:
        policies = ["t2t_q%d_f%d_m%d_%s" % (a["budget"], a["fresh"], a["mask"], a["selection"])
                    for a in arms]
    if len(policies) != len(arms) and arms:
        raise ValueError("%s: policies/arms length mismatch" % folder)
    out = []
    for index, policy in enumerate(policies):
        arm = parse_policy(policy)
        if arms:
            raw = arms[index]
            for key in ("budget", "fresh", "mask", "selection"):
                if raw.get(key) != arm[key]:
                    raise ValueError("%s: run.json arm disagrees for %s" % (folder, policy))
        out.append((policy, arm))
    return model, width, out


def arm_violations(arm):
    violations = []
    if arm["selection"] not in HARD:
        violations.append("not a hard policy")
    if arm["fresh"] != 0:
        violations.append("hard policy requires fresh=0")
    if arm["edit"] < 0:
        violations.append("negative edit seats")
    if arm["selection"] == "hard_noedit" and arm["edit"] != 0:
        violations.append("hard_noedit requires edit=0")
    if arm["selection"] != "hard_noedit" and arm["edit"] == 0:
        violations.append("hard_frontier/hard_random require edit>0")
    return violations


def _bad(violations, message):
    violations.append(message)


def _unique(values):
    return len(values) == len(set(values))


def normalize_llada(result, arm):
    events = []
    for raw in result.get("events", []):
        events.append(dict(
            raw=raw,
            stage=raw.get("stage", ""),
            block=raw.get("block"),
            rows=int(raw.get("physical_q", raw.get("actual_q", raw.get("q_tokens", 0)))),
            positions=list(raw.get("absolute_positions", [])),
            budget=int(raw.get("q_budget", arm["budget"])),
            mandatory=raw.get("mandatory_fresh"),
            fresh_selected=None,
            available_fresh=None,
            selected_fresh=None,
            reserved_extra=list(raw.get("reserved_extra_edit_positions", [])),
            reserved_strict=list(raw.get("reserved_strict_edit_positions", [])),
            strict=raw.get("hard_strict_edits"),
            fresh_edits=None,
            overflow=raw.get("overflow"),
            accepted=raw.get("accepted"),
            edited=raw.get("edited"),
            admitted=None,
            masked_before=raw.get("masked_before"),
            masked_after=raw.get("masked_after"),
            clock=raw.get("hard_clock"),
            phase=raw.get("phase"),
            absolute=True,
        ))
    return events


def normalize_gemma(result, arm, width):
    events = []
    for raw in result.get("events", []):
        events.append(dict(
            raw=raw,
            stage=raw.get("stage", ""),
            block=raw.get("block"),
            rows=int(raw.get("q", 0)),
            positions=list(raw.get("positions", [])),
            budget=arm["budget"],
            mandatory=raw.get("mandatory_fresh"),
            fresh_selected=raw.get("fresh_selected"),
            available_fresh=raw.get("available_fresh"),
            selected_fresh=raw.get("selected_fresh"),
            reserved_extra=list(raw.get("reserved_extra_edit_positions", [])),
            reserved_strict=list(raw.get("reserved_strict_edit_positions", [])),
            strict=raw.get("hard_strict_edits"),
            fresh_edits=raw.get("edited_in_fresh"),
            overflow=raw.get("overflow"),
            accepted=None,
            edited=raw.get("edited"),
            admitted=raw.get("admitted"),
            revoked=raw.get("revoked"),
            resolved=raw.get("resolved"),
            selected_mask=raw.get("selected_mask"),
            masked_before=None,
            masked_after=None,
            clock=raw.get("hard_clock"),
            phase="denoise",
            absolute=False,
        ))
    physical = result.get("physical_events", [])
    encoders = [int(rows) for phase, rows in physical if phase == "encoder"]
    decoders = [int(rows) for phase, rows in physical if phase == "decoder"]
    return events, encoders, decoders, width


def audit_reserved(event, violations, counters):
    positions = set(event["positions"])
    extra = event["reserved_extra"]
    strict = event["reserved_strict"]
    if not _unique(extra):
        _bad(violations, "reserved extra positions are not unique")
    if not set(extra) <= positions:
        _bad(violations, "reserved extra positions are not a Q subset")
    if not _unique(strict):
        _bad(violations, "reserved strict positions are not unique")
    if not set(strict) <= set(extra):
        _bad(violations, "reserved strict positions are not reserved extras")
    if not set(strict) <= positions:
        _bad(violations, "reserved strict positions are not a Q subset")
    if event["strict"] is not None and len(strict) > int(event["strict"]):
        _bad(violations, "reserved strict hits exceed total strict edits")
    counters["reserved_seats"] += len(extra)
    counters["reserved_hits"] += len(strict)


def audit_request(result, arm, model, width):
    violations = []
    counters = Counter()
    if model == "llada21":
        events = normalize_llada(result, arm)
        encoders = decoders = None
    elif model == "gemma":
        events, encoders, decoders, width = normalize_gemma(result, arm, width)
    else:
        raise ValueError("Unsupported model: %s" % model)

    gen_f = gen_q = 0
    work_rows = []
    edited_total = 0
    budget = int(arm["budget"])
    previous_clock = None
    previous_clock_block = None

    for event in events:
        stage, rows = event["stage"], event["rows"]
        if rows <= 0:
            _bad(violations, "non-positive Q")
        if len(event["positions"]) != rows:
            _bad(violations, "position count != Q")
        if not _unique(event["positions"]):
            _bad(violations, "duplicate Q positions")
        if model == "gemma":
            if any(not (0 <= int(pos) < width) for pos in event["positions"]):
                _bad(violations, "position outside canvas")
        elif stage != "prefill" and event["block"] is not None:
            lo = int(event["block"]) * width
            if any(not (lo <= int(pos) < lo + width) for pos in event["positions"]):
                _bad(violations, "absolute position outside block")
        if int(event["budget"]) != budget:
            _bad(violations, "q_budget != arm budget")
        if event["block"] is None:
            _bad(violations, "missing block id")
        if event["mandatory"] is None:
            _bad(violations, "missing mandatory fresh count")
        if event["strict"] is None:
            _bad(violations, "missing hard_strict_edits")
        if event["overflow"] is None:
            _bad(violations, "missing overflow")
        if event["clock"] is None:
            _bad(violations, "missing hard_clock")
        elif previous_clock is not None:
            same_block = event["block"] == previous_clock_block
            if (model == "llada21" or same_block) \
                    and int(event["clock"]) <= int(previous_clock):
                _bad(violations, "hard_clock not increasing")
        if event["clock"] is not None:
            previous_clock = int(event["clock"])
            previous_clock_block = event["block"]
        if model == "gemma":
            for key, label in (("available_fresh", "available_fresh"),
                               ("selected_fresh", "selected_fresh"),
                               ("revoked", "revoked"),
                               ("resolved", "resolved"),
                               ("selected_mask", "selected_mask"),
                               ("fresh_edits", "edited_in_fresh")):
                if event[key] is None:
                    _bad(violations, "missing %s" % label)
        if model == "llada21" and event["raw"].get("actual_q") is not None:
            if int(event["raw"]["actual_q"]) != rows:
                _bad(violations, "actual_q != physical_q")

        counters[stage + "_f"] += 1
        counters[stage + "_q"] += rows
        if stage != "prefill":
            gen_f += 1
            gen_q += rows
            work_rows.append(rows)
        if event["edited"] is not None:
            edited_total += int(event["edited"])
        if event["strict"] is not None:
            if int(event["strict"]) > rows:
                _bad(violations, "strict edits exceed Q")
            counters["strict_edits"] += int(event["strict"])
        if model == "gemma" and event["fresh_edits"] is not None:
            counters["fresh_edits"] += int(event["fresh_edits"])

        if model == "llada21":
            if event["mandatory"] is not None and int(event["mandatory"]) > rows:
                _bad(violations, "mandatory fresh exceeds Q")
            if event["edited"] is not None and int(event["edited"]) > rows:
                _bad(violations, "edited exceeds Q")
            if event["strict"] is not None and event["edited"] is not None \
                    and int(event["strict"]) > int(event["edited"]):
                _bad(violations, "hard_strict_edits exceeds edited")
            if event["accepted"] is not None and event["masked_before"] is not None:
                if int(event["accepted"]) > int(event["masked_before"]):
                    _bad(violations, "accepted exceeds MASK before")
            if event["masked_before"] is not None and int(event["masked_before"]) > width:
                _bad(violations, "masked_before exceeds block")
            if event["masked_after"] is not None and int(event["masked_after"]) > width:
                _bad(violations, "masked_after exceeds block")
            if None not in (event["masked_before"], event["masked_after"], event["edited"]) \
                    and int(event["masked_after"]) > int(event["masked_before"]) + int(event["edited"]):
                _bad(violations, "masked_after exceeds MASK before + edits")
            if event["overflow"] is not None \
                    and int(event["overflow"]) != max(0, rows - int(event["budget"])):
                _bad(violations, "overflow != max(0, Q-budget)")

        if model == "gemma":
            if event["revoked"] is not None and int(event["revoked"]) < 0:
                _bad(violations, "negative revoked")
            if event["admitted"] is not None and int(event["admitted"]) < 0:
                _bad(violations, "negative admitted")
            if event["edited"] is not None and int(event["edited"]) < 0:
                _bad(violations, "negative edited")
            if None not in (event["admitted"], event["edited"], event["revoked"]):
                if int(event["admitted"]) + int(event["edited"]) + int(event["revoked"]) > rows:
                    _bad(violations, "admitted+edited+revoked exceeds Q")
            if None not in (event["admitted"], event["selected_mask"]) \
                    and int(event["admitted"]) > int(event["selected_mask"]):
                _bad(violations, "admitted exceeds selected MASK")
            if None not in (event["edited"], event["revoked"], event["selected_mask"]) \
                    and int(event["edited"]) + int(event["revoked"]) > rows - int(event["selected_mask"]):
                _bad(violations, "resolved changes exceed non-MASK Q")
            if None not in (event["selected_mask"], event["selected_fresh"]) \
                    and int(event["selected_mask"]) + int(event["selected_fresh"]) > rows:
                _bad(violations, "selected MASK+fresh exceeds Q")
            if None not in (event["fresh_edits"], event["edited"]) \
                    and int(event["fresh_edits"]) > int(event["edited"]):
                _bad(violations, "edited_in_fresh exceeds edited")
            if event["resolved"] is not None \
                    and not (0 <= int(event["resolved"]) <= width):
                _bad(violations, "resolved outside block")
            if event["mandatory"] is not None and int(event["mandatory"]) > rows:
                _bad(violations, "mandatory fresh exceeds Q")
            if event["strict"] is not None and event["edited"] is not None \
                    and int(event["strict"]) != int(event["edited"]):
                _bad(violations, "hard_strict_edits != edited")
            if stage == "selected" and event["overflow"] is not None \
                    and int(event["overflow"]) != max(0, rows - int(event["budget"])):
                _bad(violations, "overflow != max(0, Q-budget)")

        if stage == "selected":
            counters["selected_steps"] += 1
            counters["selected_max_q"] = max(counters["selected_max_q"], rows)
            if rows > budget:
                counters["selected_over_steps"] += 1
                counters["selected_over_rows"] += rows
                counters["selected_overflow_rows"] += rows - budget
                if event["mandatory"] != rows:
                    _bad(violations, "Q>budget is not pure fresh overflow")
                if event["reserved_extra"]:
                    _bad(violations, "Q>budget kept reserved extra edits")
                if event["overflow"] != rows - budget:
                    _bad(violations, "overflow count mismatch at Q>budget")
            else:
                if event["overflow"] != 0:
                    _bad(violations, "overflow on in-budget selected step")
            if model == "gemma":
                if event["selected_fresh"] != event["available_fresh"]:
                    _bad(violations, "fresh not fully selected")
                if event["mandatory"] != event["available_fresh"]:
                    _bad(violations, "mandatory fresh != available fresh")
                if event["mandatory"] is not None:
                    counters["selected_fresh_required"] += int(event["mandatory"])
                if event["selected_fresh"] is not None:
                    counters["selected_fresh_hit"] += int(event["selected_fresh"])
            else:
                if event["mandatory"] is not None:
                    counters["selected_fresh_required"] += int(event["mandatory"])
            audit_reserved(event, violations, counters)
        else:
            if event["reserved_extra"]:
                _bad(violations, "non-selected reserved extra edits")
            if event["reserved_strict"]:
                _bad(violations, "non-selected reserved strict edits")
            if model == "gemma":
                if event["available_fresh"] is not None and event["selected_fresh"] != event["available_fresh"]:
                    _bad(violations, "non-selected fresh not fully selected")
                if event["mandatory"] is not None and event["mandatory"] != event["available_fresh"]:
                    _bad(violations, "non-selected mandatory fresh mismatch")
                if event["overflow"] not in (0, None):
                    _bad(violations, "non-selected overflow")

    previous = None
    for event in events:
        if previous is not None:
            if previous["block"] == event["block"]:
                if model == "gemma":
                    expected = int(previous["edited"] or 0) + int(previous["admitted"] or 0)
                    if event["available_fresh"] != expected:
                        _bad(violations, "fresh chain mismatch (available)")
                    if event["mandatory"] != expected:
                        _bad(violations, "fresh chain mismatch (mandatory)")
                    if None not in (previous["resolved"], event["resolved"],
                                    event["admitted"], event["revoked"]):
                        expected_resolved = (int(previous["resolved"])
                                             + int(event["admitted"])
                                             - int(event["revoked"]))
                        if int(event["resolved"]) != expected_resolved:
                            _bad(violations, "resolved chain mismatch")
                elif all(previous[key] is not None for key in
                         ("accepted", "edited", "masked_before", "masked_after")):
                    mask_writes = (int(previous["masked_after"])
                                   - int(previous["masked_before"])
                                   + int(previous["accepted"]))
                    if mask_writes < 0:
                        _bad(violations, "negative MASK writes")
                    expected = (int(previous["edited"])
                                + int(previous["masked_before"])
                                - int(previous["masked_after"]))
                    if expected < 0:
                        _bad(violations, "negative fresh chain")
                    if event["mandatory"] != expected:
                        _bad(violations, "fresh chain mismatch")
                if event["stage"] == "selected" and event["mandatory"] is not None:
                    overlap = len(set(previous["positions"]) & set(event["positions"]))
                    if overlap < int(event["mandatory"]):
                        _bad(violations, "fresh hit lower bound violated")
            elif event["mandatory"] not in (0, None):
                _bad(violations, "fresh carried across block")
        previous = event

    if model == "gemma":
        if not encoders:
            _bad(violations, "missing prefill encoder")
        else:
            counters["prefill_f"] = 1
            counters["prefill_q"] = encoders[0]
            counters["reencode_f"] = len(encoders) - 1
            counters["reencode_q"] = sum(encoders[1:])
            gen_f += len(encoders) - 1
            gen_q += sum(encoders[1:])
            work_rows.extend(encoders[1:])
        if len(decoders) != len(events):
            _bad(violations, "decoder physical events != trace events")
        counters["nonfresh_edits"] = counters["strict_edits"] - counters["fresh_edits"]
        if counters["nonfresh_edits"] < 0:
            _bad(violations, "fresh edits exceed strict edits")

    if result.get("generation_forwards") != gen_f:
        _bad(violations, "generation_forwards mismatch")
    if result.get("generation_query_rows") != gen_q:
        _bad(violations, "generation_query_rows mismatch")
    if result.get("generation_physical_padded_rows") != gen_q:
        _bad(violations, "generation_physical_padded_rows mismatch")
    if work_rows and result.get("max_q") != max(work_rows):
        _bad(violations, "max_q mismatch")
    if result.get("prefill_rows") is not None and counters.get("prefill_q") != result["prefill_rows"]:
        _bad(violations, "prefill_rows mismatch")
    if result.get("t2t_edits") is not None and edited_total != result["t2t_edits"]:
        _bad(violations, "t2t_edits mismatch")

    return counters, violations


def audit_arm(model, arm, results, width):
    violations = arm_violations(arm)
    counters = Counter()
    selected_max_q = 0
    for result in results:
        request_counts, request_violations = audit_request(result, arm, model, width)
        selected_max_q = max(selected_max_q, int(request_counts.pop("selected_max_q", 0)))
        counters.update(request_counts)
        index = int(result["index"])
        violations.extend("request %02d: %s" % (index, message) for message in request_violations)
    counters["selected_max_q"] = selected_max_q
    if arm["budget"] > width and counters.get("selected_steps") \
            and counters["selected_max_q"] != width:
        violations.append("saturation arm actual Q != block width")
    return counters, violations


def summary_row(model, arm, counters, violations, requests, width=None):
    def number(key):
        return counters.get(key, 0)

    commit_absent = model == "gemma" and counters.get("commit_f", 0) == 0
    reencode_absent = model == "llada21" and counters.get("reencode_f", 0) == 0
    fresh_absent = model == "llada21"
    fresh_bad = any("fresh" in item for item in violations)
    chain_bad = any(("fresh chain" in item or "fresh carried" in item
                     or "fresh hit lower bound" in item or "MASK writes" in item
                     or "resolved chain" in item) for item in violations)
    seats, hits = number("reserved_seats"), number("reserved_hits")
    strict_edits = number("strict_edits")
    fresh_edits = number("fresh_edits")
    fresh_evidence = "none" if fresh_absent else "counts"
    return dict(
        model=model,
        policy=arm["policy"],
        selection=arm["selection"],
        budget=arm["budget"],
        fresh=arm["fresh"],
        mask=arm["mask"],
        edit=arm["edit"],
        requests=requests,
        prefill_f=number("prefill_f"),
        prefill_q=number("prefill_q"),
        bootstrap_f=number("bootstrap_f"),
        bootstrap_q=number("bootstrap_q"),
        selected_steps=number("selected_steps"),
        selected_q=number("selected_q"),
        selected_max_q=number("selected_max_q"),
        selected_over_steps=number("selected_over_steps"),
        selected_over_rows=number("selected_over_rows"),
        selected_overflow_rows=number("selected_overflow_rows"),
        verify_f=number("verify_f"),
        verify_q=number("verify_q"),
        commit_f="" if commit_absent else number("commit_f"),
        commit_q="" if commit_absent else number("commit_q"),
        reencode_f="" if reencode_absent else number("reencode_f"),
        reencode_q="" if reencode_absent else number("reencode_q"),
        dense_f=number("dense_f"),
        dense_q=number("dense_q"),
        reserved_seats=seats,
        reserved_hits=hits,
        reserved_hit_rate=("%.4f" % (hits / seats)) if seats else "",
        strict_edits=strict_edits,
        fresh_edits="" if fresh_absent else fresh_edits,
        nonfresh_edits="" if fresh_absent else strict_edits - fresh_edits,
        selected_fresh_required=number("selected_fresh_required"),
        selected_fresh_hit="" if fresh_absent else number("selected_fresh_hit"),
        fresh_exact=0 if fresh_absent or fresh_bad else 1,
        fresh_evidence=fresh_evidence,
        fresh_chain_ok=0 if chain_bad else 1,
        fresh_verified=0 if violations or fresh_evidence != "positions" else 1,
        saturation=int(width is not None and arm["budget"] > width),
        ok=0 if violations else 1,
        violations=len(violations),
        first_violation=(violations[0][:200] if violations else ""),
    )


def unverified_row(model, arm, width, requests, message):
    """A row for an arm without a complete 30-request evidence set."""
    return dict(
        model=model,
        policy=arm["policy"],
        selection=arm["selection"],
        budget=arm["budget"],
        fresh=arm["fresh"],
        mask=arm["mask"],
        edit=arm["edit"],
        requests=requests,
        prefill_f="", prefill_q="",
        bootstrap_f="", bootstrap_q="",
        selected_steps="", selected_q="", selected_max_q="",
        selected_over_steps="", selected_over_rows="", selected_overflow_rows="",
        verify_f="", verify_q="",
        commit_f="", commit_q="",
        reencode_f="", reencode_q="",
        dense_f="", dense_q="",
        reserved_seats="", reserved_hits="", reserved_hit_rate="",
        strict_edits="", fresh_edits="", nonfresh_edits="",
        selected_fresh_required="", selected_fresh_hit="",
        fresh_exact=0,
        fresh_evidence="none",
        fresh_chain_ok=0,
        fresh_verified=0,
        saturation=int(width is not None and arm["budget"] > width),
        ok=0,
        violations=1,
        first_violation=message[:200],
    )


def read_arm(folder, policy, arm, model, manifest):
    files = sorted(folder.glob("??-%s.json" % policy))
    if len(files) != 30:
        raise ValueError("%s: %s expected 30 requests, found %d" % (folder, policy, len(files)))
    marker = folder / ("complete-%s.json" % policy)
    if marker.exists():
        info = json.loads(marker.read_text())
        if info.get("policy") != policy or info.get("requests") != 30:
            raise ValueError("%s: bad complete marker" % marker)
    results = []
    seen = set()
    for path in files:
        result = json.loads(path.read_text())
        if result.get("model") != model or result.get("policy") != policy:
            raise ValueError("%s: model/policy mismatch" % path)
        index = int(result["index"])
        if index in seen:
            raise ValueError("%s: duplicate request index %d" % (path, index))
        seen.add(index)
        if manifest:
            sample = manifest.get(index)
            if sample is None:
                raise ValueError("%s: request index not in manifest" % path)
            for key in ("index", "dataset", "task_id", "seed"):
                if key in sample and result.get(key) != sample[key]:
                    raise ValueError("%s: sample identity mismatch (%s)" % (path, key))
        results.append(result)
    expected = set(manifest) if manifest else set(range(30))
    if seen != expected:
        raise ValueError("%s: %s incomplete sample set" % (folder, policy))
    results.sort(key=lambda result: int(result["index"]))
    return results


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="Run directory containing run.json and 30 request JSONs; repeatable")
    parser.add_argument("--out", type=Path, required=True, help="Compact summary CSV")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "benchmarks/frozen30/manifest.json")
    parser.add_argument("--allow-violations", action="store_true",
                        help="Write the CSV but exit 0 even when the audit finds violations")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    rows = []
    seen = set()
    for folder in args.run:
        model, width, arms = read_policies(folder)
        for policy, arm in arms:
            if arm["selection"] not in HARD:
                continue
            key = (model, policy)
            if key in seen:
                raise ValueError("duplicate arm: %s %s" % key)
            seen.add(key)
            files = sorted(folder.glob("??-%s.json" % policy))
            if len(files) != 30:
                rows.append(unverified_row(
                    model, arm, width, len(files),
                    "incomplete arm: %d/30 requests" % len(files)))
                continue
            try:
                results = read_arm(folder, policy, arm, model, manifest)
            except ValueError as exc:
                rows.append(unverified_row(model, arm, width, len(files), str(exc)))
                continue
            counters, violations = audit_arm(model, arm, results, width)
            rows.append(summary_row(model, arm, counters, violations,
                                    len(results), width))

    if not rows:
        raise ValueError("No hard arms found")
    write_csv(args.out, rows)
    violations = sum(row["violations"] for row in rows)
    print("hard audit: %d arms, %d requests, %d violations -> %s" % (
        len(rows), sum(row["requests"] for row in rows), violations, args.out))
    return 1 if violations and not args.allow_violations else 0


if __name__ == "__main__":
    sys.exit(main())
