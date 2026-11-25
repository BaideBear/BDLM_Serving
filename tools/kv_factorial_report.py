"""Audit Gemma Q32 single-anchor interventions and report frozen30 contrasts.

K/D denotes keep/drop, first for persistent candidate KV, second for the
candidate's bundled raw logits and sampler actions. All four arms use the same
intra-step attention procedure at the anchor; later decoding is free to
diverge. Historical rank is the replay reference for kk. Historical
hard_once_base (neutral, no seat swap) is a separate practical reference,
not dd and not one of these contrasts.

Each contrast is keep minus drop with the other factor fixed. Macro TPF is
mean(N/F); pooled efficiency is sum(N)/sum(Q). All 30 requests are included.
F and Q count all generation work (including closure and re-encodes), not
prompt prefill or wall time. No model outputs or state hashes are published.

In KV contrasts, next_query_steps counts decoder forwards after the anchor
(1 = immediate) up to the first physical query of the candidate on the same
block, including full verify. Steps and phase are blank for never; all three
next-query fields are blank for score contrasts or ineligible requests.
"""

import argparse
from collections import Counter
import csv
from itertools import zip_longest
import json
import math
from pathlib import Path
import re
from statistics import mean


MODEL, WIDTH, Q, COUNT = "gemma", 256, 32, 30
PREFIX = "t2t_q32_f0_m32_hard_once_"
POLICIES = {arm: PREFIX + arm for arm in ("kk", "kd", "dk", "dd")}
RANK = PREFIX + "rank"
MANIFEST = Path(__file__).resolve().parents[1] / "benchmarks/frozen30/manifest.json"
CONTRASTS = (
    ("kv_at_score_keep", "kk", "dk"),
    ("kv_at_score_drop", "kd", "dd"),
    ("score_at_kv_keep", "kk", "kd"),
    ("score_at_kv_drop", "dk", "dd"),
)
IDENTITY = ("index", "dataset", "task_id", "seed")
PHASES = ("bootstrap", "selected", "verify", "dense", "reencode")
ENDPOINTS = {
    "N": "output_tokens", "F": "generation_forwards", "Q": "generation_query_rows",
    "physical_Q": "generation_physical_padded_rows",
}
REPLAY_FIELDS = (
    "status", "output_ids", "output_text", "truncated", "eos", "prefill_rows",
    *ENDPOINTS.values(), "max_q", "forwards_by_phase", "q_by_phase", "q_histogram",
    "tpf", "token_efficiency", "t2t_edits", "physical_events",
)
STATE = ("trajectory_sha256", "accepted_sha256")
SHARED_PLAN = (
    "base", "targeted", "random", "victim", "candidate", "random_candidate",
    "fresh", "masks", "stratum", "clock", "executed", "pre_sha256",
    "pre_row_sha256", "forward_row_sha256", "post_other_kv_sha256", "post_rng_sha256",
)
ANCHOR_HASHES = (
    "pre_sha256", "pre_row_sha256", "forward_row_sha256", "after_row_sha256",
    "post_other_kv_sha256", "post_nonkv_sha256", "post_rng_sha256", "after_sha256",
)


def require(ok, where, message):
    if not ok:
        raise ValueError(f"{where}: {message}")


def read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def fingerprint(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def run_sources(folder, policies):
    meta = read_json(folder / "run.json")
    where = folder.name
    for field, expected in (("model", MODEL), ("block", WIDTH), ("batch_size", 1), ("cap", 2048)):
        require(meta[field] == expected, where, f"unexpected {field}")
    require(read_json(folder / "qualified.json") == dict(model=MODEL, full_width=WIDTH, matched=COUNT),
            where, "missing frozen30 full-width qualification")
    names, arms = meta["policies"], meta["arms"]
    require(len(names) == len(arms) and len(names) == len(set(names)), where, "invalid arm manifest")
    sources = {}
    for name, arm in zip(names, arms):
        if name not in policies:
            continue
        require(arm == dict(budget=Q, fresh=0, mask=Q, selection=name[len("t2t_q32_f0_m32_"):]),
                where, "expected Q32 with no reserved edit seats")
        require(read_json(folder / f"complete-{name}.json") == dict(policy=name, requests=COUNT),
                where, f"incomplete {name}")
        sources[name] = folder
    require(sources, where, "no requested policies in run")
    return sources


def load_scores(path, tasks):
    numeric = ("nominal_q", "fresh", "mask", "edit", "max_q", *ENDPOINTS,
               "correct", "failed", "truncated")
    required = {"model", "policy", "index", "dataset", "task_id", *numeric}
    scores = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames and required <= set(reader.fieldnames)
                and len(set(reader.fieldnames)) == len(reader.fieldnames),
                "scores", "expected aggregated t2t_results requests CSV")
        for line, row in enumerate(reader, 2):
            if row["policy"] not in POLICIES.values():
                continue
            where = f"scores line {line}"
            try:
                index = int(row["index"])
                values = {k: int(row[k]) for k in numeric}
                seed = int(row["seed"]) if "seed" in row else None
            except (ValueError, TypeError):
                raise ValueError(f"{where}: invalid numeric field") from None
            key = row["policy"], index
            require(key not in scores and index in tasks, where, "duplicate or unexpected request")
            require((row["model"], row["dataset"], row["task_id"]) ==
                    (MODEL, tasks[index]["dataset"], tasks[index]["task_id"]),
                    where, "wrong frozen request identity")
            require(seed is None or seed == tasks[index]["seed"], where, "wrong frozen request seed")
            require(tuple(values[k] for k in ("nominal_q", "fresh", "mask", "edit")) == (Q, 0, Q, 0),
                    where, "wrong score arm budget")
            require(values["correct"] in (0, 1) and values["failed"] == 0
                    and values["truncated"] in (0, 1), where, "invalid scored outcome")
            scores[key] = values
    require(len(scores) == COUNT * len(POLICIES), "scores", "expected exactly 30 scores per factorial arm")
    return scores


def physical_work(row, where):
    """Reconcile the decoder trace with the independent forward observer."""
    trace = row["events"]
    physical = row["physical_events"]
    require(physical and physical[0] == ["encoder", row["prefill_rows"]] and row["prefill_rows"] > 0,
            where, "invalid prefill")
    decoders, f, q, histogram = [], Counter(), Counter(), Counter()
    for i, (kind, width) in enumerate(physical[1:], 1):
        require(type(width) is int and width > 0, where, "invalid physical Q")
        if kind == "encoder":
            phase = "reencode"
        else:
            require(kind == "decoder" and len(decoders) < len(trace), where, "unmatched physical event")
            event = trace[len(decoders)]
            phase = event["stage"]
            positions = seat_rows(event["positions"], where, "physical decoder")
            require(phase in PHASES[:-1] and event["q"] == len(positions) == width
                    and (phase == "selected" or width == WIDTH), where, "decoder stage or Q mismatch")
            decoders.append(i)
        f[phase] += 1
        q[phase] += width
        histogram[str(width)] += 1
    require(decoders and len(trace) == len(decoders), where, "unmatched decoder trace")
    n, total_f, total_q, padded = (row[k] for k in ENDPOINTS.values())
    require(all(type(x) is int for x in (n, total_f, total_q, padded)) and n >= 0
            and total_f == sum(f.values()) and total_q == padded == sum(q.values())
            and dict(f) == row["forwards_by_phase"] and dict(q) == row["q_by_phase"]
            and dict(histogram) == row["q_histogram"] and row["max_q"] == max(map(int, histogram)),
            where, "physical work and request/phase counters disagree")
    require(math.isclose(row["tpf"], n / total_f, rel_tol=0, abs_tol=1e-10)
            and math.isclose(row["token_efficiency"], n / total_q, rel_tol=0, abs_tol=1e-10),
            where, "work counters disagree with TPF/efficiency")
    return decoders


def load_request(folder, policy, task, score=None):
    where = f"request {task['index']:02d} {policy}"
    try:
        row = read_json(folder / f"{task['index']:02d}-{policy}.json")
        require(all(row[k] == task[k] for k in IDENTITY) and type(row["seed"]) is int
                and row["model"] == MODEL and row["policy"] == policy, where, "wrong frozen request or seed")
        require(row["status"] == "success" and type(row["truncated"]) is bool and type(row["eos"]) is bool
                and row["truncated"] != row["eos"], where, "unsuccessful or invalid request")
        require(isinstance(row["output_ids"], list) and isinstance(row["output_text"], str)
                and row["output_tokens"] <= len(row["output_ids"]), where, "invalid output")
        decoders = physical_work(row, where)
        require(all(fingerprint(e.get(k)) for e in row["events"] for k in STATE),
                where, "missing trajectory evidence")
        if score is not None:
            require(all(score[k] == row[v] for k, v in ENDPOINTS.items())
                    and score["max_q"] == row["max_q"] and score["truncated"] == row["truncated"],
                    where, "score CSV does not match request work/outcome")
    except KeyError as exc:
        raise ValueError(f"{where}: missing {exc.args[0]}") from None
    return row, decoders


def require_fields(actual, expected, keys, where, label):
    for k in keys:
        require(k in actual and k in expected and actual[k] == expected[k], where, f"{label}: {k}")


def historical_replay(row, history, where):
    """Match the historical deterministic timeline, not diagnostic wall time."""
    require_fields(row, history, REPLAY_FIELDS, where, "historical rank replay")
    require(len(row["events"]) == len(history["events"]), where, "historical rank trace length")
    for event, old in zip(row["events"], history["events"]):
        require_fields(event, old, (k for k in old if k != "intervention"), where,
                       "historical rank event core")
        if "intervention" in old:
            require_fields(event["intervention"], old["intervention"], old["intervention"], where,
                           "historical rank intervention core")


def anchor_at(row, where):
    hits = [i for i, e in enumerate(row["events"]) if "intervention" in e]
    require(len(hits) <= 1, where, "multiple interventions")
    return hits[0] if hits else -1


def seat_rows(rows, where, label):
    require(isinstance(rows, list) and all(type(i) is int and 0 <= i < WIDTH for i in rows)
            and rows == sorted(set(rows)), where, f"invalid {label} rows")
    return set(rows)


def verify_anchor(event, arm, where):
    p = event["intervention"]
    require(all(fingerprint(p.get(k)) for k in ANCHOR_HASHES), where, "missing/invalid anchor fingerprint")
    kv, score = (letter == "k" for letter in arm)
    require(p["persist_kv"] is kv and p["use_score"] is score, where, "factor flags do not match arm")
    expected = p["forward_row_sha256"] if kv else p["pre_row_sha256"]
    require(p["after_row_sha256"] == expected, where, "after_row_sha256 violates KV persistence")
    require(p["after_sha256"] == event["trajectory_sha256"], where, "anchor trajectory fingerprint mismatch")
    base, picked, fresh, masks = (seat_rows(p[k], where, k) for k in ("base", "executed", "fresh", "masks"))
    victim, candidate = p["victim"], p["candidate"]
    require(event["stage"] == "selected" and event["q"] == len(base) == len(picked) == Q
            and event["positions"] == p["executed"] == p["targeted"]
            and event["hard_clock"] == p["clock"], where, "physical Q32 anchor mismatch")
    require(type(victim) is int and type(candidate) is int and base - picked == {victim}
            and picked - base == {candidate} and fresh <= base & picked and candidate not in masks
            and (not base & masks or picked & masks)
            and p["stratum"] == ("mask" if victim in masks else "slack"),
            where, "not a legal targeted one-seat swap")
    require(seat_rows(p["action_rows"], where, "sampler action") ==
            (picked if score else picked - {candidate}), where, "sampler action rows violate score switch")
    return p


def verify_request(rows, history, decoder_indices, correct, index):
    where = f"request {index:02d}"
    ref_anchor = anchor_at(history, f"{where} historical rank")
    for arm, row in rows.items():
        require(anchor_at(row, f"{where} {arm}") == ref_anchor, where, f"{arm} anchor differs from history")
    historical_replay(rows["kk"], history, f"{where} kk")
    if ref_anchor == -1:
        for arm in ("kd", "dk", "dd"):
            historical_replay(rows[arm], history, f"{where} {arm} ineligible")
            require(rows[arm]["events"] == rows["kk"]["events"] and correct[arm] == correct["kk"],
                    where, f"{arm} ineligible arm differs from kk")
        return -1, "none"

    ref_row, ref_event = rows["kk"], rows["kk"]["events"][ref_anchor]
    ref_end = decoder_indices["kk"][ref_anchor] + 1
    plans = {}
    for arm, row in rows.items():
        ctx = f"{where} {arm}"
        event = row["events"][ref_anchor]
        physical_end = decoder_indices[arm][ref_anchor] + 1
        require(row["events"][:ref_anchor] == ref_row["events"][:ref_anchor]
                and row["physical_events"][:physical_end] == ref_row["physical_events"][:ref_end],
                ctx, "pre-anchor trajectory or physical prefix differs")
        require_fields(event, ref_event, ("block", "clock", "hard_clock", "positions", "q"),
                       ctx, "unmatched physical anchor")
        plans[arm] = verify_anchor(event, arm, ctx)
        require_fields(plans[arm], plans["kk"], SHARED_PLAN, ctx, "unmatched anchor plan/state")
    for a, b in (("kk", "dk"), ("kd", "dd")):
        ctx = f"{where} {a}/{b}"
        require_fields(plans[a], plans[b], ("post_nonkv_sha256",), ctx, "same-score anchor")
        require_fields(rows[a]["events"][ref_anchor], rows[b]["events"][ref_anchor],
                       STATE, ctx, "same-score anchor trajectory")
    return ref_anchor, plans["kk"]["stratum"]


def first_difference(a, b, start, field):
    """Zero-based decoder offset from the anchor; blank if none (or ineligible)."""
    if start >= 0:
        def series(events):
            return ((e["block"], e["clock"], e["stage"], e[field]) for e in events[start:])

        for offset, (x, y) in enumerate(zip_longest(series(a), series(b))):
            if x != y:
                return offset
    return ""


def next_candidate_query(events, anchor):
    event = events[anchor]
    candidate = event["intervention"]["candidate"]
    for steps, later in enumerate(events[anchor + 1:], 1):
        if later["block"] != event["block"]:
            break
        if candidate in later["positions"]:
            return steps, later["stage"], "immediate" if steps == 1 else "later"
    return "", "", "never"


def pair_columns(a, b):
    return {**{f"keep_{k}": a[k] for k in a}, **{f"drop_{k}": b[k] for k in a},
            **{f"delta_{k}": a[k] - b[k] for k in a}}


def paired_rows(task, rows, correct, anchor, stratum):
    common = dict(model=MODEL, **task, anchor=anchor, stratum=stratum)
    for name, a, b in CONTRASTS:
        x, y = rows[a], rows[b]
        key = dict(**common, contrast=name, keep_policy=POLICIES[a], drop_policy=POLICIES[b])

        def endpoints(r, mark):
            return dict(**{k: r[v] for k, v in ENDPOINTS.items() if k != "physical_Q"},
                        tpf=r["tpf"], token_efficiency=r["token_efficiency"],
                        correct=mark, truncated=int(r["truncated"]))

        request = dict(
            **key, output_changed=int(x["output_ids"] != y["output_ids"]),
            first_canvas_diff=first_difference(x["events"], y["events"], anchor, STATE[0]),
            first_accepted_diff=first_difference(x["events"], y["events"], anchor, STATE[1]),
            **pair_columns(endpoints(x, correct[a]), endpoints(y, correct[b])),
        )
        for side, row in (("keep", x), ("drop", y)):
            steps, phase, status = (next_candidate_query(row["events"], anchor)
                                    if anchor >= 0 and a[0] != b[0] else ("", "", ""))
            request.update({f"{side}_next_query_steps": steps, f"{side}_next_query_phase": phase,
                            f"{side}_next_query_status": status})
        phases = []
        for phase in PHASES:
            fx, fy = (r["forwards_by_phase"].get(phase, 0) for r in (x, y))
            qx, qy = (r["q_by_phase"].get(phase, 0) for r in (x, y))
            if fx or fy:
                phases.append(dict(**key, phase=phase, **pair_columns(dict(F=fx, Q=qx), dict(F=fy, Q=qy))))
        yield request, phases


def summarize(rows, contrast, a, b):
    import numpy as np
    from tools.budget_index import paired_uncertainty, request_key, stratified_draws
    totals = {side: {k: sum(r[f"{side}_{k}"] for r in rows) for k in ("N", "F", "Q", "correct")}
              for side in ("keep", "drop")}
    for side, t in totals.items():
        t["macro_tpf"] = mean(r[f"{side}_tpf"] for r in rows)
        t["pooled_efficiency"] = t["N"] / t["Q"]
    x, y = (np.array([[r[f"{side}_{k}"] for k in ("correct", "N", "F", "Q")] for r in rows])
            for side in ("keep", "drop"))
    intervals = paired_uncertainty(x, y, stratified_draws([request_key(r) for r in rows]))
    return dict(model=MODEL, contrast=contrast, keep_policy=POLICIES[a], drop_policy=POLICIES[b],
                requests=len(rows), historical_rank_replays=len(rows), anchors=sum(r["anchor"] >= 0 for r in rows),
                output_changed=sum(r["output_changed"] for r in rows),
                wins=sum(r["delta_correct"] > 0 for r in rows), losses=sum(r["delta_correct"] < 0 for r in rows),
                ties=sum(r["delta_correct"] == 0 for r in rows), **pair_columns(totals["keep"], totals["drop"]),
                **{k: v for k, v in intervals.items() if k.endswith(("_hi", "_lo"))})


def write_table(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main(runs, scores_path, history_path, out):
    require(len(runs) == 2 and len(set(p.resolve() for p in runs)) == 2,
            "runs", "pass two distinct --run directories")
    manifest = read_json(MANIFEST)["samples"]
    require([r["index"] for r in manifest] == list(range(COUNT)), "manifest", "expected ordered frozen30")
    tasks = {r["index"]: {k: r[k] for k in IDENTITY} for r in manifest}
    sources = {}
    for folder in runs:
        group = run_sources(folder, POLICIES.values())
        require(not (group.keys() & sources.keys()), "runs", "duplicate factorial policy")
        sources.update(group)
    require(sources.keys() == set(POLICIES.values()), "runs", "expected all four factorial policies")
    run_sources(history_path, (RANK,))
    scores = load_scores(scores_path, tasks)
    requests, phases = [], []
    for i, task in tasks.items():
        history, _ = load_request(history_path, RANK, task)
        rows, indices, correct = {}, {}, {}
        for arm, policy in POLICIES.items():
            score = scores[policy, i]
            rows[arm], indices[arm] = load_request(sources[policy], policy, task, score)
            correct[arm] = score["correct"]
        try:
            anchor, stratum = verify_request(rows, history, indices, correct, i)
        except KeyError as exc:
            raise ValueError(f"request {i:02d}: missing {exc.args[0]}") from None
        for row, work in paired_rows(task, rows, correct, anchor, stratum):
            requests.append(row)
            phases.extend(work)
    summary = [summarize([r for r in requests if r["contrast"] == name], name, a, b)
               for name, a, b in CONTRASTS]
    # Do not publish anything until all input validation has passed.
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("requests", requests), ("phases", phases), ("summary", summary)):
        write_table(out / f"gemma-kv-factorial-{name}.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, action="append", type=Path, help="repeat for two run directories")
    parser.add_argument("--scores", required=True, type=Path, help="aggregated scored requests CSV")
    parser.add_argument("--history", required=True, type=Path, help="original one-seat-v2-gemma run")
    parser.add_argument("--out", required=True, type=Path, help="destination for three sanitized CSVs")
    args = parser.parse_args()
    try:
        main(args.run, args.scores, args.history, args.out)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"kv_factorial_report: {exc}\n")
    print(f"Historical rank trajectory reproduced by kk on all {COUNT} requests.")
