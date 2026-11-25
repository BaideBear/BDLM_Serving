"""Matched MASK trajectories in the historical first-seat experiment (CPU only).

An observation is the after-state of a real denoise, including the anchor at 0.
Waits are row sums: 0..3, or 4 for no hit. Stable means through observation 3
only. Padding repeats state, not work. Base output IDs are references, not truth.
"""
import argparse
from collections import Counter
from pathlib import Path
from statistics import mean

from tools.causal_seat_report import MODES, inspect, require
from tools.conditional_seat import (DATASETS, DEFAULT_MANIFEST, DEFAULT_RUNS, STRATA,
                                    check_manifest, policy_name, read_json)
from tools.t2t_results import table

H = 4
KINDS = ("common_mask", "mask_victim")
PAIRS = (("rank", "base"), ("random", "base"), ("rank", "random"))
AGREE = tuple(f"agree_t{i}" for i in range(H))
COUNTS = ("carried_row_obs", "anchor_admitted", "branch_missing", "endpoint_missing",
          "endpoint_different", *AGREE, "first_hit", "first_wait_sum", "stable_wait_sum")


def strip_watch(raw):
    return dict(raw, events=[{k: v for k, v in e.items() if k != "beneficiary"}
                             for e in raw["events"]])


def historical_replay(new, old):
    keys = ("model", "policy", "index", "dataset", "task_id", "seed", "status", "output_ids",
            "output_tokens", "generation_forwards", "generation_query_rows",
            "generation_physical_padded_rows", "forwards_by_phase", "q_by_phase")
    require(all(new[k] == old[k] for k in keys) and
            new.get("physical_events") == old.get("physical_events"), "Historical replay mismatch")
    require(len(new["events"]) == len(old["events"]), "Historical event count mismatch")
    for a, b in zip(new["events"], old["events"], strict=True):
        # Older anchors lack the newer victim acceptance diagnostics. Every
        # archived field, including the actual hashes, must still be reproduced.
        require(("intervention" in a) == ("intervention" in b) and
                all(a.get(k) == v for k, v in b.items() if k != "intervention") and
                all(a["intervention"].get(k) == v for k, v in b.get("intervention", {}).items()),
                "Historical event mismatch")


def window(raw, at, rows, offsets, seen):
    states = []
    previous = [None] * len(rows)  # every watched row was MASK before the anchor
    events = raw["events"]
    for i, e in enumerate(events):
        watched = (at >= 0 and i >= at and e["block"] == events[at]["block"] and
                   e["stage"] in ("bootstrap", "dense", "selected", "verify"))
        require(watched == ("beneficiary" in e), f"Missing or unexpected beneficiary at event {i}")
        if not watched:
            continue
        b = e["beneficiary"]
        require(isinstance(b, dict) and b.get("rows") == rows and
                all(type(r) is int for r in b["rows"]), "Wrong watched MASK rows/order")
        ids, off = b.get("ids"), b.get("output_offsets")
        require(isinstance(ids, list) and len(ids) == len(rows) and all(
            t is None or (type(t) is int and t >= 0) for t in ids), "Invalid beneficiary IDs")
        require(isinstance(off, list) and len(off) == len(rows) and all(type(o) is int for o in off)
                and len({o - r for o, r in zip(off, rows)}) <= 1, "Misaligned output offsets")
        if offsets is None:
            offsets = off
        require(off == offsets, "Output offsets changed across observations or arms")
        selected = (set(e["positions"]) if "positions" in e else
                    {pos - 32 * e["block"] for pos in e["absolute_positions"]})
        require(all(row in selected or before == after for row, before, after in
                    zip(rows, previous, ids, strict=True)), "Off-Q beneficiary changed state")
        previous = ids
        # Equal post-state (even after divergence/reconvergence) implies equal
        # accepted IDs on the same watched rows. Check beyond H as well.
        for key in ("accepted_sha256", "trajectory_sha256"):
            digest = e.get(key)
            require(isinstance(digest, str) and len(digest) == 64, "Missing denoise state evidence")
            require(seen.setdefault((key, digest), ids) == ids, "Same state, different watched IDs")
        if len(states) < H:
            states.append(ids)
    return offsets, states


def measure(raw):
    """Return arm and paired counts for one complete base/rank/random triplet."""
    at, plans = inspect([strip_watch(r) for r in raw])
    p = plans[0] if plans else {}
    stratum = p.get("stratum", "none")
    require(not plans or stratum in ("slack", "mask"), "Unknown anchor stratum")
    rows = sorted(set(p["base"]) & set(p["masks"])) if plans else []
    ref = raw[0]["output_ids"]
    records, pairs, seen, offsets = [], [], {}, None
    for arm, r in zip(MODES, raw, strict=True):
        offsets, states = window(r, at, rows, offsets, seen)
        if plans:
            plan = r["events"][at]["intervention"]
            require(sum(t is not None for t in states[0]) == plan["after_admissions"],
                    "Observed admissions disagree with anchor execution")
        carried = H - len(states) if states else 0
        states += states[-1:] * carried
        counts = {kind: Counter() for kind in KINDS}
        for j, (row, off) in enumerate(zip(rows, offsets or (), strict=True)):
            c = counts["mask_victim" if row == p["victim"] else "common_mask"]
            c["watched"] += 1
            c["carried_row_obs"] += carried
            c["anchor_admitted"] += states[0][j] is not None
            returned = 0 <= off < len(r["output_ids"])
            c["branch_missing"] += not returned  # All watched offsets, including outside base.
            if not 0 <= off < len(ref):
                continue
            c["eligible"] += 1
            c["endpoint_missing"] += not returned
            c["endpoint_different"] += returned and r["output_ids"][off] != ref[off]
            hits = [s[j] == ref[off] for s in states]
            for key, hit in zip(AGREE, hits, strict=True):
                c[key] += hit
            c["first_hit"] += any(hits)
            c["first_wait_sum"] += hits.index(True) if any(hits) else H
            c["stable_wait_sum"] += max((t + 1 for t, hit in enumerate(hits) if not hit), default=0)
        for kind, c in counts.items():
            records.append(dict(model=r["model"], index=r["index"], dataset=r["dataset"],
                                stratum=stratum, row_kind=kind, arm=arm, watched=c["watched"],
                                eligible=c["eligible"], **{k: c[k] for k in COUNTS}))
    by_arm = {(r["arm"], r["row_kind"]): r for r in records}
    for left, right in PAIRS:
        for kind in KINDS:
            a, b = by_arm[left, kind], by_arm[right, kind]
            pairs.append({**{k: a[k] for k in ("model", "index", "dataset", "stratum", "row_kind")},
                          "pair": f"{left}_vs_{right}", "watched": a["watched"], "eligible": a["eligible"],
                          **{f"delta_{k}": a[k] - b[k] for k in COUNTS}})
    return records, pairs


def summarize(records, pairs):
    """Add counts per arm or signed counts per pair; never pool requests across arms."""
    result = []
    for data, key, prefix in ((records, "arm", ""), (pairs, "pair", "delta_")):
        for name in dict.fromkeys(r[key] for r in data):
            for kind in KINDS:
                group = [r for r in data if r[key] == name and r["row_kind"] == kind]
                for dataset in DATASETS:
                    for stratum in STRATA:
                        sub = [r for r in group if dataset in ("all", r["dataset"]) and
                               stratum in ("all", r["stratum"])]
                        waits = [r[prefix + "stable_wait_sum"] / r["eligible"] for r in sub if r["eligible"]]
                        result.append(dict(model=group[0]["model"], dataset=dataset, stratum=stratum,
                            row_kind=kind, measure=key, comparison=name, requests=len(sub),
                            anchors=sum(r["stratum"] != "none" for r in sub),
                            eligible_requests=sum(r["eligible"] > 0 for r in sub),
                            watched=sum(r["watched"] for r in sub), eligible=sum(r["eligible"] for r in sub),
                            request_mean_stable_wait=mean(waits) if waits else "",
                            requests_faster=sum(w < 0 for w in waits) if key == "pair" else "",
                            requests_slower=sum(w > 0 for w in waits) if key == "pair" else "",
                            **{k: sum(r[prefix + k] for r in sub) for k in COUNTS}))
    return result


def main(run, out):
    meta = read_json(run / "run.json")
    model = meta["model"]
    require(model in DEFAULT_RUNS, "Expected llada21 Q4 or gemma Q32")
    history, _, q = DEFAULT_RUNS[model]
    policies = [policy_name(q, mode) for mode in MODES]
    for policy in policies:
        require(meta["policies"].count(policy) == 1 and read_json(run / f"complete-{policy}.json") ==
                dict(policy=policy, requests=30), f"Missing or incomplete {policy}")
    identity, records, pairs = [], [], []
    for i in range(30):
        raw = []
        for policy in policies:
            filename = f"{i:02d}-{policy}.json"
            r = read_json(run / filename)
            require((r["model"], r["policy"], r["index"]) == (model, policy, i), "Wrong request file")
            historical_replay(strip_watch(r), read_json(history / filename))
            raw.append(r)
        identity.append({k: raw[0][k] for k in ("index", "dataset", "task_id", "seed")})
        points, contrasts = measure(raw)
        records.extend(points)
        pairs.extend(contrasts)
    check_manifest(DEFAULT_MANIFEST, [(model, identity)])
    summary = summarize(records, pairs)
    for data, prefix in ((records, ""), (pairs, "delta_"), (summary, "")):
        for r in data:
            for metric in ("first_wait", "stable_wait"):
                k = prefix + metric
                r[k + "_mean"] = r[k + "_sum"] / r["eligible"] if r["eligible"] else ""
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("requests", records), ("paired", pairs), ("summary", summary)):
        table(out / f"{model}-beneficiary-{name}.csv", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("run", "out"):
        parser.add_argument("--" + arg, required=True, type=Path)
    args = parser.parse_args()
    try:
        main(args.run, args.out)
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
