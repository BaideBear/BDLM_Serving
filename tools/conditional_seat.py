"""Exact request splicing for conditional single-seat interventions.

The conditional policy looks only at the first legal one-seat candidate.  It
swaps in the measured candidate when the victim is a MASK (``mask``) or an old
backfill row (``backfill``/``slack``); otherwise it abandons the request for
good.  Because the three measured arms share the pre-intervention state and the
base arm exactly reproduces the independent noedit run, the conditional
trajectory is exactly the rank arm on matching requests and exactly the base
arm elsewhere.  This module performs that splice, verifies its premises, and
publishes per-request and aggregate tables.

This is a post-hoc reconstruction of measured trajectories, not a rerun and not
independent evidence.  The conditional policy itself was never executed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from tools.causal_seat_report import first_difference, inspect
from tools.t2t_results import table

ARMS = ("base", "rank", "random")
CONDITION_STRATUM = {"all": None, "backfill": "slack", "mask": "mask"}
DATASETS = ("all", "gsm8k", "math500", "humaneval")
STRATA = ("all", "slack", "mask", "none")
POLICIES = {
    "noedit": (None, None),
    "uncond_rank": ("rank", "all"),
    "cond_backfill_rank": ("rank", "backfill"),
    "cond_mask_rank": ("rank", "mask"),
    "uncond_random": ("random", "all"),
    "cond_backfill_random": ("random", "backfill"),
    "cond_mask_random": ("random", "mask"),
}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks/frozen30/manifest.json"
DEFAULT_PUBLISHED = ROOT / "experiments/t2t/placement/one-seat"
DEFAULT_OUT = ROOT / "experiments/t2t/placement/one-seat/conditional"
DEFAULT_RUNS = {
    "llada21": (Path("/data1/project_data/gx-erqw-20261123/runtime/t2t-native-1125/one-seat-v1-llada21"),
                Path("/data1/project_data/gx-erqw-20261123/runtime/t2t-native-1125/hard-v1-llada21"), 4),
    "gemma": (Path("/data1/project_data/gx-erqw-20261123/runtime/t2t-native-1125/one-seat-v2-gemma"),
              Path("/data1/project_data/gx-erqw-20261123/runtime/t2t-native-1125/hard-v1-gemma"), 32),
}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def policy_name(q, arm):
    return f"t2t_q{q}_f0_m{q}_hard_once_{arm}"


def noedit_name(q):
    return f"t2t_q{q}_f0_m{q}_hard_noedit"


def load_arm(run, q, arm):
    run = Path(run)
    policy = policy_name(q, arm)
    complete = read_json(run / f"complete-{policy}.json")
    if complete != dict(policy=policy, requests=30):
        raise ValueError(f"Incomplete arm {policy}: {complete}")
    rows = [read_json(run / f"{i:02d}-{policy}.json") for i in range(30)]
    for i, row in enumerate(rows):
        if row["index"] != i or row["policy"] != policy or row["status"] != "success":
            raise ValueError(f"Bad request {i} in {policy}")
    raw_marks = read_json(run / f"marks-{policy}.json")
    if set(raw_marks) != {str(i) for i in range(30)}:
        raise ValueError(f"Bad marks for {policy}")
    return rows, {str(i): int(bool(raw_marks[str(i)])) for i in range(30)}


def load_control(control, q):
    control = Path(control)
    policy = noedit_name(q)
    rows = [read_json(control / f"{i:02d}-{policy}.json") for i in range(30)]
    for i, row in enumerate(rows):
        if row["index"] != i or row["policy"] != policy or row["status"] != "success":
            raise ValueError(f"Bad noedit request {i}")
    raw_marks = read_json(control / f"marks-{policy}.json")
    if set(raw_marks) != {str(i) for i in range(30)}:
        raise ValueError("Bad noedit marks")
    return rows, {str(i): int(bool(raw_marks[str(i)])) for i in range(30)}


def check_manifest(manifest_path, row_lists):
    manifest = read_json(manifest_path)
    samples = manifest["samples"]
    expected = {(s["index"], s["dataset"], s["task_id"], s["seed"]) for s in samples}
    if len(samples) != 30 or len(expected) != 30:
        raise ValueError("Manifest is not the fixed 30-request list")
    counts = Counter(s["dataset"] for s in samples)
    if counts != Counter(gsm8k=10, math500=10, humaneval=10):
        raise ValueError(f"Manifest dataset counts are not 10/10/10: {counts}")
    for label, rows in row_lists:
        actual = {(r["index"], r["dataset"], r["task_id"], r["seed"]) for r in rows}
        if len(rows) != 30 or actual != expected:
            raise ValueError(f"{label} does not match the fixed 30-request manifest")
        if [r["index"] for r in rows] != list(range(30)):
            raise ValueError(f"{label} is not ordered 0..29")
    return dict(requests=30, datasets=dict(counts))


def verify_base_noedit(base_rows, noedit_rows, base_marks, noedit_marks):
    if base_marks != noedit_marks:
        raise ValueError("Base and noedit marks differ")
    for i, (b, n) in enumerate(zip(base_rows, noedit_rows, strict=True)):
        for key in b.keys() | n.keys():
            if key in ("policy", "events"):
                continue
            if b.get(key) != n.get(key):
                raise ValueError(f"Base/noedit field {key} differs on request {i}")
        if len(b["events"]) != len(n["events"]):
            raise ValueError(f"Base/noedit event count differs on request {i}")
        for j, (x, y) in enumerate(zip(b["events"], n["events"], strict=True)):
            # Base may add diagnostic keys (intervention, trajectory_sha256,
            # accepted_sha256).  Every noedit event field must be preserved.
            for key, value in y.items():
                if x.get(key) != value:
                    raise ValueError(f"Base/noedit event {j} field {key} differs on request {i}")
    return len(base_rows)


def metrics(row, correct):
    n, f, q = row["output_tokens"], row["generation_forwards"], row["generation_query_rows"]
    out = dict(correct=int(correct), N=n, F=f, Q=q,
               physical_Q=row["generation_physical_padded_rows"],
               tpf=n / f, token_efficiency=n / q, q_per_f=q / f)
    if abs(out["tpf"] - row["tpf"]) > 1e-9 or abs(out["token_efficiency"] - row["token_efficiency"]) > 1e-9:
        raise ValueError(f"Metric identity broken on request {row['index']}")
    return out


def select_rows(base_rows, arm_rows, strata, anchors, condition):
    out = []
    for i, (base, arm, stratum, anchor) in enumerate(
            zip(base_rows, arm_rows, strata, anchors, strict=True)):
        target = CONDITION_STRATUM[condition]
        hit = stratum != "none" and (target is None or stratum == target)
        out.append(dict(index=i, dataset=base["dataset"], task_id=base["task_id"],
                        seed=base["seed"], anchor=anchor, stratum=stratum,
                        intervened=int(hit), source_arm="arm" if hit else "base",
                        row=arm if hit else base))
    return out


def build_policy(name, family, condition, base_rows, arm_rows, noedit_rows,
                 strata, anchors, marks):
    if name == "noedit":
        selected = [dict(index=i, dataset=noedit_rows[i]["dataset"],
                         task_id=noedit_rows[i]["task_id"], seed=noedit_rows[i]["seed"],
                         anchor=anchors[i], stratum=strata[i], intervened=0,
                         source_arm="noedit", row=noedit_rows[i])
                        for i in range(len(noedit_rows))]
    else:
        selected = select_rows(base_rows, arm_rows, strata, anchors, condition)
    for x in selected:
        if x["source_arm"] == "base":
            correct = marks["base"][str(x["index"])]
        elif x["source_arm"] == "arm":
            correct = marks[family][str(x["index"])]
        else:
            correct = marks["noedit"][str(x["index"])]
        x.update(metrics(x["row"], correct))
    return selected


def aggregate(selected, reference, dataset="all", stratum="all"):
    rows = [x for x in selected
            if (dataset == "all" or x["dataset"] == dataset)
            and (stratum == "all" or x["stratum"] == stratum)]
    if not rows:
        return dict(requests=0, anchors=0, intervened=0, correct=0, failed=0,
                    truncated=0, wins=0, losses=0, delta_correct=0, N=0, F=0, Q=0,
                    physical_Q=0, tpf=0.0, token_efficiency=0.0, q_per_f=0.0,
                    noedit_correct=0, noedit_failed=0, noedit_truncated=0,
                    noedit_N=0, noedit_F=0, noedit_Q=0, noedit_physical_Q=0,
                    noedit_tpf=0.0, noedit_token_efficiency=0.0, noedit_q_per_f=0.0,
                    delta_N=0, delta_F=0, delta_Q=0, delta_physical_Q=0,
                    delta_tpf=0.0, delta_token_efficiency=0.0, delta_q_per_f=0.0)
    refs = {x["index"]: x for x in reference}
    ref_rows = [refs[x["index"]] for x in rows]
    n = sum(x["N"] for x in rows)
    f = sum(x["F"] for x in rows)
    q = sum(x["Q"] for x in rows)
    rn = sum(x["N"] for x in ref_rows)
    rf = sum(x["F"] for x in ref_rows)
    rq = sum(x["Q"] for x in ref_rows)
    correct = sum(x["correct"] for x in rows)
    ref_correct = sum(x["correct"] for x in ref_rows)
    return dict(
        requests=len(rows),
        anchors=sum(x["stratum"] != "none" for x in rows),
        intervened=sum(x["intervened"] for x in rows),
        correct=correct,
        failed=sum(x["row"]["status"] != "success" for x in rows),
        truncated=sum(bool(x["row"]["truncated"]) for x in rows),
        wins=sum(x["correct"] > refs[x["index"]]["correct"] for x in rows),
        losses=sum(x["correct"] < refs[x["index"]]["correct"] for x in rows),
        delta_correct=correct - ref_correct,
        N=n, F=f, Q=q, physical_Q=sum(x["physical_Q"] for x in rows),
        tpf=mean(x["N"] / x["F"] for x in rows),
        token_efficiency=n / q, q_per_f=q / f,
        noedit_correct=ref_correct,
        noedit_failed=sum(x["row"]["status"] != "success" for x in ref_rows),
        noedit_truncated=sum(bool(x["row"]["truncated"]) for x in ref_rows),
        noedit_N=rn, noedit_F=rf, noedit_Q=rq,
        noedit_physical_Q=sum(x["physical_Q"] for x in ref_rows),
        noedit_tpf=mean(x["N"] / x["F"] for x in ref_rows),
        noedit_token_efficiency=rn / rq, noedit_q_per_f=rq / rf,
        delta_N=n - rn, delta_F=f - rf, delta_Q=q - rq,
        delta_physical_Q=sum(x["physical_Q"] for x in rows)
                         - sum(x["physical_Q"] for x in ref_rows),
        delta_tpf=mean(x["N"] / x["F"] for x in rows)
                  - mean(x["N"] / x["F"] for x in ref_rows),
        delta_token_efficiency=n / q - rn / rq,
        delta_q_per_f=q / f - rq / rf,
    )


def request_row(model, name, family, condition, x, ref):
    return dict(
        model=model, policy=name, family=family or "", condition=condition or "",
        index=x["index"], dataset=x["dataset"], task_id=x["task_id"], seed=x["seed"],
        anchor=x["anchor"], stratum=x["stratum"], intervened=x["intervened"],
        source_arm=x["source_arm"], correct=x["correct"],
        noedit_correct=ref["correct"], delta_correct=x["correct"] - ref["correct"],
        win=int(x["correct"] > ref["correct"]), loss=int(x["correct"] < ref["correct"]),
        N=x["N"], F=x["F"], Q=x["Q"], physical_Q=x["physical_Q"],
        tpf=x["tpf"], token_efficiency=x["token_efficiency"], q_per_f=x["q_per_f"],
        noedit_N=ref["N"], noedit_F=ref["F"], noedit_Q=ref["Q"],
        noedit_physical_Q=ref["physical_Q"], noedit_tpf=ref["tpf"],
        noedit_token_efficiency=ref["token_efficiency"],
        noedit_q_per_f=ref["q_per_f"],
        delta_N=x["N"] - ref["N"], delta_F=x["F"] - ref["F"],
        delta_Q=x["Q"] - ref["Q"],
        delta_physical_Q=x["physical_Q"] - ref["physical_Q"],
        delta_tpf=x["tpf"] - ref["tpf"],
        delta_token_efficiency=x["token_efficiency"] - ref["token_efficiency"],
        delta_q_per_f=x["q_per_f"] - ref["q_per_f"],
    )


def audit_swap(plans):
    """Re-check the one-seat swap contract from the recorded plans.

    Returns (rank_failures, random_failures, stratum_failures).  This is an
    independent re-check of what ``inspect`` enforces, so the published checks
    table can report a count rather than a bare assertion.
    """
    rank, random, stratum = [], [], []
    if not plans:
        return rank, random, stratum
    p0, p1, p2 = plans
    base_set, masks, fresh = set(p0["base"]), set(p0["masks"]), set(p0["fresh"])
    for label, plan, key in (("rank", p1, "candidate"),
                             ("random", p2, "random_candidate")):
        picked = set(plan["executed"])
        victim, extra = p0["victim"], p0[key]
        if base_set - picked != {victim} or picked - base_set != {extra}:
            (rank if label == "rank" else random).append("not exactly one swap")
        if extra in fresh | masks or victim in fresh:
            (rank if label == "rank" else random).append("fresh/mask contract")
        if (victim in masks) != (p0["stratum"] == "mask"):
            stratum.append(label)
    return rank, random, stratum


def anchor_row(model, i, base, rank, random, anchor, plans, marks):
    if not plans:
        return dict(
            model=model, index=i, dataset=base["dataset"], task_id=base["task_id"],
            seed=base["seed"], anchor=-1, stratum="none", clock="", victim="",
            candidate="", random_candidate="", base_positions="", targeted_positions="",
            random_positions="", fresh="", masks="", pre_sha256="", post_rng_sha256="",
            base_after_sha256="", rank_after_sha256="", random_after_sha256="",
            base_after_admissions="", rank_after_admissions="", random_after_admissions="",
            base_executed="", rank_executed="", random_executed="",
            base_correct=marks["base"][str(i)], rank_correct=marks["rank"][str(i)],
            random_correct=marks["random"][str(i)], rank_output_changed=0,
            random_output_changed=0, rank_delta_F=0, rank_delta_Q=0, rank_delta_N=0,
            rank_delta_TPF=0.0, random_delta_F=0, random_delta_Q=0, random_delta_N=0,
            random_delta_TPF=0.0, rank_first_canvas_diff=-1,
            rank_first_accepted_diff=-1, random_first_canvas_diff=-1,
            random_first_accepted_diff=-1)
    p0, p1, p2 = plans
    start = max(anchor, 0)

    def delta(a, b):
        return dict(F=a["generation_forwards"] - b["generation_forwards"],
                    Q=a["generation_query_rows"] - b["generation_query_rows"],
                    N=a["output_tokens"] - b["output_tokens"],
                    TPF=a["tpf"] - b["tpf"])

    def diffs(a, b):
        return (first_difference(a["events"][start:], b["events"][start:], "trajectory_sha256"),
                first_difference(a["events"][start:], b["events"][start:], "accepted_sha256"))

    rd, nd = delta(rank, base), delta(random, base)
    (rc, ra), (nc, na) = diffs(rank, base), diffs(random, base)
    return dict(
        model=model, index=i, dataset=base["dataset"], task_id=base["task_id"],
        seed=base["seed"], anchor=anchor, stratum=p0["stratum"], clock=p0["clock"],
        victim=p0["victim"], candidate=p0["candidate"],
        random_candidate=p0["random_candidate"],
        base_positions=json.dumps(p0["base"]), targeted_positions=json.dumps(p0["targeted"]),
        random_positions=json.dumps(p0["random"]), fresh=json.dumps(p0["fresh"]),
        masks=json.dumps(p0["masks"]), pre_sha256=p0["pre_sha256"],
        post_rng_sha256=p0["post_rng_sha256"],
        base_after_sha256=p0["after_sha256"], rank_after_sha256=p1["after_sha256"],
        random_after_sha256=p2["after_sha256"],
        base_after_admissions=p0["after_admissions"],
        rank_after_admissions=p1["after_admissions"],
        random_after_admissions=p2["after_admissions"],
        base_executed=json.dumps(p0["executed"]), rank_executed=json.dumps(p1["executed"]),
        random_executed=json.dumps(p2["executed"]),
        base_correct=marks["base"][str(i)], rank_correct=marks["rank"][str(i)],
        random_correct=marks["random"][str(i)],
        rank_output_changed=int(rank["output_ids"] != base["output_ids"]),
        random_output_changed=int(random["output_ids"] != base["output_ids"]),
        rank_delta_F=rd["F"], rank_delta_Q=rd["Q"], rank_delta_N=rd["N"],
        rank_delta_TPF=rd["TPF"], random_delta_F=nd["F"], random_delta_Q=nd["Q"],
        random_delta_N=nd["N"], random_delta_TPF=nd["TPF"],
        rank_first_canvas_diff=rc, rank_first_accepted_diff=ra,
        random_first_canvas_diff=nc, random_first_accepted_diff=na,
    )


def causal_rows(base_rows, rank_rows, random_rows, marks, anchors, plans):
    out = {}
    for pair, a, b, arm_a, arm_b in (
            ("rank_vs_base", rank_rows, base_rows, "rank", "base"),
            ("random_vs_base", random_rows, base_rows, "random", "base"),
            ("rank_vs_random", rank_rows, random_rows, "rank", "random")):
        rows = []
        for i in range(30):
            x, y = a[i], b[i]
            start = max(anchors[i], 0)
            rows.append(dict(
                index=i, dataset=x["dataset"], task_id=x["task_id"], pair=pair,
                anchor=anchors[i], stratum=plans[i][0]["stratum"] if plans[i] else "none",
                first_canvas_diff=first_difference(
                    x["events"][start:], y["events"][start:], "trajectory_sha256"),
                first_accepted_diff=first_difference(
                    x["events"][start:], y["events"][start:], "accepted_sha256"),
                output_changed=int(x["output_ids"] != y["output_ids"]),
                correct=marks[arm_a][str(i)], ref_correct=marks[arm_b][str(i)],
                delta_F=x["generation_forwards"] - y["generation_forwards"],
                delta_Q=x["generation_query_rows"] - y["generation_query_rows"],
                delta_N=x["output_tokens"] - y["output_tokens"],
                delta_TPF=x["tpf"] - y["tpf"]))
        out[pair] = rows
    return out


def check_published_questions(published, model, q, manifest_path):
    """Check that the published one-seat tables use the same fixed questions."""
    manifest = read_json(manifest_path)
    expected = {(s["index"], s["dataset"], s["task_id"]) for s in manifest["samples"]}
    bad = []
    for name, policy in ((f"{model}-causal-requests.csv", None),
                         (f"{model}-online-requests.csv", policy_name(q, "rank"))):
        rows = read_csv(Path(published) / name)
        if policy:
            rows = [r for r in rows if r["policy"] == policy]
        actual = {(int(r["index"]), r["dataset"], r["task_id"]) for r in rows}
        if actual != expected:
            bad.append(name)
    return bad


def compare_summary(ours, published):
    bad = []
    for key in ("requests", "correct", "failed", "truncated", "N", "F", "Q", "physical_Q"):
        if int(published[key]) != int(ours[key]):
            bad.append(f"{key}: {ours[key]} != {published[key]}")
    for key in ("tpf", "token_efficiency", "q_per_f"):
        if abs(float(published[key]) - float(ours[key])) > 1e-9:
            bad.append(f"{key}: {ours[key]} != {published[key]}")
    return bad


def compare_causal(ours, published):
    by_key = {(r["pair"], int(r["index"])): r for r in published}
    bad = []
    for pair, rows in ours.items():
        for r in rows:
            p = by_key.get((pair, r["index"]))
            if p is None:
                bad.append(f"missing {pair}/{r['index']}")
                continue
            for key in ("anchor", "first_canvas_diff", "first_accepted_diff",
                        "output_changed", "correct", "ref_correct"):
                if int(p[key]) != int(r[key]):
                    bad.append(f"{pair}/{r['index']} {key}: {r[key]} != {p[key]}")
            for key in ("delta_F", "delta_Q", "delta_N"):
                if int(p[key]) != int(r[key]):
                    bad.append(f"{pair}/{r['index']} {key}: {r[key]} != {p[key]}")
            if abs(float(p["delta_TPF"]) - r["delta_TPF"]) > 1e-9:
                bad.append(f"{pair}/{r['index']} delta_TPF: {r['delta_TPF']} != {p['delta_TPF']}")
    return bad


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llada-run", type=Path, default=DEFAULT_RUNS["llada21"][0])
    parser.add_argument("--llada-control", type=Path, default=DEFAULT_RUNS["llada21"][1])
    parser.add_argument("--gemma-run", type=Path, default=DEFAULT_RUNS["gemma"][0])
    parser.add_argument("--gemma-control", type=Path, default=DEFAULT_RUNS["gemma"][1])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--published", type=Path, default=DEFAULT_PUBLISHED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-published", action="store_true")
    args = parser.parse_args(argv)

    models = {
        "llada21": dict(run=args.llada_run, control=args.llada_control, q=4),
        "gemma": dict(run=args.gemma_run, control=args.gemma_control, q=32),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    request_rows, summary_rows, vs_rows, anchor_rows, stratum_rows, checks = [], [], [], [], [], []

    def add_check(model, name, scope, passed, detail):
        checks.append(dict(model=model, check=name, scope=scope,
                           passed=int(bool(passed)), detail=detail))

    for model, cfg in models.items():
        run, control, q = Path(cfg["run"]), Path(cfg["control"]), cfg["q"]
        meta = read_json(run / "run.json")
        if meta["model"] != model:
            raise ValueError(f"{run} is {meta['model']}, expected {model}")
        arms = {arm: load_arm(run, q, arm) for arm in ARMS}
        noedit_rows, noedit_marks = load_control(control, q)
        base_rows, rank_rows, random_rows = (arms[a][0] for a in ARMS)
        marks = {"base": arms["base"][1], "rank": arms["rank"][1],
                 "random": arms["random"][1], "noedit": noedit_marks}

        check_manifest(args.manifest, [
            ("base", base_rows), ("rank", rank_rows),
            ("random", random_rows), ("noedit", noedit_rows)])
        add_check(model, "fixed_question_list", "manifest+base/rank/random/noedit",
                  True, f"30/30; 10 gsm8k + 10 math500 + 10 humaneval; "
                        f"index/task_id/seed match")
        verified = verify_base_noedit(base_rows, noedit_rows, marks["base"], noedit_marks)
        add_check(model, "base_reproduces_noedit", "30 requests", verified == 30,
                  f"{verified}/30; all non-policy fields and noedit event fields equal")

        anchors, plans, strata = [], [], []
        rank_bad, random_bad, stratum_bad = [], [], []
        for i in range(30):
            at, pl = inspect([base_rows[i], rank_rows[i], random_rows[i]])
            anchors.append(at)
            plans.append(pl)
            strata.append(pl[0]["stratum"] if pl else "none")
            r, n, s = audit_swap(pl)
            rank_bad += [f"request {i}: {x}" for x in r]
            random_bad += [f"request {i}: {x}" for x in n]
            stratum_bad += [f"request {i}: {x}" for x in s]
        anchored = sum(a >= 0 for a in anchors)
        add_check(model, "same_state_anchor_plan", "30 requests", True,
                  f"{anchored} anchors + {30 - anchored} none; "
                  f"prefixes/anchor/plan/pre_sha256/post_rng_sha256 equal")
        add_check(model, "rank_swap_exact", "anchored requests", not rank_bad,
                  f"{anchored}/{anchored}; exactly one victim->candidate swap"
                  if not rank_bad else "; ".join(rank_bad[:3]))
        add_check(model, "random_swap_exact", "anchored requests", not random_bad,
                  f"{anchored}/{anchored}; exactly one victim->random_candidate swap"
                  if not random_bad else "; ".join(random_bad[:3]))
        add_check(model, "stratum_consistent", "anchored requests", not stratum_bad,
                  "victim in masks <=> stratum=mask on all anchored requests"
                  if not stratum_bad else "; ".join(stratum_bad[:3]))
        add_check(model, "metric_identities", "30 requests x 4 arms", True,
                  "tpf=N/F and token_efficiency=N/Q on every selected row")

        policy_rows = {}
        noedit_selected = None
        for name, (family, condition) in POLICIES.items():
            if name == "noedit":
                selected = build_policy(name, None, None, base_rows, base_rows,
                                        noedit_rows, strata, anchors, marks)
                noedit_selected = selected
            else:
                arm_rows = rank_rows if family == "rank" else random_rows
                selected = build_policy(name, family, condition, base_rows,
                                        arm_rows, noedit_rows, strata, anchors, marks)
            policy_rows[name] = selected

        exact = True
        for name, (family, condition) in POLICIES.items():
            for x in policy_rows[name]:
                if name == "noedit":
                    exact &= x["row"] is noedit_rows[x["index"]]
                elif x["intervened"]:
                    arm = rank_rows if family == "rank" else random_rows
                    exact &= x["row"] is arm[x["index"]]
                else:
                    exact &= x["row"] is base_rows[x["index"]]
                if name != "noedit":
                    target = CONDITION_STRATUM[condition]
                    expected = x["stratum"] != "none" and (
                        target is None or x["stratum"] == target)
                    exact &= x["intervened"] == int(expected)
        add_check(model, "conditional_splice_exact", "all policies", exact,
                  "every policy row is the measured base/rank/random row object; "
                  "match=>rank; no-match=>base; none=>base")

        ref_by_index = {x["index"]: x for x in noedit_selected}
        for name, (family, condition) in POLICIES.items():
            for x in policy_rows[name]:
                request_rows.append(request_row(model, name, family, condition,
                                                x, ref_by_index[x["index"]]))
            for dataset in DATASETS:
                summary_rows.append(dict(model=model, policy=name, family=family or "",
                                         condition=condition or "", dataset=dataset,
                                         **aggregate(policy_rows[name], noedit_selected, dataset)))
            for stratum in STRATA:
                stratum_rows.append(dict(model=model, policy=name, stratum=stratum,
                                         **aggregate(policy_rows[name], noedit_selected,
                                                     "all", stratum)))
        for row in summary_rows:
            if row["model"] != model or row["dataset"] != "all":
                continue
            vs = dict(row)
            for key in ("correct", "N", "F", "Q", "tpf", "token_efficiency", "q_per_f"):
                ref = row[f"noedit_{key}"]
                vs[f"delta_{key}_pct"] = 100.0 * row[f"delta_{key}"] / ref if ref else 0.0
            vs_rows.append(vs)
        for i in range(30):
            anchor_rows.append(anchor_row(model, i, base_rows[i], rank_rows[i],
                                          random_rows[i], anchors[i], plans[i], marks))

        if not args.skip_published:
            published = Path(args.published)
            if not published.exists():
                add_check(model, "published_tables", "one-seat", False,
                          f"missing {published}")
            else:
                pub_summary = read_csv(published / f"{model}-online-summary.csv")
                pub_causal = read_csv(published / f"{model}-causal-requests.csv")
                bad_questions = check_published_questions(
                    published, model, q, args.manifest)
                add_check(model, "published_question_list", "manifest+published",
                          not bad_questions,
                          "OK" if not bad_questions else "; ".join(bad_questions))
                for name, arm in (("uncond_rank", "rank"), ("uncond_random", "random")):
                    policy = policy_name(q, arm)
                    ours = next(r for r in summary_rows
                                if r["model"] == model and r["policy"] == name
                                and r["dataset"] == "all")
                    pub = next(r for r in pub_summary
                               if r["policy"] == policy and r["dataset"] == "all")
                    bad = compare_summary(ours, pub)
                    add_check(model, f"{name}_matches_published", "all", not bad,
                              "OK" if not bad else "; ".join(bad[:5]))
                causal = causal_rows(base_rows, rank_rows, random_rows, marks, anchors, plans)
                bad = compare_causal(causal, pub_causal)
                add_check(model, "causal_requests_match_published", "30x3 pairs", not bad,
                          "OK" if not bad else "; ".join(bad[:5]))

    table(out / "conditional-checks.csv", checks)
    if not all(c["passed"] for c in checks):
        failed = [c for c in checks if not c["passed"]]
        raise ValueError("Conditional splice checks failed: "
                         + "; ".join(f"{c['model']}/{c['check']}: {c['detail']}" for c in failed))
    table(out / "conditional-requests.csv", request_rows)
    table(out / "conditional-summary.csv", summary_rows)
    table(out / "conditional-vs-noedit.csv", vs_rows)
    table(out / "conditional-anchor.csv", anchor_rows)
    table(out / "conditional-by-stratum.csv", stratum_rows)
    print(f"wrote {out}")
    for row in vs_rows:
        print(f"{row['model']:8s} {row['policy']:22s} correct={row['correct']:2d} "
              f"dC={row['delta_correct']:+d} w/l={row['wins']}/{row['losses']} "
              f"F={row['F']:5d} Q={row['Q']:6d} TPF={row['tpf']:.4f} "
              f"tokEff={row['token_efficiency']:.5f}")


if __name__ == "__main__":
    main()
