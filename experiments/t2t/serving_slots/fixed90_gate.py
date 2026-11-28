"""Compare independently audited, fixed-offer request runs; a partial matrix cannot pass."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import re
from statistics import mean

from .fixed_request_comparison import COHORTS, MODELS, collect
from tools.fixed_admission_rows import FAMILY, MANIFEST_SHA, check

MANIFEST = Path(__file__).resolve().parents[3] / "benchmarks/frozen90/manifest.json"
ROOT = Path(__file__).resolve().parent / "controls/fixed-request-comparison"
# An entry pins the audited implementation AND its fixed-from-prompt driver.
# Add other producers only after checking their source, physical bills and prompt parity.
LOCK = {("llada21", 16): (
    "7c584c5426d77fad3eec4aa397c1f58a86b7ea093ccded2574ee8b76fa44c993",
    "0e5e429bf2a51451a72ea431244a61b6be74b9e2d75bf0e7b76f0e32f392d279",
    "aac79bb98f76cfe422079843f1dcc53a9e8c9b5cf56008d43f3ad4a4d3791518",
    "a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692")}


def controls(model, q):
    b = MODELS[model][-2]
    return ["Flow W2"] if q == 2*b else (["IBS capped", "Native", "Flow W1"] if q == b else ["IBS capped"])


def aggregate(rows):
    n = sum(r["N"] for r in rows)
    history = sum(r["Q"] for r in rows)
    return dict(n=len(rows), correct=sum(r["correct"] for r in rows), N=n,
                F=sum(r["F"] for r in rows), decoder_F=sum(r["decoder_F"] for r in rows),
                encoder_F=sum(r["encoder_F"] for r in rows), history_Q=history,
                macro_TPF=mean(r["N"]/r["F"] for r in rows), token_efficiency=n/history,
                peak=max(r["peak"] for r in rows), new_block_Q=sum(r["build_Q"] for r in rows),
                over_offer_Q=sum(r["over"] for r in rows), terminals=dict(Counter(r["terminal"] for r in rows)))


def audited(doc, samples, lock):
    key = doc["model"], doc["q"]
    check(key in lock and doc["family"] == FAMILY and doc["manifest_sha256"] == MANIFEST_SHA
          and doc["native_prompts_verified"] is True and doc["cohort"] == "frozen90"
          and tuple(doc[k] for k in ("executor_sha256","runner_sources_sha256","checkpoint","scorer")) == lock[key],
          f"unqualified fixed candidate identity: {key}")
    check(doc["complete"] is True and doc["verified"] == 90 and not doc["missing"] and not doc["sealed_failures"],
          "incomplete audit; do not pass partial files as results")
    check(set(doc["cases"]) == set(doc["seal_digests"]) == {str(i) for i in range(90)}, "case coverage")
    check(len(set(doc["seal_digests"].values())) == 90 and
          all(re.fullmatch(r"[0-9a-f]{64}", x) for x in doc["seal_digests"].values()), "case seals")
    rows = []
    for i, s in enumerate(samples):
        r = doc["cases"][str(i)]
        check(i == s["index"] and r["task_id"] == s["task_id"] and r["seed"] == s["seed"], "task/seed drift")
        check(type(r["correct"]) is bool and r["terminal"] in ("eos","cap","native_failure")
              and (r["terminal"] != "native_failure" or not r["correct"]), "invalid terminal/scoring")
        check(all(type(r[k]) is int and r[k] >= 0 for k in
                  ("N","F","Q","peak","over","build_Q","decoder_F","encoder_F"))
              and r["decoder_F"] > 0 and r["F"] == r["decoder_F"] + r["encoder_F"] and 0 < r["peak"] <= r["Q"]
              and r["Q"] <= r["decoder_F"] * r["peak"] and r["N"] <= 2048
              and (r["peak"] > key[1]) == bool(r["over"]) and r["over"] <= r["build_Q"] <= r["Q"],
              "invalid or unreconciled work")
        rows.append(r)
    groups = {c:aggregate(g) for c,g in (("original30", rows[:30]), ("new60", rows[30:]), ("frozen90", rows))}
    full = groups["frozen90"]
    check(all(full[k] == doc[k] for k in ("N","correct"))
          and full["F"] == doc["counts"]["F"] and full["history_Q"] == doc["counts"]["Q"]
          and isclose(full["macro_TPF"],doc["TPF"]) and isclose(full["token_efficiency"],doc["TE"]),
          "aggregate differs from case seals")
    return key, groups


def evaluate(docs, *, lock=None, refs=None):
    check(sha256(MANIFEST.read_bytes()).hexdigest() == MANIFEST_SHA, "changed frozen90 manifest")
    samples = json.loads(MANIFEST.read_bytes())["samples"]
    baselines = {(r["model"],r["budget"],r["cohort"],r["method"]):r for r in (collect() if refs is None else refs)}
    values = {}
    for doc in docs:
        key, groups = audited(doc, samples, LOCK if lock is None else lock)
        check(key not in values, f"duplicate candidate tier: {key}")
        values[key] = groups
    result = []
    for model, tiers in MODELS.items():
        for q in tiers:
            required = controls(model,q)
            cell = dict(model=model, q=q, required=required, status="incomplete")
            result.append(cell)
            if (model,q) not in values:
                continue
            cell["cohorts"] = cohorts = values[model,q]
            cell["comparisons"] = comparisons = {}
            enough, metrics, both = True, True, True
            for c, arm in cohorts.items():
                comparisons[c] = {}
                for name in [*required, "IBS nominal", "Live profile"]:
                    ref = baselines.get((model,q,c,name))
                    if not ref or ref["n"] != COHORTS[c]:
                        if name in required: enough = False
                        continue
                    d = comparisons[c][name] = dict(correct=arm["correct"]-ref["correct"],
                        macro_TPF=arm["macro_TPF"]-ref["macro_TPF"],
                        TE_pp=None if ref["token_efficiency"] is None else
                        100*(arm["token_efficiency"]-ref["token_efficiency"]), baseline_source=ref["source"])
                    if name in required:
                        metrics &= d["TE_pp"] is not None
                        if c == "frozen90":
                            both &= min(d["correct"],d["macro_TPF"]) >= 0
            cell["status"] = "ready_to_compare" if enough else "incomplete"
            cell["three_metrics_complete"] = enough and metrics
            cell["meets_performance"] = both if enough else None
            cell["candidate_hard_cap_observed"] = cohorts["frozen90"]["peak"] <= q
    ready = sum(r["status"] == "ready_to_compare" for r in result)
    return dict(family=FAMILY, ready_tiers=ready, total_tiers=len(result), status=(
        "incomplete" if ready != len(result) else "ready_to_compare"),
        meets_performance=(all(r["meets_performance"] for r in result) if ready == len(result) else None),
        three_metrics_complete=all(r.get("three_metrics_complete",False) for r in result), tiers=result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audits", nargs="*", type=Path, help="physically audited, complete frozen90 reports only")
    p.add_argument("--require-ready", action="store_true", help="exit nonzero until all ten comparisons are ready")
    args = p.parse_args()
    files = args.audits or sorted(ROOT.glob("*-full-admission-q*-frozen90.json"))
    matrix = evaluate([json.loads(path.read_bytes()) for path in files])
    print(json.dumps(matrix, indent=2, allow_nan=False))
    if args.require_ready and (matrix["status"] != "ready_to_compare" or not matrix["three_metrics_complete"]):
        raise SystemExit(1)
