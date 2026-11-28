"""Second-pass analysis of a sealed six-plus-84 LLaDA allowance confirmation."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from tools.agreement_fixed90_comparison import SIX, cohorts, dataset_cost, paired_intervals, require


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load_stage(path, expected):
    audit = json.loads((path/"offline-audit.json").read_bytes())
    require(audit["status"] == "pass", "missing successful physical audit")
    for name in ("receipts.ndjson", "registration.json"):
        require(audit["sha256"][name] == digest(path/name), "audited stage digest mismatch")
    rows = {}
    for line in (path/"receipts.ndjson").read_text().splitlines():
        entry = json.loads(line)
        i = entry["index"]
        require(type(i) is int and i in expected and i not in rows, "foreign or duplicate request")
        vals = {a: entry["arms"][a] for a in ("stock", "apply", "capped")}
        for v in vals.values():
            require(type(v["correct"]) is bool and all(type(v[k]) is int for k in ("N","F","Q"))
                    and v["N"] >= 0 and 0 < v["F"] <= v["Q"]
                    and (not v["failure"] or (not v["correct"] and v["N"] == 0)),
                    "invalid physical bill or excluded failure")
        rows[i] = vals
    require(set(rows) == expected, "incomplete audited stage")
    return rows, audit


def run(six, rest, manifest, anchor):
    ds = json.loads(manifest.read_bytes())
    require(digest(manifest) == "2e67d12f5045d486d1676a4b9cd35718b0791ad1b2a3c81fdec86a27bd773e62",
            "different frozen task registry")
    x, ax = load_stage(six, SIX)
    y, ay = load_stage(rest, set(range(90))-SIX)
    reg = json.loads((rest/"registration.json").read_bytes())
    require(reg["six_audit"] == digest(six/"offline-audit.json") and set(reg["fresh"]) == set(y)
            and set(reg["reused"]) == set(x) and ay["fresh"] == 84 and ay["reused"] == 6,
            "does not confirm this exact preregistered six")
    require(ay["sha256"]["complete.json"] == digest(rest/"complete.json"), "unsealed summary")
    rows = x | y
    old = json.loads(anchor.read_bytes())
    historical = cohorts({i: {a: v[a] for a in ("stock", "apply")} for i, v in rows.items()})
    for group in historical:
        for a in ("stock", "apply"):
            for k in ("N", "F", "Q", "correct"):
                require(old["cohorts"][group][a][k] == historical[group][a][k], "earlier control drift")
    out = {"source_sha256": {"anchors": ax["sha256"]["receipts.ndjson"], "remaining": ay["sha256"]["receipts.ndjson"],
                             "full_audit": digest(rest/"offline-audit.json"), "archived_v1": digest(anchor)},
           "model": "llada21", "fixed_q": 16, "compared_to": {}}
    for k in ("stock","apply"):
        pair = {i: {"stock": v[k], "apply": v["capped"]} for i,v in rows.items()}
        summary = {}
        for group, v in cohorts(pair).items():
            summary[group] = {"n": v["requests"], "control": v["stock"], "allowance": v["apply"],
                              **{f: v[f] for f in ("d_tpf_from_N","d_tpf_from_F","gained","lost",
                                                    "F_less","F_equal","F_more")}}
        ci = paired_intervals(pair,ds["samples"])
        ci["delta"] = f"allowance_minus_{k}"
        groups = dataset_cost(pair,ds["samples"])["datasets"]
        for g in groups.values():
            g["control_correct"] = g.pop("stock_correct")
            g["allowance_correct"] = g.pop("apply_correct")
        out["compared_to"][k] = {"cohorts": summary, "intervals": ci, "by_dataset": groups}
    return out


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("six","rest","manifest","anchor"):
        p.add_argument(arg,type=Path)
    args=p.parse_args()
    print(json.dumps(run(**vars(args)),indent=2,sort_keys=True))
