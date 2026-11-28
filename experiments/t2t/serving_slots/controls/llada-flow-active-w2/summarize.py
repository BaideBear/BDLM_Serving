"""Summarize compact receipts only; reject duplicate, unsealed or misbilled pairs."""
import hashlib
import json
from pathlib import Path
from statistics import mean


def digest(data):
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def describe(rows):
    arms = {}
    for a in ("flow_w2", "active_w2"):
        rr = [r[a] for r in rows]
        s = {k: sum(x[k] for x in rr) for k in
             ("N", "F", "Q", "one", "two", "prefill_Q", "correct", "eos")}
        s.update(macro_TPF=mean(x["N"]/x["F"] for x in rr), pooled_TE=s["N"]/s["Q"])
        arms[a] = s
    x,y = arms.values()
    return dict(pairs=len(rows), **arms, ids_equal=sum(r["ids_equal"] for r in rows),
                all_decisions_equal=sum(r["first_decision_fork"] is None for r in rows),
                quality_wins=[r["index"] for r in rows if r["active_w2"]["correct"] and
                              not r["flow_w2"]["correct"]],
                quality_losses=[r["index"] for r in rows if r["flow_w2"]["correct"] and
                                not r["active_w2"]["correct"]],
                Q_saved=x["Q"]-y["Q"], geometry_saved=32*y["one"],
                tick_count_contribution=64*(x["F"]-y["F"]),
                pairs_with_less_Q=sum(r["active_w2"]["Q"]<r["flow_w2"]["Q"] for r in rows),
                F_equal=sum(r["active_w2"]["F"]==r["flow_w2"]["F"] for r in rows))


def summary(d):
    reg = json.loads((d/"registration.json").read_bytes())
    tag = hashlib.sha256((d/"registration.json").read_bytes()).hexdigest()
    rows = [json.loads(l) for l in (d/"compact.ndjson").read_bytes().splitlines()]
    q, done = (json.loads((d/p).read_bytes()) for p in ("qualified.json","complete.json"))
    order = reg["pilot"]+[i for i in range(90) if i not in reg["pilot"]]
    if (len(rows)!=90 or [z["index"] for z in rows]!=order or
        not q["cases"]==order[:6] or not tag==q["reg"]==done["reg"] or
        digest(rows[:6])!=q["pairs_sha256"] or digest(rows)!=done["pairs_sha256"] or
        done["requests"]!=180 or done["pairs"]!=90):
        raise ValueError("frozen pair/pilot identity or seal mismatch")
    for r in rows:
        if not r["fresh_matches_archive"] or r["reg"]!=tag:
            raise ValueError("invalid baseline or registration")
        a,b = r["flow_w2"],r["active_w2"]
        if (any(t["F"]!=t["one"]+t["two"] for t in (a,b)) or
            a["Q"]!=64*a["F"] or b["Q"]!=32*b["one"]+64*b["two"] or
            a["prefill_Q"]!=b["prefill_Q"] or
            r["ids_equal"]!=(a["ids_sha256"]==b["ids_sha256"])):
            raise ValueError("billing or output digest mismatch")
    return {k:describe(v) for k,v in dict(pilot=rows[:6],original30=[r for r in rows if r["index"]<30],
                                          new60=[r for r in rows if r["index"]>=30],frozen90=rows).items()}


if __name__ == "__main__":
    print(json.dumps(summary(Path(__file__).resolve().parent),sort_keys=True,indent=2))
