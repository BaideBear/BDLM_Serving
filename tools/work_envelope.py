"""Compare actual work, not just nominal windows, on existing paired runs.

This is an empirical cost envelope: no curve fitting, interpolation, or new
baseline runs. Mixed-width comparisons are labelled as such. Describing one
row count does not equate attention FLOPs, elapsed time or per-request costs.
"""
import argparse
import csv
from fractions import Fraction
from pathlib import Path
from statistics import mean


def load(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def pack(rows):
    result = {}
    for r in rows:
        key = int(r["index"]), r["dataset"], r["task_id"]
        if key in result:
            raise ValueError("duplicate request")
        n, f, q = (int(r[k]) for k in ("N","F","physical_Q"))
        assert q == int(r["Q"]) and q >= f > 0
        result[key] = (n, f, q, int(r["correct"]))
    if len(result) != 30:
        raise ValueError("expected 30 frozen requests")
    return result


def metrics(rows):
    n, f, q, correct = map(sum, zip(*rows.values()))
    return dict(tpf=mean(Fraction(a,b) for a,b,c,d in rows.values()), N=n, F=f, Q=q,
                q_per_f=Fraction(q,f), efficiency=Fraction(n,q), correct=correct)


def compare(a,b):
    if set(a) != set(b):
        raise ValueError("must pair identical tasks")
    x, y = metrics(a), metrics(b)
    both = x["correct"] > y["correct"] and x["tpf"] >= y["tpf"]
    return dict(**{k:float(v) if isinstance(v,Fraction) else v for k,v in x.items()},
                **{"ref_"+k:float(v) if isinstance(v,Fraction) else v for k,v in y.items()},
                quality_up_tpf_not_lower=int(both),
                total_Q_not_higher=int(x["Q"]<=y["Q"]),
                pooled_Q_per_F_not_higher=int(x["q_per_f"]<=y["q_per_f"]),
                requests_Q_not_higher=sum(a[i][2]<=b[i][2] for i in a),
                requests_Q_per_F_not_higher=sum(a[i][2]*b[i][1]<=b[i][2]*a[i][1] for i in a),
                wins=sum(a[i][3]>b[i][3] for i in a), losses=sum(a[i][3]<b[i][3] for i in a))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--arm-requests", type=Path, required=True)
    ap.add_argument("--policy", required=True, action="append")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    arms, refs, pairs = load(args.arm_requests), load(args.baseline), []
    for pol in args.policy:
        study = [r for r in arms if r["policy"] == pol]
        a = pack(study)
        model, nominal = study[0]["model"], int(study[0]["nominal_q"])
        source = [r for r in refs if r["model"]==model and r["role"]=="baseline"
                  and r["policy"].startswith(("ibs_","obs_")) and r["policy"].split("_")[-1].isdigit()]
        for ref in sorted({r["policy"] for r in source}):
            width = int(ref.split("_")[-1])
            b = pack([r for r in source if r["policy"]==ref])
            pairs.append(dict(model=model,policy=pol,nominal=nominal,ref=ref,ref_window=width,
                              same_nominal=int(nominal==width),**compare(a,b)))
    if not pairs:
        raise ValueError("no IBS / OBS reference")
    with args.out.open("w", newline="") as f:
        w=csv.DictWriter(f,list(pairs[0]),lineterminator="\n")
        w.writeheader()
        w.writerows(pairs)


if __name__ == "__main__":
    main()
