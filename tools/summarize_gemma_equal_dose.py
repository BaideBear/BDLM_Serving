"""Seal the A3 six only after reproducing the retained prior tracks."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean


def seal(root, frozen, earlier):
    def read(p): return json.loads(p.read_bytes())
    cases = (7,13,25,33,55,71)
    summary = read(root/"summary.json")
    if (root.joinpath("runner.exit").read_text().strip() != "0" or
        summary["gate"] != "passed" or tuple(summary["cases"]) != cases or
        summary["schema"] != "gemma-paid-agreement-equal-dose-six-v1" or summary["q"] != 128):
        raise ValueError("unsealed or unexpected run")
    rows = []
    for j,i in enumerate(cases):
        name = f"case-{i:02}.json"
        a,old,formal = (read(p/name) for p in (root,earlier,frozen))
        if (a != summary["per_case"][j] or {k:v for k,v in a.items() if k!="equal_dose"} != old
                or any(a.get(k) != formal.get(k) for k in ("index","task_id","seed","first_extra","stock_observer_identical"))
                or formal["error"] is not None or not a["stock_observer_identical"]):
            raise ValueError(f"three-arm replay failed on {i}")
        for k,v in a["arms"].items():
            if any(v[f] != formal["arms"][k][f]
                   for f in ("bill","trigger","payment","N","correct","scoring")):
                raise ValueError(f"prior A3 formal arm mismatch on {i}/{k}")
        x=a["equal_dose"]
        fork=x["fork"]
        rows.append(dict(index=i,dataset=a["dataset"],fork=None if fork is None else {
            k:fork[k] for k in ("tick","block","k","repeat","current","paid_Q")},
            arms=x["arms"], doses={k:sum(n for _,n in v) for k,v in x["doses"].items()},
            case_sha256=sha256((root/name).read_bytes()).hexdigest()))
    def agg(arm, selection):
        vs=[r["arms"][arm] for r in selection]
        n=sum(v["N"] for v in vs)
        f=sum(v["bill"]["F"] for v in vs)
        q=sum(v["bill"]["Q"] for v in vs)
        return dict(correct=sum(v["correct"] for v in vs), N=n,F=f,Q=q,te=n/q,
                    tpf=mean(v["N"]/v["bill"]["F"] for v in vs))
    return dict(q=128,device=summary["gpu"],rows=rows,cohorts={
        label:{arm:agg(arm,s) for arm in ("stock","repeat","current")}
        for label,s in (("six",rows),("orig3",rows[:3]),("added3",rows[3:]))})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("root","frozen","earlier"): parser.add_argument(name,type=Path)
    print(json.dumps(seal(**vars(parser.parse_args())),indent=2))
