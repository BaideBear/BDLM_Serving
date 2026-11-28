"""Count opportunities, not corrections, on the measured A3 admission payments."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


def audit(root, prior):
    def read(p): return json.loads(p.read_bytes())
    S=(7,13,25,33,55,71)
    s=read(root/"summary.json")
    if ((root/"runner.exit").read_text().strip()!="0" or s["gate"]!="passed" or
        s["schema"]!="gemma-paid-agreement-payment-observer-six-v1" or tuple(s["cases"]) != S):
        raise ValueError("unsealed or unknown registration")
    rows,tot=[],{k:Counter() for k in ("native","extra")}
    for j,i in enumerate(S):
        f=root/f"case-{i:02}.json"
        a=read(f)
        if (a != s["per_case"][j] or not a["payment_observer"]["identical"] or
            {k:v for k,v in a.items() if k!="payment_observer"} != read(prior/f.name)):
            raise ValueError("baseline or observer changed")
        original=a["arms"]["apply"]
        chosen={(o["tick"],p) for o in original["agreement"]["extra_events"] for p in o["extra"]}
        covered=set()
        counts={k:Counter() for k in tot}
        tape=a["payment_observer"]["events"]
        if len(tape) != original["bill"]["paid_rows"] or len(tape)!=len({r["pos"] for r in tape}):
            raise ValueError("not one immutable IBS admission per payment")
        for r in tape:
            key=r["write"],r["pos"]
            if ((r["role"]=="extra") != (key in chosen) or not r["write"]<r["payment"] or
                not 0<=r["written_p"]<=r["top_p"]<=1):
                raise ValueError("bad paid-row observation")
            covered.add(key)
            c=counts[r["role"]]; c["payments"]+=1
            changed=r["top"]!=r["old"]
            c["changed_top"]+=changed
            c["ordinary_changed_above_09"]+=int(changed and r["ordinary"] and r["top_p"]>.9)
        if not chosen<=covered:
            raise ValueError("unlinked extra admission")
        for k in tot: tot[k].update(counts[k])
        rows.append(dict(index=i,roles=counts,sha256=sha256(f.read_bytes()).hexdigest()))
    exposed=[r["index"] for r in rows if r["roles"]["extra"]["ordinary_changed_above_09"]]
    return dict(q=128,device=s["gpu"],total=tot,exposed=exposed,
                eligible_for_followup=len(exposed)>=2,rows=rows)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","prior"):p.add_argument(name,type=Path)
    print(json.dumps(audit(**vars(p.parse_args())),indent=2))
