"""Reconstruct optional-prefix opportunities on fixed LLaDA traces, without rescoring or a model."""
from collections import Counter
import json
from .fixed_family_report import DATA, report
from .llada_all_slots90 import require


def replay(row):
    """Replay v1 eligibility; check the actual selected prefix before attributing any missed seat."""
    q=row["q"]
    require(q<=32 and row["law"]==[1,8], "This diagnostic covers the registered single-block law")
    origin=row["prefill_Q"]
    end=origin+2048
    unresolved=set(range(origin,end))
    seen=set()
    result=Counter()
    for e in row["trace"]:
        lo,hi=max(origin,e["block"]*32),min(end,(e["block"]+1)*32)
        dirty=set(e["dirty_before"])
        masks=unresolved.intersection(range(lo,hi))
        cold=masks-seen
        edge=range(min(masks,default=lo)-1,lo-1,-1)
        pref=[r for r in edge if r in seen and r not in dirty]
        room=q-min(q,len(dirty))
        avail=max(0,room-int(bool(masks-dirty)))
        selected=pref[:min(avail,room//8)] if not cold else []
        require(e["offer"]==q and set(selected)==set(e["prefix"]), "Prefix replay differs from model tape")
        category=("no_unresolved" if not masks else "cold" if cold else "no_clean_prefix" if not pref
                  else "no_spare_row" if not avail else "rounded_to_zero" if not selected else "used")
        result[category]+=1
        seen.update(e["rows"])
        unresolved.difference_update(e["admitted"])
        unresolved.update(e["revoked"])
    require(sum(result.values())==row["F"], "Not all decoder calls attributed")
    return result


if __name__=="__main__":
    root=DATA/"llada21-fixed-family-v1/cpu/law-1-8"
    require(report(DATA,"llada21","cpu")["protocol_smoke3"],"Missing paired three-case test")
    indices=(0,10,20)
    print(json.dumps(dict(model="llada21",device="cpu",law=[1,8],indices=indices,
        counts={q:sum((replay(json.loads((root/f"{i:02}-q{q}.json").read_bytes()))
                       for i in indices),Counter()) for q in (4,8,16,32)}),indent=2))
