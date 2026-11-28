"""Summarize sealed H4 triplets using only the compact receipts."""
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean


def digest(o):
    return sha256(json.dumps(o,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def cohort(rs):
    arms={}
    for name in ("control","noop","lead"):
        a=[r[name] for r in rs]
        s={k:sum(v[k] for v in a) for k in ("correct","N","F","Q","one","two","prefill_Q","eos")}
        arms[name]=dict(s,macro_TPF=mean(v["N"]/v["F"] for v in a),pooled_TE=s["N"]/s["Q"])
    c,t=arms["control"],arms["lead"]
    if 32*((t["F"]-c["F"])+(t["two"]-c["two"]))!=t["Q"]-c["Q"]:
        raise ValueError("physical Q decomposition")
    return dict(arms,count=len(rs),exposed=sum(bool(r["first"]) for r in rs),
                lead_admissions=sum(len(r["lead_votes"]) for r in rs),
                noop_eligible_ticks=sum(len(r["noop_votes"]) for r in rs),
                ids_equal=sum(r["ids_equal"] for r in rs),
                gains=[r["index"] for r in rs if r["lead"]["correct"] and not r["control"]["correct"]],
                losses=[r["index"] for r in rs if not r["lead"]["correct"] and r["control"]["correct"]],
                delta_Q=t["Q"]-c["Q"],delta_F=t["F"]-c["F"],
                calls_with_second_slot_delta=t["two"]-c["two"],
                fewer_F=sum(r["lead"]["F"]<r["control"]["F"] for r in rs),
                more_F=sum(r["lead"]["F"]>r["control"]["F"] for r in rs),
                fewer_Q=sum(r["lead"]["Q"]<r["control"]["Q"] for r in rs),
                more_Q=sum(r["lead"]["Q"]>r["control"]["Q"] for r in rs))


def run(d):
    reg=json.loads((d/"registration.json").read_bytes())
    tag=sha256((d/"registration.json").read_bytes()).hexdigest()
    rs=[json.loads(x) for x in (d/"compact.ndjson").read_text().splitlines()]
    done,q=(json.loads((d/p).read_bytes()) for p in ("complete.json","qualified.json"))
    order=reg["pilot"]+[i for i in range(90) if i not in reg["pilot"]]
    if ([r["index"] for r in rs]!=order or len(rs)!=90 or done["requests"]!=270 or
        not done["registration"]==q["registration"]==tag or done["sha"]!=digest(rs) or
        q["sha"]!=digest(rs[:6]) or q["cases"]!=order[:6]):
        raise ValueError("unsealed or duplicated panel")
    for r in rs:
        if not (r["reference_replayed"] and r["noop_exact"] and r["registration"]==tag and
                r["control"]==r["noop"] and (r["first"] or r["control"]==r["lead"])):
            raise ValueError("control or exposure mismatch")
    return {name:cohort(rows) for name,rows in dict(pilot=rs[:6],original30=[r for r in rs if r["index"]<30],
                new60=[r for r in rs if r["index"]>=30],frozen90=rs).items()}


if __name__=="__main__":
    print(json.dumps(run(Path(__file__).resolve().parent),sort_keys=True,indent=2))
