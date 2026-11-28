"""Reconcile the A0 fixed-Q64 seat screen from the six archived physical tapes."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean


def require(ok, why):
    if not ok:
        raise ValueError(why)


def compact(root):
    report=json.loads((root/"reducer-report.json").read_bytes())
    indices=(7,13,25,33,55,71)
    require(report["classification"]=="valid" and report["valid_case_count"]==len(indices)
            and not report["invalid_cases"], "not a validated six")
    rows=[]
    for i in indices:
        f=root/f"case-{i:02d}.json"
        blob=f.read_bytes()
        require(sha256(blob).hexdigest()==report["sha256"][f.name], "physical receipt drift")
        r=json.loads(blob); a=r["arms"]
        require(r["index"]==i and r["seed"]==17000+i and not r["error"] and
                a["stock"]["physical"]==a["observer"]["physical"] and
                a["stock"]["bill"]==a["observer"]["bill"] and r["stock_observer_identical"],
                "invalid control")
        row={"index":i}
        for name in ("stock","seat"):
            v=a[name]; b=v["bill"]; p=v["physical"]
            d=[len(x["pos"]) for x in p["decoder"]]
            ev=p["events"]; enc=[width for kind,width in ev if kind=="E"]
            require(len(enc)>0 and ev[0][0]=="E" and len(ev)==len(enc)+len(d) and
                    [width for kind,width in ev if kind=="D"]==d and
                    (len(d),len(enc)-1,sum(enc[1:]),len(ev)-1,sum(d),max(d))==
                    (b["D"],b["E"],b["encoder_Q"],b["F"],b["Q"],b["peak_decoder_Q"]) and
                    all(0<n<=64 for n in d) and p["projection_tape"] and
                    all(t==d for t in p["projection_tape"].values()), "invalid forward or layer bill")
            require(b["admissions"]==b["paid_rows"], "unreconciled writes")
            row[name]={k:b[k] for k in ("F","Q","D","E","encoder_Q","peak_decoder_Q")}
            row[name].update(N=v["N"],correct=v["correct"])
        exchanges=a["seat"]["seat"]["exchanges"]
        require(exchanges and len({e["block"] for e in exchanges})==len(exchanges), "invalid dose")
        fork=r["first_fork"]; e=exchanges[0]
        require(e["tick"]==fork["tick"] and fork["reference"]["positions"]==e["before"]
                and fork["treatment"]["positions"]==e["after"] and len(e["before"])==len(e["after"])==64
                and set(e["before"])-set(e["after"])=={e["donor_absolute"]}
                and set(e["after"])-set(e["before"])=={e["prefix_absolute"]}
                and set(e["due"])<=set(e["before"])&set(e["after"]) and e["resident_prefix"]
                and max(e["after"])==max(e["before"]), "first exchange not physically matched")
        row.update(exchanges=len(exchanges), same_output=a["stock"]["output_sha256"]==a["seat"]["output_sha256"])
        rows.append(row)
    totals={}
    for name in ("stock","seat"):
        v=[r[name] for r in rows]
        t={k:sum(x[k] for x in v) for k in ("correct","N","F","Q","D","E","encoder_Q")}
        t.update(peak_decoder_Q=max(x["peak_decoder_Q"] for x in v),
                 tpf=mean(x["N"]/x["F"] for x in v),te=t["N"]/t["Q"])
        totals[name]=t
    return {"source":root.name,"source_sha256":report["sha256"],"cases":rows,"totals":totals}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    print(json.dumps(compact(p.parse_args().root),indent=2,sort_keys=True))
