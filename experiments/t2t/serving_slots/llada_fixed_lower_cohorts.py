"""Read the public LLaDA profile's lower fixed arms on the receipt host."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean


def check(ok, why):
    if not ok:
        raise ValueError(why)


def load(path):
    return json.loads(path.read_bytes())


def summarize(q4, cross, manifest):
    tasks=load(manifest)
    nr=tasks["samples_sha256"]
    identities=tasks["samples"]
    regs=[load(d/"run.json") for d in (q4,cross)]
    q4reg,greg=regs
    check(q4reg["q"]==4 and q4reg["protocol"]=="llada-frontier-one-seat-v1"
          and q4reg["samples"]==greg["samples_sha256"]==nr
          and q4reg["planner"]=="4883c47d2e8e036f76d386e16c299ea49d05ba646dc387da39c3dbc925bb9833"
          and greg["protocol"]=="llada-cross-race-v1" and greg["law"]==[1,8]
          and greg["retirement"]=="all_resident_no_masks_no_token_debt"
          and load(cross/"complete.json")["run"]==sha256((cross/"run.json").read_bytes()).hexdigest()
          and [r["index"] for r in identities]==list(range(90)), "Wrong baseline run")
    result={}
    for q,root,arm in ((4,q4,"old"),(8,cross,"c8-noborrow"),(16,cross,"c16-borrow")):
        train={}
        hashes={}
        for i,t in enumerate(identities):
            p=root/"rows"/(f"{i:02d}.json" if q==4 else f"{arm}-{i:02d}.json")
            if not p.exists() and q==4:
                continue
            blob=p.read_bytes()
            r=json.loads(blob)
            check(all(r[k]==t[k] for k in ("index","task_id","dataset","seed")), "Wrong frozen identity")
            if q==4:
                a=r["old"]
                check(len(r["input"])==len(a["output_digest"])==64 and not a["actions"],"Wrong Q4 arm")
            else:
                a=r
                check(a["arm"]==arm and a["q"]==q and a["cross_open"]=="head"
                      and a["borrow"]==(q==16) and a["law"]==[1,8] and a["samples_sha256"]==nr
                      and a["projection_modules"]==20 and a["encoder_Q"]==0 and "H800" in a["gpu"],
                      "Wrong lower offer")
                trace=a["trace"]
                check(len(trace)==a["F"] and sum(e["q"] for e in trace)==a["Q"]==a["decoder_Q"]
                      and max(e["q"] for e in trace)==a["max_decoder_Q"],"Incorrect physical totals")
                for e in trace:
                    w=e["q"]
                    check(e["offer"]==q and e["stage"]!="prefill" and e["prefix_kv_length"]>0
                          and 0<w<=q and e["layer_physical_q"]==[w]*20 and e["F"]==1
                          and e["actual_q"]==e["physical_q"]==len(set(e["absolute_positions"]))==w,
                          "Invalid history or layer width")
            check(0<a["max_decoder_Q"]<=q and a["F"]<=a["Q"]<=a["F"]*q and a["F"]>0
                  and type(a["correct"]) is bool and not (a["failure"] and a["correct"]),"Invalid request")
            train[i]=a
            hashes[i]=sha256(blob).digest()
        cohorts={}
        for name,idx in (("original30",range(30)),("new60",range(30,90)),("frozen90",range(90))):
            if not all(i in train for i in idx):
                continue
            rs=[train[i] for i in idx]
            v={k:sum(r[k] for r in rs) for k in ("N","F","Q","correct")}
            cohorts[name]=dict(**v,n=len(rs),peak=max(r["max_decoder_Q"] for r in rs),
                               macro_TPF=mean(r["N"]/r["F"] for r in rs),
                               failures=sum(bool(r["failure"]) for r in rs),
                               receipts_sha256=sha256(b"".join(hashes[i] for i in idx)).hexdigest())
        result[str(q)]=dict(arm=arm,source=str(root),cohorts=cohorts)
    return dict(profile="quarter_edit_warm",samples_sha256=nr,tiers=result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("q4","cross","manifest"):
        p.add_argument("--"+k,type=Path,required=True)
    print(json.dumps(summarize(**vars(p.parse_args())),indent=2,sort_keys=True))
