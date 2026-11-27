"""Audit one paired GPU gate study, on its host. Only print aggregate data."""
import argparse
import json
from pathlib import Path
from statistics import mean


def check(ok, message):
    if not ok:
        raise ValueError(message)


def validate(row, spec, task, digest):
    for k in ("arm","q","cross_open","borrow","coverage"):
        check(row[k]==spec[k], f"configuration mismatch: {k}")
    check(all(row[k]==task[k] for k in ("index","dataset","task_id","seed"))
          and row["samples_sha256"]==digest and row["law"]==[1,8], "identity mismatch")
    q=row["q"]
    check(row["projection_modules"]==20 and row["encoder_Q"]==0 and row["device_uuid"]
          and type(row["correct"]) is bool and (not row["failure"] or not row["correct"]), "invalid receipt")
    f,h,t,ro=0,0,0,0
    for e in row["trace"]:
        if e["stage"]=="prefill":
            continue
        w=e["actual_q"]
        check(0<w<=q and e["offer"]==q and w==e["physical_q"]==len(e["absolute_positions"])
              and len(set(e["absolute_positions"]))==w and e["layer_physical_q"]==[w]*20
              and e["head_q"]+e["tail_q"]==w and e["F"]==1, "decoder tape mismatch")
        f+=1
        h+=e["head_q"]
        t+=e["tail_q"]
        ro+=len(e["tail_read_only"])
    check((f,h,t,ro,h+t,h+t)==tuple(row[k] for k in
          ("F","head_Q","tail_Q","tail_read_only_Q","Q","decoder_Q")) and f>0 and row["N"]>=0,
          "nonconserving workload")
    return {k:row[k] for k in ("N","F","Q","head_Q","tail_Q","tail_read_only_Q","input_sha256",
                                "device_uuid","correct","failure","head_full_F","full64_F")}


def summarize(root, manifest):
    reg=json.loads((root/"run.json").read_bytes())
    digest=manifest["samples_sha256"]
    check(reg["protocol"]=="llada-cross-race-v2" and reg["samples_sha256"]==digest
          and reg["law"]==[1,8] and reg["retirement"]=="all_resident_no_masks_no_token_debt",
          "wrong gate registration")
    tasks={x["index"]:x for x in manifest["samples"]}
    check(set(tasks)==set(range(90)), "not frozen90")
    arms={s["arm"]:s for s in reg["arms"]}
    contrasts=[(f"c{q}-noborrow",f"c{q}-borrow") for q in (8,16)]
    contrasts += [(f"c64-{p}",f"c64-full-{p}") for p in ("off","eager","half","quarter")]
    contrasts += [(f"c64-{cov}off",f"c64-{cov}{g}") for cov in ("","full-")
                  for g in ("eager","half","quarter")]
    check(set(arms)=={a for pair in contrasts for a in pair},"wrong gate arms")
    data={a:{} for a in arms}
    for a,s in arms.items():
        for i in tasks:
            path=root/"rows"/f"{a}-{i:02}.json"
            if path.exists():
                data[a][i]=validate(json.loads(path.read_bytes()),s,tasks[i],digest)
    out={}
    for label,idx in (("original30",range(30)),("new60",range(30,90)),("frozen90",range(90))):
        cohort=out[label]={}
        for a,b in contrasts:
            both=sorted(set(idx)&data[a].keys()&data[b].keys())
            v=cohort[a+" / "+b]=dict(n=len(both),expected=len(idx))
            check(all(all(data[a][i][k]==data[b][i][k] for k in ("input_sha256","device_uuid"))
                      for i in both),"not a same-device matched pair")
            if len(both)!=len(idx):
                continue
            for m in (a,b):
                rs=[data[m][i] for i in both]
                stats={k:sum(r[k] for r in rs) for k in
                       ("correct","N","F","Q","head_Q","tail_Q","tail_read_only_Q","head_full_F","full64_F")}
                stats.update(macro_TPF=mean(r["N"]/r["F"] for r in rs),
                             token_efficiency=stats["N"]/stats["Q"],failures=sum(bool(r["failure"]) for r in rs))
                v[m]=stats
            v["rescues"]=sum(not data[a][i]["correct"] and data[b][i]["correct"] for i in both)
            v["losses"]=sum(data[a][i]["correct"] and not data[b][i]["correct"] for i in both)
    return dict(samples_sha256=digest,protocol=reg["protocol"],registered_sources=reg["sources"],cohorts=out)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root",type=Path)
    ap.add_argument("manifest",type=Path)
    args=ap.parse_args()
    print(json.dumps(summarize(args.root,json.loads(args.manifest.read_bytes())),indent=2,sort_keys=True))
