"""Sealed checkpoints only. Re-audit on the GPU host without loading a model."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from .gemma_cold_frontier_last import TIERS,audit
from .gemma_cold_frontier_last_eval import ROOT,RT,HIST,GPU,SCORER,read,sha
from .gemma_joint_verify_eval import ORDER,check,digest
from .gemma_rejected_noise_report import interval


def group(q,name,rows):
    s=dict(q=q,cohort=name,pairs=len(rows),exposed=len(rows),same_ids=sum(r["same_ids"] for r in rows),
           rescues=sum(r["on_correct"]>r["off_correct"] for r in rows),
           losses=sum(r["on_correct"]<r["off_correct"] for r in rows))
    # Fixed off trajectories only; this is not evidence that a fusion is legal.
    s["off_close_only_macro_TPF_ceiling"]=mean(r["off_N"]/(r["off_F"]-r["off_closing_drain_F"]) for r in rows)
    paired=[tuple(dict(dataset=r["dataset"],correct=r[a+"_correct"],TPF=r[a+"_TPF"]) for a in ("off","on")) for r in rows]
    for metric in ("correct","TPF"):
        s["delta_"+metric+"_lo95"],s["delta_"+metric+"_hi95"]=interval(paired,lambda p:p[metric]) or (None,None)
    for a in ("off_","on_"):
        for k in rows[0]:
            if k.startswith(a) and not k.endswith(("sha256","_TPF","_TE","_first_local_start")):
                s[k]=(max if k==a+"peak" else sum)(r[k] for r in rows)
        s[a+"macro_TPF"]=mean(r[a+"TPF"] for r in rows)
        s[a+"TE"]=s[a+"N"]/s[a+"decoder_Q"]
        s[a+"pooled_TPF"]=s[a+"N"]/s[a+"F"]
        for k in ("warm_F","cold_F","warm_entry_frontier_masks"):
            s[a+k+"_per_head"]=s[a+k]/s[a+"heads"]
    return s


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=int,choices=(3,30,90),default=90)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    n=a.checkpoint
    reg=read(ROOT/"registration.json")
    hreg=digest(reg)
    mark=read(ROOT/f"checkpoint-{n}.json")
    mp=Path("/gs/runtime/verified-eos-1128/repo/benchmarks/frozen90/manifest.json")
    m=read(mp)
    tasks={x["index"]:x for x in m["samples"]}
    check(reg["schema"]=="cold-frontier-last-a1-v1" and reg["gpu"]==GPU and reg["scorer"]==SCORER and
          reg["inputs"]==m["samples_sha256"] and reg["manifest"]==sha(mp) and reg["policy"]==sha(RT/"registration.md")
          and all(sha(Path(__file__).with_name(k+".py"))==v for k,v in reg["source"].items())
          and mark==dict(registration=hreg,tiers=list(TIERS),pairs_per_tier=n),"Bad cohort or registration")
    if n==90:
        check(read(ROOT/"complete.json")==dict(**mark,outcome="frozen90") and
              len(list(ROOT.glob("C*-pair-*.json")))==180,"Not complete")
    refhash=[]
    hashes={}
    pairs=[]
    cohorts=[]
    for q in TIERS:
        rset=[]
        for i in sorted(ORDER[:n]):
            p=ROOT/f"C{q}-pair-{i:02}.json"
            r=read(p)
            h=read(HIST/p.name)
            refhash.append(sha(HIST/p.name))
            check(r["q"]==h["q"]==q and r["pid"]==reg["pid"] and r["registration"]==hreg and
                  r["ref"]==refhash[-1] and h["source"]==reg["base"] and h["scorer_sha256"]==SCORER and
                  h["samples_sha256"]==reg["inputs"] and r["off"]==h["on"] and
                  all(r[k]==h[k]==tasks[i][k] for k in ("index","seed","task_id","dataset")),"Replay mismatch")
            row=dict(q=q,index=i,seed=r["seed"],task_id=r["task_id"],dataset=r["dataset"],
                     first_tick=0,first_width=q,off_first_local_start=0,on_first_local_start=q,
                     same_ids=int(r["off"]["output_ids"]==r["on"]["output_ids"]))
            check(r["off"]["tape"][0]!=r["on"]["tape"][0],"No physical exposure")
            for arm in ("off","on"):
                x=r[arm]
                t=r["ticks"][arm]
                ph=audit(x,t,q,arm=="on")
                check(ph==r["phases"][arm] and ph["dirty_Q"]==ph["writes"] and
                      type(x["correct"]) is bool and x["decoder_Q"]==sum(x["route_Q"].values()) and
                      x["decoder_F"]==sum(x["route_F"].values()) and
                      t[0]["cold"]==list(range(q,2*q) if arm=="on" else range(q)),"Invalid physical or cold ledger")
                d={k:x[k] for k in ("F","N","decoder_F","decoder_Q","encoder_F","encoder_Q")}
                d.update(ph,correct=int(x["correct"]),ids_sha256=digest(x["output_ids"]),
                         TE=x["N"]/x["decoder_Q"],TPF=x["N"]/x["F"])
                cold=[e for e in t if e["phase"]=="cold"]
                d.update(cold_writes=sum(e["writes"] for e in cold),cold_dirty_Q=sum(e["dirty"] for e in cold),
                         cold_idle_Q=q*ph["cold_F"]-ph["cold_Q"],cold_drain_only_F=ph["cold_F"]-ph["cold_sample_F"])
                check(ph["cold_Q"]==ph["cold_Q_first"]+d["cold_dirty_Q"] and
                      0<=d["cold_writes"]-d["cold_dirty_Q"]<=q*ph["heads"],"Cold payment mismatch")
                first=[e for e in t if e["head"]==0]
                d.update(first_head_warm_F=sum(e["phase"]=="warm" for e in first),
                         first_head_warm_entry_masks=next(e["first_tile_masks"] for e in first if e["phase"]!="cold"))
                row.update({arm+"_"+k:v for k,v in d.items()})
            rset.append(row)
            hashes[p.name]=sha(p)
        pairs.extend(rset)
        cohorts.append(group(q,"anchors3",[v for v in rset if v["index"] in (0,10,20)]))
        if n>=30:
            cohorts.append(group(q,"original30",rset[:30]))
        if n==90:
            cohorts.extend([group(q,"new60",rset[30:]),group(q,"frozen90",rset)])
    if n==90:
        check(reg["reference_digest"]==digest(refhash),"Mixed references")
    a.out.mkdir(parents=True,exist_ok=True)
    for name,data in (("pairs",pairs),("cohorts",cohorts)):
        with (a.out/(name+".csv")).open("w",newline="") as f:
            w=csv.DictWriter(f,data[0],lineterminator="\n")
            w.writeheader()
            w.writerows(data)
    (a.out/"provenance.json").write_text(json.dumps(dict(registration=reg,checkpoint=mark,
             pair_files_sha256=digest(sorted(hashes.items())),report_sha256=sha(Path(__file__)),
             canonical_TE="sum(N)/sum(selected history-reading decoder Q)",raw=str(ROOT)),
             sort_keys=True,indent=2)+"\n")


if __name__=="__main__":
    main()
