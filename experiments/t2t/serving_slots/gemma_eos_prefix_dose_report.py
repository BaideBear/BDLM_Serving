"""Offline export at a sealed checkpoint; token IDs and raw tapes stay on A800."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import mean

from .gemma_eos_prefix_dose import ROLES, first_change
from .gemma_eos_prefix_dose_eval import HIST, ROOT, RT, sha
from .gemma_joint_verify_eval import GPU, ORDER, check, digest
from .gemma_rejected_noise_report import interval


def read(p):
    return json.loads(p.read_bytes())


def bill(x, notes):
    check(type(x["correct"]) is bool and len(notes)==len(x["tape"])==x["decoder_F"]
          and x["F"]==x["decoder_F"]+x["encoder_F"] and x["encoder_Q"]==256*x["encoder_F"],
          "Bad request totals")
    f,q=Counter(),Counter()
    for j,n in enumerate(notes):
        width=n["width"]
        route=n["route"]
        check(n["tick"]==j and 0<width<=64 and all(n[k]>=0 for k in ROLES) and
              sum(n[k] for k in ROLES)==width and 0<=n["cold_mask"]<=n["mask"] and
              n["verify_dirty"]==(n["dirty"] if route.startswith("verify") else 0) and
              (not route.startswith("verify") or width==64) and
              0<=n["extra_width"]==n["more_prefix"]-n["fewer_masks"] and
              n["more_prefix"]==len(n.get("added",())) and n["fewer_masks"]==len(n.get("removed",())),
              "Bad physical row classification")
        q[route]+=width
        f[route]+=1
    check(q==x["route_Q"] and f==x["route_F"] and sum(q.values())==x["decoder_Q"],
          "Unreconciled physical trace")
    total={k:x[k] for k in ("F","N","decoder_F","encoder_F","decoder_Q","encoder_Q")}
    total.update({k:sum(n[k] for n in notes) for k in (*ROLES,"verify_dirty","cold_mask",
                                                         "more_prefix","fewer_masks","extra_width")})
    total.update(total_Q=x["decoder_Q"]+x["encoder_Q"], max_Q=max(n["width"] for n in notes),
                 correct=int(x["correct"]),dose_ticks=sum(n["more_prefix"]>0 for n in notes),
                 mask_trade_ticks=sum(n["fewer_masks"]>0 for n in notes),
                 slack_fill_ticks=sum(n["extra_width"]>0 for n in notes),
                 sparse_ticks=f["sparse"],drain_ticks=f["drain"],
                 verify_ticks=f["verify"]+f["verify_paid"])
    return total


def group(rows, name):
    result=dict(cohort=name,pairs=len(rows),exposed=sum(r["first_tick"] is not None for r in rows),
                same_ids=sum(r["same_ids"] for r in rows),rescues=sum(r["on_correct"]>r["off_correct"] for r in rows),
                losses=sum(r["on_correct"]<r["off_correct"] for r in rows))
    pairs=[tuple(dict(dataset=r["dataset"],correct=r[a+"_correct"],tpf=r[a+"_N"]/r[a+"_F"])
                 for a in ("off","on")) for r in rows]
    for k in ("tpf","correct"):
        v=interval(pairs,lambda p:p[k])
        result[f"delta_{k}_lo95"],result[f"delta_{k}_hi95"]=v or (None,None)
    for a in ("off_","on_"):
        for k in rows[0]:
            if k.startswith(a) and not k.endswith(("sha256","_TPF","_TE","_per_total_Q_diagnostic")):
                fn=max if k==a+"max_Q" else sum
                result[k]=fn(r[k] for r in rows)
        n=result[a+"N"]
        result[a+"macro_TPF"]=mean(r[a+"N"]/r[a+"F"] for r in rows)
        result[a+"TE"]=n/result[a+"decoder_Q"]
        result[a+"N_per_total_Q_diagnostic"]=n/result[a+"total_Q"]
        result[a+"pooled_TPF"]=n/result[a+"F"]
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=int,choices=(3,30,90),default=90)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    n=args.checkpoint
    reg=read(ROOT/"registration.json")
    mark=read(ROOT/f"checkpoint-{n}.json")
    manifest=read(Path("/gs/runtime/verified-eos-1128/repo/benchmarks/frozen90/manifest.json"))
    tasks={t["index"]:t for t in manifest["samples"]}
    check(reg["schema"]=="eos-prefix-dose-q64-v1" and reg["denominators"]==[8,4] and
          reg["policy"]==sha(RT/"registration.md") and reg["gpu"]==GPU and
          reg["samples"]==manifest["samples_sha256"] and mark["registration"]==digest(reg) and
          mark["pairs"]==n, "Wrong registration or checkpoint")
    if n==90:
        check(read(ROOT/"complete.json")==dict(**mark,outcome="frozen90") and
              len(list(ROOT.glob("C64-pair-*.json")))==90, "Not frozen90")
    rows=[]
    hashes={}
    for i in ORDER[:n]:
        p=ROOT/f"C64-pair-{i:02}.json"
        r=read(p)
        hp=HIST/p.name
        ref=read(hp)
        check(r["registration"]==digest(reg) and r["reference_sha256"]==sha(hp) and
              ref["source"]==reg["base"] and ref["scorer_sha256"]==reg["scorer"] and
              ref["gpu"]==GPU and ref["q"]==64 and ref["on"]==r["off"] and
              all(r[k]==tasks[i][k]==ref[k] for k in ("index","dataset","task_id","seed")),
              "Mixed study or incomplete control replay")
        x,y=r["off"],r["on"]
        first=first_change(x,y,[r["rows"][a] for a in ("off","on")])
        check(first==r["first"], "First aligned difference mismatch")
        row=dict(index=i,dataset=r["dataset"],task_id=r["task_id"],seed=r["seed"],
                 first_tick=first and first["tick"],first_head=first and first["head"],
                 first_added="" if not first else ",".join(map(str,first["added"])),
                 first_removed="" if not first else ",".join(map(str,first["removed"])),
                 same_ids=int(x["output_ids"]==y["output_ids"]))
        for arm in ("off","on"):
            x=r[arm]
            row.update({arm+"_"+k:v for k,v in bill(x,r["rows"][arm]).items()})
            row[arm+"_ids_sha256"]=digest(x["output_ids"])
            row[arm+"_TPF"]=x["N"]/x["F"]
            row[arm+"_TE"]=x["N"]/x["decoder_Q"]
            row[arm+"_N_per_total_Q_diagnostic"]=x["N"]/row[arm+"_total_Q"]
        # Historical taxonomy gave verification precedence over mandatory rows.
        # Check the control partition independently, never synthesize the 1/4 arm.
        oldp=HIST.parent/"paid-close-1128"/f"C64-fuse1-{i:02}.json"
        old=read(oldp)
        overlap=row["off_verify_dirty"]
        suffix=(ref["on"]["hit"] or {}).get("theoretical_Q",0)
        check(sha(oldp)==ref["reference_sha256"] and old["fused_dirty_Q"]==overlap and
              old["decoder_Q"]==row["off_decoder_Q"]+suffix and
              old["categories"]==dict(mandatory=row["off_dirty"]-overlap,unresolved=row["off_mask"],
                    prefix=row["off_prefix"],verify=row["off_verify_clean"]+overlap+suffix,
                    rebuild=0,**{"clean-other":0}), "Historical physical class partition mismatch")
        rows.append(row)
        hashes[p.name]=sha(p)
    check(sum(r["first_tick"] is not None for r in rows)==mark["exposed"], "Bad exposure denominator")
    rows.sort(key=lambda r:r["index"])
    groups=[group([r for r in rows if r["index"] in (0,10,20)],"anchors3")]
    if n>=30:
        groups.append(group(rows[:30],"original30"))
    if n==90:
        groups.extend([group(rows[30:],"new60"),group(rows,"frozen90")])
    args.out.mkdir(parents=True,exist_ok=True)
    for filename,data in (("pairs",rows),("cohorts",groups)):
        with (args.out/(filename+".csv")).open("w",newline="") as f:
            out=csv.DictWriter(f,data[0].keys(),lineterminator="\n")
            out.writeheader()
            out.writerows(data)
    (args.out/"provenance.json").write_text(json.dumps(dict(registration=reg,
        checkpoint=mark,report_sha256=sha(Path(__file__)),pair_files_sha256=digest(sorted(hashes.items())),
        TE="sum(N) / sum(selected history-reading decoder_Q)",
        N_per_total_Q_diagnostic="sum(N) / sum(decoder_Q + post-prefill encoder_Q); not TE"),
        indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    main()
