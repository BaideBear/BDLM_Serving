"""Check sealed receipts without a model; only emit scalar counts and hashes."""
import csv
import json
from pathlib import Path
from statistics import mean

from .gemma_ibs_prefix_bridge import CASES,TIERS,audit,match,require
from .gemma_ibs_prefix_bridge_eval import DEADLINE,GPU,PINS,RESULT,RT,SAMPLES,digest,read,sha


def summarize(q,group,rows):
    expected={"screen9":9,"original_anchors3":3,"new_anchors6":6}[group]
    s=dict(q=q,cohort=group,n=len(rows),expected=expected,missing=expected-len(rows),
           replayed=len(rows),exposed=sum(r["first"] is not None for r in rows),
           first_logits_changed=sum(r["first_logits_changed"] is True for r in rows),
           changed_decisions=sum(r["decision_tick"] is not None for r in rows),
           same_final_IDs=sum(r["same_ids"] for r in rows),
           rescues=sum(r["on_correct"]>r["off_correct"] for r in rows),
           losses=sum(r["on_correct"]<r["off_correct"] for r in rows))
    if rows:
        for a in ("off","on"):
            for k in ("correct","N","F","decoder_F","decoder_Q","encoder_F","encoder_Q","heads",
                      "eligible_ticks","extra_Q","admissions","dirty_paid","warm_F","mask_F","truncated"):
                s[a+"_"+k]=sum(r[a+"_"+k] for r in rows)
            s[a+"_macro_TPF"]=mean(r[a+"_TPF"] for r in rows)
            s[a+"_TE"]=s[a+"_N"]/s[a+"_decoder_Q"]
            s[a+"_peak"]=max(r[a+"_peak"] for r in rows)
        s["delta_macro_TPF"]=s["on_macro_TPF"]-s["off_macro_TPF"]
        s["delta_decoder_Q"]=s["on_decoder_Q"]-s["off_decoder_Q"]
    return s


def report():
    from bdlm_serving.evaluation.frozen90 import load_tasks
    marks=[m for m in ("complete","partial") if (RESULT/(m+".json")).exists()]
    require(len(marks)==1,"No unique terminal marker")
    status,=marks
    reg=read(RESULT/"registration.json")
    rd=digest(reg)
    plan=[(q,i) for i in CASES for q in TIERS]
    marker=read(RESULT/(status+".json"))
    n=marker["n"]
    paths=[RESULT/f"q{q}-{i:02}.json" for q,i in plan[:n]]
    require(marker["reg"]==rd and 0<n<=18 and (n==18)==(status=="complete") and
            set(paths)==set(RESULT.glob("q*-*.json")) and reg["uuid"]==GPU and reg["deadline"]==DEADLINE and
            reg["case_indices"]==list(CASES) and reg["tiers"]==list(TIERS) and reg["pins"]==PINS and
            reg["registration"]==sha(RT/"registration.md") and reg["launch"]==sha(RT/"launch.sh") and
            reg["model"]==read(RT/"checkpoint.json") and all(v==sha(Path(__file__).with_name(k+".py"))
            for k,v in reg["sources"].items()) and sha(SAMPLES)==PINS["samples"],"Seal/registration mismatch")
    tasks=load_tasks(SAMPLES)
    rows=[]
    inputs={}
    for (q,i),p in zip(plan[:n],paths,strict=True):
        r=read(p)
        a,b=r["off"],r["on"]
        u=match(a,b)
        require(r["first"]==u and r["reg"]==rd and r["pid"]==reg["pid"] and r["q"]==q and
                all(r[k]==tasks[i][k] for k in ("index","dataset","task_id","seed")) and
                r["plain_sha"]==digest({k:v for k,v in a.items() if k not in ("notes","bill")}) and
                digest(a["enc"][0][0])==digest(b["enc"][0][0])==r["prompt_sha"] and
                inputs.setdefault(i,r["input_sha"])==r["input_sha"],"Not a same-board paired replay")
        d=dict(q=q,index=i,dataset=r["dataset"],task_id=r["task_id"],seed=r["seed"],plain_sha=r["plain_sha"],
               pair_sha=sha(p),input_sha=r["input_sha"],same_ids=int(a["output_ids"]==b["output_ids"]),
               first=u["first"],first_head=u.get("first_head"),first_added=u.get("first_added",0),
               first_base_Q=u.get("first_base_Q"),first_logits_changed=u.get("first_logits_changed"),
               decision_tick=u.get("decision_tick"),first_added_local=" ".join(map(str,u.get("first_spare_rows",[]))))
        for arm,x in (("off",a),("on",b)):
            enc=list(map(tuple,x["enc"]))
            bill=audit(x["notes"],x["trace"],list(map(tuple,x["events"])),x["output_ids"],
                       enc,enc[0][0],set(r["eos"]),q,arm=="on")
            require(bill==x["bill"] and bill["admissions"]==bill["dirty_paid"]==256*bill["heads"] and
                    all(x[k]==bill[k] for k in ("F","decoder_F","decoder_Q","encoder_F","encoder_Q")) and
                    type(x["correct"]) is bool and 0<=x["N"]<=len(x["output_ids"]),"Bill mismatch")
            d.update({arm+"_"+k:v for k,v in bill.items()})
            d.update({arm+"_N":x["N"],arm+"_correct":int(x["correct"]),arm+"_TPF":x["N"]/x["F"],
                      arm+"_TE":x["N"]/x["decoder_Q"],arm+"_output_sha":digest(x["output_ids"]),
                      arm+"_truncated":int(x["output_ids"][-1] not in r["eos"])})
        rows.append(d)
    groups=[]
    for q in TIERS:
        qr=[r for r in rows if r["q"]==q]
        for g,part in (("screen9",qr),("original_anchors3",[r for r in qr if r["index"]<30]),
                       ("new_anchors6",[r for r in qr if r["index"]>=30])):
            groups.append(summarize(q,g,part))
    s32,s64=groups[0],groups[3]
    decision=None if n!=18 else dict(exposure=bool(s32["exposed"] and s64["exposed"]),
            net_Q32_gain=s32["on_correct"]>s32["off_correct"],
            preserved_Q32_TPF=s32["on_macro_TPF"]>=.98*s32["off_macro_TPF"],
            no_net_Q64_loss=s64["on_correct"]>=s64["off_correct"])
    prov=dict(status=status,marker=marker,registration=reg,registration_digest=rd,
              same_board_unmodified_replays=n,raw=str(RESULT),report_sha256=sha(Path(__file__)),
              missing_pairs=plan[n:],screen=decision,pairs_digest=digest([r["pair_sha"] for r in rows]),
              macro_TPF="mean(N/(decoder_F+postprefill_encoder_F))",TE="sum(N)/sum(selected decoder Q)",
              historical_replay="independent on A1; no cross-host per-row equivalence asserted")
    return rows,groups,prov


def main():
    rows,groups,prov=report()
    out=RT/"compact"
    out.mkdir(exist_ok=True)
    for name,data in (("pairs",rows),("cohorts",groups)):
        with (out/(name+".csv")).open("w",newline="") as fp:
            w=csv.DictWriter(fp,list(dict.fromkeys(k for r in data for k in r)),lineterminator="\n")
            w.writeheader()
            w.writerows(data)
    (out/"provenance.json").write_text(json.dumps(prov,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    main()
