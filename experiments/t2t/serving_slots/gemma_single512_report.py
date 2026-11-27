"""Read and validate the complete remote physical tapes; export compact paired metrics only."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean

from .gemma_joint_verify_report import COHORTS, FIELDS, REG, table, tpf
from .gemma_rejected_noise_report import interval
from .gemma_single512_eval import QUAL, check, pair, path


def ledger(root,r):
    p=path(root,r["index"],r["on"]).with_suffix(".ledger.json.gz")
    blob=p.read_bytes()
    check(hashlib.sha256(blob).hexdigest()==r["ledger_sha256"],"Damaged ledger")
    tape=json.loads(gzip.decompress(blob))
    tr,ph=tape["trace"],tape["physical_trace"]
    check(ph[0]["kind"]=="encoder" and not ph[0]["paid"] and all(x["paid"] for x in ph[1:]),
          "Unregistered prefill or free forward")
    origin=len(ph[0]["positions"])
    dec=[x for x in ph[1:] if x["kind"]=="decoder"]
    enc=[x for x in ph[1:] if x["kind"]=="encoder"]
    check(len(ph)-1==r["F"] and len(tr)==len(dec)==r["decoder_F"] and len(enc)==r["encoder_F"]
          and sum(len(x["positions"]) for x in enc)==r["encoder_Q"]
          and sum(len(x["positions"]) for x in dec)==r["decoder_Q"],"Physical counters differ")
    due=set()
    prev=None
    resident=None
    window=full=seals=write_count=paid_count=joint=0
    resets=Counter()
    positives=[]
    for t,(e,d) in enumerate(zip(tr,dec)):
        o=e["ordinal"]
        if e["opened"] is not None:
            check(resident is None and e["opened"]==o+1,"Wrong tail")
            resident=e["opened"]
        pos=d["positions"]
        actual=set(pos)
        sam=e["sampled_absolute"]
        check(e["offer"]==512 and pos==e["positions"]==sorted(actual) and
              e["tick"]==d["tick"]==t and e["route"]==d["phase"] and 0<len(pos)<=512
              and due==set(e["dirty_before_absolute"]) and due<=actual,"Stale or unbilled decoder work")
        paid_count+=len(due)
        wrote=set(e["written_absolute"])
        write_count+=len(wrote)
        due=(due-actual)|wrote
        check(due==set(e["dirty_after_absolute"]),"Missing post-write debt")
        head=list(range(origin+256*o,origin+256*(o+1)))
        qualified=(resident is None and e["route"]=="dense_single" and pos==sam==head)
        key=(o,origin+256*o)
        m=e["single512"]
        check(m["qualified"]==qualified and m["resident"]==1+int(resident is not None) and
              key[1]==m["epoch"] and m["applied"]==(r["on"] and m["hit"]),"Bad scope/latch")
        window+=int(qualified and prev==key)
        full+=int(qualified)
        if not qualified:
            resets[e["route"]+("_resident2" if resident is not None else "_resident1")]+=1
        if m["hit"]:
            check(qualified and prev==key,"Nonconsecutive positive in ledger")
            hp=e["parts"][0]
            masks=e["head_mask_before"]-len(hp["admitted"])+len(hp["revoked"])
            nx=tr[t+1]
            if m["applied"]:
                check(nx["positions"]==head and not nx["sampled_absolute"] and set(nx["dirty_before_absolute"])==due
                      and not nx["dirty_after_absolute"] and nx["route"] in ("verify","verify_paid")
                      and nx["parts"][0]["phase"]=="verify" and "verified_digest" in nx,
                      "Missing fresh complete paid verify after stop")
            positives.append((t,o,masks,hp["age_after"],len(due),bool(masks and hp["age_after"]<48),nx["route"]))
        seals+=int("verified_digest" in e)
        joint+=int(e["route"]=="verify_joint_paid")
        if e["encoded"] or e["discarded"]:
            resident=None
        prev=key if qualified else None
    s=r["stop"]
    check(not due and paid_count==write_count and seals==1+len(enc) and joint==len(r["joint_hits"])
          and (full,window,dict(resets))==(s["qualified"],s["pairs"],s["resets"])
          and positives==[tuple(v[k] for k in ("tick","ordinal","masks","age","dirty","actionable","next_route"))
                          for v in s["hits"]],"Final ledger or hit denominator mismatch")
    return (dict(write_rows=write_count,refresh_rows=paid_count,closed_heads=seals,joint_closes=joint),
            [e["verified_digest"] for e in tr if "verified_digest" in e])


def report(root,history,out):
    q=json.loads((root/"qualification.json").read_text())
    ids=range(90) if q["passed"] else QUAL
    rows,events,ps=[],[],{}
    reg=None
    for i in ids:
        a,b=ps[i]=pair(root,history,i)
        sig={k:a[k] for k in REG}
        check(reg in (None,sig) and q["source"]==sig["source"],"Mixed registration")
        reg=sig
        row=dict(index=i,dataset=a["dataset"],seed=a["seed"],history_exact=1,same_ids=a["ids"]==b["ids"])
        seals=[]
        for label,r in (("off",a),("on",b)):
            s=r["stop"]
            counts,digests=ledger(root,r)
            seals.append(digests)
            row.update({label+"_"+k:v for k,v in counts.items()})
            row.update({label+"_"+k:r[k] for k in (*FIELDS,"max_decoder_Q","ledger_sha256")})
            row.update({label+"_"+k:s[k] for k in ("qualified","pairs","positives","actionable","applied")})
            row[label+"_TPF"]=round(tpf(r),8)
            row[label+"_resets"]=json.dumps(s["resets"],sort_keys=True,separators=(",",":"))
            row[label+"_route_F"]=json.dumps(r["route_F"],sort_keys=True,separators=(",",":"))
            row[label+"_route_Q"]=json.dumps(r["route_Q"],sort_keys=True,separators=(",",":"))
            row[label+"_first_hit"]=next((v["tick"] for v in s["hits"]),None)
            row[label+"_first_actionable"]=next((v["tick"] for v in s["hits"] if v["actionable"]),None)
            events.extend(dict(index=i,arm=label,**v) for v in s["hits"])
        row["identical_seals"]=seals[0]==seals[1]
        row["first_different_seal"]=next((o for o,(x,y) in enumerate(zip(*seals)) if x!=y),
                                         min(map(len,seals)) if len(seals[0])!=len(seals[1]) else None)
        check(not row["identical_seals"] or row["same_ids"],"Same verified blocks, different returned IDs")
        rows.append(row)
    check(q["indices"]==list(QUAL) and q["actionable"]==sum(ps[i][1]["stop"]["actionable"] for i in QUAL)
          and q["passed"]==bool(q["actionable"]),"Bad qualification")
    summaries=[]
    for name,keys in COHORTS if q["passed"] else (("qualification",QUAL),):
        cases=[ps[i] for i in keys]
        check(len({a["task_id"] for a,b in cases})==len(keys) and
              Counter(a["dataset"] for a,b in cases)==dict.fromkeys(("gsm8k","math500","humaneval"),len(keys)//3),
              "Wrong task composition")
        ss=dict(cohort=name,matched=len(cases),expected=len(keys),missing=0)
        for j,label in enumerate(("off","on")):
            arm=[p[j] for p in cases]
            ss.update({label+"_"+k:sum(r[k] for r in arm) for k in FIELDS})
            ss.update({label+"_"+k:sum(r["stop"][k] for r in arm) for k in
                       ("qualified","pairs","positives","actionable","applied")})
            ss[label+"_actionable_cases"]=sum(bool(r["stop"]["actionable"]) for r in arm)
            ss[label+"_positive_cases"]=sum(bool(r["stop"]["positives"]) for r in arm)
            ss[label+"_macro_TPF"]=round(mean(map(tpf,arm)),6)
            ss[label+"_pooled_decoder_efficiency"]=round(sum(r["N"] for r in arm)/sum(r["decoder_Q"] for r in arm),8)
            ss[label+"_pooled_paid_efficiency"]=round(sum(r["N"] for r in arm)/sum(r["decoder_Q"]+r["encoder_Q"] for r in arm),8)
        for key,m in (("TPF",tpf),("accuracy",lambda r:int(r["correct"])),("mean_F",lambda r:r["F"])):
            ss["delta_"+key]=round(mean(m(b)-m(a) for a,b in cases),6)
            ss[key+"_lo95"],ss[key+"_hi95"]=interval(cases,m) or (None,None)
        ss.update(changed_ids=sum(a["ids"]!=b["ids"] for a,b in cases),
                  rescues=sum(b["correct"] and not a["correct"] for a,b in cases),
                  losses=sum(a["correct"] and not b["correct"] for a,b in cases))
        summaries.append(ss)
    out.mkdir(parents=True,exist_ok=True)
    table(out/"pairs.csv",rows)
    table(out/"cohorts.csv",summaries)
    if events:
        table(out/"hits.csv",events)
    (out/"provenance.json").write_text(json.dumps(dict(**reg,root=str(root),history=str(history),qualification=q,
         payload="complete physical/row/epoch ledgers remain on A800"),indent=2,sort_keys=True)+"\n")
    print(json.dumps(summaries))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("root","history","out"):
        ap.add_argument("--"+k,type=Path,required=True)
    a=ap.parse_args()
    report(a.root,a.history,a.out)
