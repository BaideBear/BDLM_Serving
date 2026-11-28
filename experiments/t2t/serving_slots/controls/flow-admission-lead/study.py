"""H4 C/N/L study, using the sealed compact W2 collector without changing it."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

D=Path(__file__).resolve().parent
ROOT=Path("/data1/project_data/gx-erqw-20261123")
BASE=ROOT/"results/llada-flow-active-w2-h4-v1"
sys.path.insert(0,str(BASE))
import llada_flow_active_w2_study as old
from lead import factory
from bdlm_serving.methods import flowblock
flowblock.__path__.insert(0,str(BASE))
from bdlm_serving.methods.flowblock import control, llada_active_w2 as module
from experiments.t2t.serving_slots import shared_fixed_shard as shard, shared_llada_worker as worker
from audit_raw import MASK,EOS,B,CAP,verify

check,read,save,js,digest=old.check,old.read,old.save,old.js,old.hashed
ARMS=("control","noop","lead")


def plain(t):
    return {k:v for k,v in t.items() if k!="lead_vote"}


def compare_same(a,b):
    keys=("index","seed","arm","task_id","dataset","official","official_return_length","upstream_sources",
          "prompt_ids","output_ids","output_text","N","eos","remaining_masks","layers","calls")
    check(all(a[k]==b[k] for k in keys) and [plain(t) for t in a["ticks"]]==[plain(t) for t in b["ticks"]]
          and a["events"][1:]==b["events"][1:],"fresh control/noop or historical mismatch")


def validate(r,arm):
    """Independent gate reconstruction from two adjacent *recorded* samples."""
    p=len(r["prompt_ids"])
    first=p//B*B
    blocks=((p+CAP+B-1)//B*B-first)//B
    ticks=r["ticks"]
    votes=[]
    for j,t in enumerate(ticks):
        v=t.get("lead_vote")
        # Only a still-current sampled single head can be forecasted.
        prev=ticks[j-1] if j else None
        s=t["slots"][0]
        expect=(prev is not None and prev["open"]==t["open"]==1 and prev["commit"]==t["commit"]
                and not prev["slots"][0]["fin"] and not s["fin"] and t["right"]<blocks and
                j+1<len(ticks) and ticks[j+1]["commit"]==t["commit"])
        if expect:
            f=s["frontier"]
            last=prev["slots"][0]["frontier"]
            expect=f<.8<=f+max(0.,f-last)
        check(bool(v)==(arm!="control" and expect),"forecast opportunity mismatch")
        if v:
            check(v==dict(tick=j,head=t["commit"],prev_f=last,f=f,opened=(arm=="lead")),
                  "recorded frontier mismatch")
            n=ticks[j+1]
            check(n["open"]==(2 if arm=="lead" else 1) and n["physical_width"]==B*n["open"],
                  "forecast did not map to next physical call")
            votes.append(dict(tick=j,head=t["head"],prev_f=last,f=f,following_width=n["physical_width"]))
    return votes


def audit_lead_canvas(r):
    """Reconstruct the native lifecycle, replacing only its below-.8 deferral."""
    from collections import Counter
    prompt=r["prompt_ids"]
    first=len(prompt)//B*B
    length=(len(prompt)+CAP+B-1)//B*B
    canvas=prompt+[MASK]*(length-len(prompt))
    blocks=(length-first)//B
    cp,rp=0,1
    done=set()
    steps,frontiers,post=Counter(),Counter(),Counter()
    ended=False
    for t in r["ticks"]:
        check(not ended and (cp,rp)==(t["commit"],t["right"]),"unauthorized admission/commit")
        for g,s in enumerate(t["slots"],cp):
            lo=first+g*B
            sl=slice(lo,lo+B)
            imm=[k<len(prompt) for k in range(lo,lo+B)]
            check(s["ids"]==canvas[sl] and s["prompt"]==imm and s["pe0"]==post[g] and
                  s["fin0"]==(g in done) and all(not y or a==b for y,a,b in
                  zip(imm,s["ids"],s["new"],strict=True)) and (g not in done or s["ids"]==s["new"]),
                  "token/prompt/sticky state")
            steps[g]+=1
            frontiers[g],post[g]=s["frontier"],s["pe"]
            if s["fin"] or steps[g]>=B+16+1:
                done.add(g)
            canvas[sl]=s["new"]
        while cp in done:
            lo=first+cp*B
            cp+=1
            if EOS in canvas[lo:lo+B]:
                canvas[lo+B:]=[EOS]*(length-lo-B)
                ended=True
                break
            if cp==blocks:
                ended=True
                break
        while not ended and rp-cp<2 and rp<blocks:
            if rp>cp and frontiers[rp-1]<.8 and not t.get("lead_vote",{}).get("opened"):
                break
            rp+=1
    out=canvas[len(prompt):len(prompt)+CAP]
    if EOS in out:
        out=out[:out.index(EOS)+1]
    check(ended and out==r["output_ids"],"unexplained EOS/output")


def compare_treatment(c,l,votes):
    if not votes:
        compare_same(c,l)
        return None
    t=votes[0]["tick"]
    check(t+1<len(c["ticks"]) and all(plain(c["ticks"][j])==plain(l["ticks"][j]) for j in range(t+1)),
          "first difference precedes admission")
    a,b=(x["ticks"][t+1] for x in (c,l))
    check(a["commit"]==b["commit"] and a["head"]==b["head"] and a["open"]==1 and b["open"]==2 and
          all(a["slots"][0][k]==b["slots"][0][k] for k in ("ids","prompt","fin0","pe0")) and
          all(v==MASK for v in b["slots"][1]["ids"]) and not any(b["slots"][1]["prompt"]),
          "first differing call is not admission-only")
    return votes[0]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("check","run","report"))
    mode=p.parse_args().mode
    reg=read(D/"registration.json")
    tag=sha256((D/"registration.json").read_bytes()).hexdigest()
    for f,k in [("llada_active_w2.py","parent_adapter"),("llada_flow_active_w2_study.py","parent_collector")]:
        check(sha256((BASE/f).read_bytes()).hexdigest()==reg[k],"parent changed")
    check(read(BASE/"complete.json")["reg"]==reg["parent_registration"] and
          reg["offline_gate"]["distinct_requests"]>=10,"offline gates")
    shard.check_sources(read(old.DEPLOY/"registration.json"))
    tasks=shard.load_tasks(old.DATA/"samples.json")
    order=reg["pilot"]+[i for i in range(90) if i not in reg["pilot"]]
    original=module.ActiveW2
    classes={a:original if a=="control" else factory(original,control,a=="lead") for a in ARMS}
    controls=ROOT/"sources/flowblock-controls"
    if mode=="check":
        from types import SimpleNamespace
        with patch.object(control,"control_directory",lambda _:controls):
            for cls in classes.values():
                cls(SimpleNamespace(device=None),MASK,EOS,dict(confidence_threshold=.5,
                    editing_threshold=.5,max_post_steps=16),compact=True)
        print("sealed source, panel, three executable policies: OK")
        return
    if mode=="run":
        check(not list(D.glob("??-*.json")), "refusing duplicate run")
        shard.check_board(reg["board"])
        handle=worker.load(Path("/data0/ysy/models/LLaDA2.1-mini"))
    rows=[]
    with patch.object(control,"control_directory",lambda _:controls):
        for k,i in enumerate(order):
            task=tasks[i]
            ref=read(BASE/f"{i:02d}-active_w2.json")
            data,stats,votes={},{},{}
            for arm in ARMS[k%3:]+ARMS[:k%3]:
                path=D/f"{i:02d}-{arm}.json"
                if mode=="run":
                    check(time.time()<reg["deadline"]-90,"deadline")
                    ids,_=worker.canonical_prompt(handle["tok"],task,ref["prompt_ids"])
                    with patch.object(module,"ActiveW2",classes[arm]):
                        r=old.one(handle,task,ids,"active_w2")
                    r.update(study_arm=arm,registration=tag)
                    save(path,r)
                else:
                    r=read(path)
                check(r["study_arm"]==arm and r["registration"]==tag and r["prompt_ids"]==ref["prompt_ids"]
                      and all(r[n]==ref[n] for n in ("upstream_sources","seed","task_id","index","dataset")),
                      "identity mismatch")
                data[arm],stats[arm],votes[arm]=r,old.audit(r,task),validate(r,arm)
                if arm=="lead":
                    audit_lead_canvas(r)
                else:
                    verify(r)
                    compare_same(r,ref)
            c,n,l=(data[a] for a in ARMS)
            compare_same(c,n)
            first=compare_treatment(c,l,votes["lead"])
            if first:
                check(votes["noop"] and all(first[w]==votes["noop"][0][w] for w in ("tick","head","prev_f","f")),
                      "unexplained first trigger")
            row=dict(index=i,task_id=task["task_id"],seed=task["seed"],dataset=task["dataset"],
                     registration=tag,order=list(ARMS[k%3:]+ARMS[:k%3]),**stats,
                     reference_replayed=True,noop_exact=True,lead_votes=votes["lead"],noop_votes=votes["noop"],
                     first=first,ids_equal=c["output_ids"]==l["output_ids"])
            rows.append(row)
            print(js(row),flush=True)
            if k==5:
                check(any(z["first"] for z in rows),"no exposed qualification; do not extend")
                if mode=="run":
                    save(D/"qualified.json",dict(cases=order[:6],sha=digest(rows),registration=tag))
    if mode=="run":
        save(D/"complete.json",dict(cases=90,requests=270,sha=digest(rows),registration=tag))


if __name__=="__main__":
    main()
