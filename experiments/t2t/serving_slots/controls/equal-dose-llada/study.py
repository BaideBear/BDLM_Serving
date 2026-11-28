"""Exact six only; `report` re-audits saved H4 receipts without loading a model."""
import argparse
import json
import os
import time
from itertools import zip_longest
from pathlib import Path
from statistics import mean
from unittest.mock import patch

from adapter import install
from audit import bill, fork
from tools.llada_ibs_eos_eval import BASE,CAP,CKPT,CONTROLS,DATA,KEYS,compact,digest,file_sha,read
from tools.llada_ibs_eos_stop import require
from bdlm_serving.evaluation import scoring
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.methods.optimus import fixed,upstream
from experiments.t2t.serving_slots import ibs_prefix_pair as pair,shared_llada_worker as worker

D=Path(__file__).resolve().parent
SRC=BASE/"runtime/llada-ibs-q16-paid-agreement-frozen90-h0-v1"
HIST=BASE/"results/llada-ibs-q16-paid-agreement-h0-v1"
ARMS=("stock","repeat","current")


def save(p,x):
    with p.open("x") as f: print(compact(x),file=f)


def main(mode):
    reg=read(D/"registration.json")
    s=file_sha(D/"registration.json")
    require(reg["indices"]==[7,13,25,33,55,71] and reg["arms"]==list(ARMS) and
            not reg["extension"] and reg["cap"]==CAP==2048 and reg["q"]==16,"panel")
    for path,key in ((DATA/"samples.json","samples_sha256"),(HIST/"registration.json","prior_six_registration"),
            (HIST/"receipts.ndjson","prior_six_receipts"),(SRC/"q16-source.sha256","q16_source_manifest")):
        require(file_sha(path)==reg[key],str(path))
    hreg=read(HIST/"registration.json")
    hrows={r["index"]:r for r in map(json.loads,(HIST/"receipts.ndjson").read_text().splitlines())}
    require(all(file_sha(CONTROLS/k)==v for k,v in hreg["panel"]["opt_controls"].items()) and
            set(hrows)==set(reg["indices"]),"controls")
    exc=set(hreg["excluded"])
    for arm in ARMS: install(fixed,arm,{},exc)
    if mode=="check": return
    tasks=load_tasks(DATA/"samples.json")
    label="optcap_ibs_16"
    if mode=="run":
        import torch
        u=reg["uuid"]
        require(os.environ.get("CUDA_VISIBLE_DEVICES")==u and torch.cuda.device_count()==1 and
                str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-")==u[4:],"not H4")
        require(not list(D.glob("case-*.json")) and not (D/"start.json").exists(),"refuse rerun")
        h=worker.load(CKPT)
        tok,model=h["tok"],h["model"]
        special=set(tok.all_special_ids)|{h["mask_id"],h["eos_id"]}|{
            i for i,t in tok.added_tokens_decoder.items() if t.special}|{
            t["id"] for t in read(CKPT/"tokenizer.json")["added_tokens"] if t["special"]}
        require(special==exc and h["fingerprint"]==hreg["checkpoint"] and h["scorer"]==hreg["scorer"]
                and str(torch.__version__)==hreg["torch"],"different runtime")
        save(D/"start.json",dict(reg=s,uuid=u,pid=os.getpid(),start_ticks=int(
            Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19]),checkpoint=h["fingerprint"],
            scorer=h["scorer"],excluded_sha=digest(sorted(special)),eos=h["eos_id"],mask=h["mask_id"]))
    start=read(D/"start.json")
    require(start["reg"]==s and start["uuid"]==reg["uuid"],"start receipt")
    records=[]
    with patch.object(pair,"Q",16),patch.dict(pair.LABELS,{0:label}),patch.object(upstream,"control_directory",
            lambda name:CONTROLS):
        for ordinal,i in enumerate(reg["indices"]):
            task=tasks[i]
            historical=HIST/"pairs"/f"{i:02}.json"
            require(file_sha(historical)==hrows[i]["raw_sha"],"pilot evidence")
            ref=read(historical)
            stock=ref["arms"]["stock"]
            require(all(stock[k]==task[k] for k in KEYS),"frozen task")
            order=ARMS[ordinal%3:]+ARMS[:ordinal%3]
            path=D/f"case-{i:02}.json"
            if mode=="run":
                prompt,_=worker.canonical_prompt(tok,task,stock["prompt_ids"])
                rows,tapes={},{}
                for a in order:
                    require(time.time()<reg["deadline"]-90,"deadline")
                    t=tapes[a]=dict(certificate=None,ticks=[],writes=[])
                    torch.manual_seed(task["seed"])
                    with patch.object(fixed.FixedOptimus,"run",install(fixed,a,t,exc)):
                        rows[a]=pair.generate(model,tok,task,prompt,CAP,0)
                save(path,dict(reg=s,order=order,rows=rows,tapes=tapes))
            raw=read(path)
            require(raw["reg"]==s and raw["order"]==list(order),"arm order")
            rows,tapes=raw["rows"],raw["tapes"]
            stats={}
            for a,r in rows.items():
                require(all(r[k]==task[k] for k in KEYS) and r["prompt_ids"]==stock["prompt_ids"]
                        and r["generation_cap"]==CAP,"different case")
                v=stats[a]=bill(r,tapes[a],a,eos=start["eos"],mask=start["mask"],excluded=exc)
                v["correct"]=r["status"]=="success" and bool(scoring.score(r["output_text"],task)["correct"])
            parity={a:rows[a]==ref["arms"][name] and tapes[a]==ref["tapes"][name]
                    for a,name in (("stock","stock"),("repeat","apply"))}
            require(all(parity.values()),f"{i}: historical original paid/repeat mismatch: {parity}")
            ids1,ids2=(rows[a]["output_ids"] for a in ("repeat","current"))
            first_id=next((j for j,(a,b) in enumerate(zip_longest(ids1,ids2)) if a!=b),None)
            stamp=dict(index=i,seed=task["seed"],dataset=task["dataset"],task_id=task["task_id"],
                registration=s,order=order,stats=stats,first=fork(rows,tapes),first_output_offset=first_id,
                exact_history=parity,raw_sha=file_sha(path))
            print(compact(stamp),flush=True)
            records.append(stamp)
    def summarize(rs):
        result={}
        for a in ARMS:
            rr=[x["stats"][a] for x in rs]
            sums={k:sum(r[k] for r in rr) for k in ("N","F","Q","extra","eligible","correct","paid_D","paid_Q")}
            result[a]=dict(**sums,macro_TPF=mean(r["N"]/r["F"] for r in rr),
                pooled_TE=sums["N"]/sums["Q"],failures=sum(bool(r["failure"]) for r in rr))
        return result
    done=dict(cases=reg["indices"],requests=18,sha=digest(records),extension=False,
              forks=sum(x["first"] is not None for x in records),reg=s,
              groups={"all6":summarize(records),"original3":summarize(records[:3]),"added3":summarize(records[3:])})
    if mode=="run": save(D/"complete.json",done)
    else: require(done==read(D/"complete.json"),"summary seal")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("check","run","report"))
    main(parser.parse_args().mode)
