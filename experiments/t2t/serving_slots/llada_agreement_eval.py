"""Single-factor fixed-Q64 paid proposal agreement, with a compact paired report."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import mean
import time

from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.evaluation import scoring
from bdlm_serving.methods.t2t.budget_family import Law
from .llada_close_factor import ANCHORS, SAMPLES
from .llada_fixed_family_eval import CAP, KEYS, ORDER, SCORER_SHA, evaluate_fixed, require
from .llada_seat_report import paired


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def run(a):
    import torch
    from . import llada_agreement as rule
    from bdlm_serving.methods.t2t import llada_budget_request as base
    from .cross_race_llada import load_gpu, save
    from .llada_hfull_single_factor import request_classes
    tasks=load_tasks(a.samples)
    device=os.environ["CUDA_VISIBLE_DEVICES"]
    require(torch.cuda.is_available() and torch.cuda.device_count()==1 and device.startswith("GPU-")
            and sha(a.samples)==SAMPLES and sha(Path(scoring.__file__))==SCORER_SHA and
            sha(a.control/"run.json")==a.control_sha,"missing fixed condition/control")
    prior=json.loads((a.control/"run.json").read_bytes())
    require(prior["protocol"]=="llada-warm-successor-q64-v1" and prior["q"]==64 and
            prior["samples"]==SAMPLES and prior["scorer"]==SCORER_SHA and
            prior["order"]==list(ORDER[:30]) and len(list((a.control/"rows").glob("*.json")))==30,
            "unsealed old arm")
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/"rows").mkdir()
    reference={f"{i:02d}":sha(a.control/"rows"/f"{i:02d}.json") for i in range(30)}
    sources=[Path(x) for x in (__file__,rule.__file__,base.__file__)]
    save(a.out/"run.json", dict(protocol="llada-q64-paid-repeat-v1",q=64,order=ORDER,
         samples=SAMPLES,scorer=SCORER_SHA,device=device,checkpoint=str(a.checkpoint),
         sources={p.name:sha(p) for p in sources},control=str(a.control),control_sha=a.control_sha,
         reference=reference,reference_device=prior["device"],pid=os.getpid()))
    tok, model=load_gpu(a,lambda:None)
    mask=tok.convert_tokens_to_ids("<|mask|>")
    _, control, _=request_classes()
    cases={"old":control,"repeat":rule.request_class(control)}
    exposed=0
    for j,i in enumerate(ORDER):
        ref=json.loads((a.reference/f"{i:02d}-native32.json").read_bytes())
        require(time.time()<a.deadline-60 and i==ref["index"] and ref["generation_cap"]==CAP,"stop")
        arms={}
        for k in (("old","repeat") if j%2==0 else ("repeat","old")):
            arms[k]=evaluate_fixed(model,tok,tasks[i],ref["prompt_ids"],64,Law(),
                 cross=("quarter",False),coverage="head",warm_head_sample=True,
                 audit_admission=True,request_cls=cases[k])
        old,new=arms["old"],arms["repeat"]
        first, exposure=rule.first_change(old,new,mask)
        row={k:tasks[i][k] for k in KEYS}
        for k,v in arms.items():
            require(all(row[t]==v[t] for t in KEYS) and 0<v["max_decoder_Q"]<=64 and
                    v["Q"]==v["decoder_Q"]==sum(e["q"] for e in v["trace"]) and
                    all(e["layer_physical_q"]==[e["q"]]*20 for e in v["trace"]),
                    "incorrect device or per-layer bill")
            row[k]=dict(**{t:v[t] for t in ("correct","failure","N","F","Q","max_decoder_Q")},
                        output_digest=sha256(json.dumps(v["output_ids"]).encode()).hexdigest())
        require(old["input_sha256"]==new["input_sha256"],"unmatched prompt")
        row.update(first=first,input=old["input_sha256"],**exposure)
        if i<30:
            refpath=a.control/"rows"/f"{i:02d}.json"
            require(sha(refpath)==reference[f"{i:02d}"],"historical control changed")
            refrow=json.loads(refpath.read_bytes())
            require(row["input"]==refrow["input"] and all(row[t]==refrow[t] for t in KEYS),
                    "different reference input")
            row["reference_equal"]=all(val==refrow["old"][key] for key,val in row["old"].items())
        save(a.out/"rows"/f"{i:02d}.json",row)
        exposed+=bool(first)
        print(i,first,exposure,flush=True)
        if j+1==len(ANCHORS):
            require(exposed>0 and all(json.loads((a.out/"rows"/f"{v:02d}.json").read_bytes())
                      ["reference_equal"] for v in ANCHORS),"weak pilot or unmatched control")


def report(root, manifest):
    meta=json.loads((root/"run.json").read_bytes())
    require(meta["protocol"]=="llada-q64-paid-repeat-v1" and meta["q"]==64 and
            meta["samples"]==manifest["samples_sha256"]==SAMPLES and meta["scorer"]==SCORER_SHA and
            meta["order"]==list(ORDER) and set(meta["reference"])=={f"{i:02d}" for i in range(30)},
            "invalid run")
    rows={}
    files=sorted((root/"rows").glob("*.json"))
    for p in files:
        r=json.loads(p.read_bytes())
        i=r["index"]
        require(p.name==f"{i:02d}.json" and all(r[k]==manifest["samples"][i][k] for k in KEYS) and
                bool(r["first"])==bool(r["extra_rows"]) and
                all(0<v["F"]<=v["Q"]<=64*v["F"] and 0<v["max_decoder_Q"]<=64 and
                    v["N"]>=0 and (not v["failure"] or not v["correct"]) for v in
                    (r["old"],r["repeat"])) and
                (r["first"] is not None or r["old"]==r["repeat"]),"invalid case")
        rows[i]=r
    require(set(rows)==set(ORDER[:len(rows)]),"noncanonical partial run")
    out={}
    for name, indices in (("original30",range(30)),("new60",range(30,90)),("frozen90",range(90))):
        if not set(indices)<=rows.keys():
            continue
        part=[rows[i] for i in indices]
        c=out[name]=dict(pairs=len(part),extra_rows=sum(r["extra_rows"] for r in part),
            eligible=sum(r["eligible"] for r in part),exposed=sum(bool(r["first"]) for r in part),
            original_reference_mismatches=[r["index"] for r in part if r.get("reference_equal") is False],
            rescued=sum(r["repeat"]["correct"] and not r["old"]["correct"] for r in part),
            lost=sum(r["old"]["correct"] and not r["repeat"]["correct"] for r in part),
            ci95=paired(part,"old","repeat"))
        for k in ("old","repeat"):
            vals=[r[k] for r in part]
            s={t:sum(r[t] for r in vals) for t in ("N","F","Q","correct")}
            c[k]=dict(**s,macro_TPF=mean(r["N"]/r["F"] for r in vals),TE=s["N"]/s["Q"],
                      failed=sum(bool(r["failure"]) for r in vals))
    return dict(registration=meta,finished=len(rows),cohorts=out,
                rows_sha256=sha256(b"".join(bytes.fromhex(sha(p)) for p in files)).hexdigest())


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    s=ap.add_subparsers(dest="mode",required=True)
    ev=s.add_parser("run")
    for k in ("samples","checkpoint","reference","out","control"):
        ev.add_argument("--"+k,type=Path,required=True)
    ev.add_argument("--control-sha",required=True)
    ev.add_argument("--deadline",type=int,required=True)
    re=s.add_parser("report")
    re.add_argument("root",type=Path)
    re.add_argument("manifest",type=Path)
    a=ap.parse_args()
    if a.mode=="run":
        run(a)
    else:
        print(json.dumps(report(a.root,json.loads(a.manifest.read_bytes())),indent=2,sort_keys=True))
