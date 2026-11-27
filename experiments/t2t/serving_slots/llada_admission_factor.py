"""Does M2T admission, without changing the row planner, explain the LLaDA gap?"""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import mean
import time

from bdlm_serving.evaluation import scoring
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.methods.t2t.budget_family import Law
from .llada_close_factor import ANCHORS, SAMPLES
from .llada_fixed_family_eval import CAP, KEYS, ORDER, SCORER_SHA, evaluate_fixed, require


def decisions(e, mask_id, cutoff, warm=False):
    require(cutoff in ("high","low","warm_head"),"unregistered cutoff")
    sampled, props = e["sampled"], e.get("proposals", [])
    require(len(sampled)==len(props) and len(set(sampled))==len(sampled),"unmeasured proposal")
    groups, natural, fallback, edits, revokes = {}, set(), set(), set(), set()
    for p,(old,draft,confidence,lo,hi) in zip(sampled,props):
        require(not hi or lo,"invalid threshold ordering")
        if old==mask_id:
            groups.setdefault(p//32,[]).append((p,confidence,draft))
            relaxed = cutoff=="low" or (cutoff=="warm_head" and warm and p//32==e["block"])
            if (lo if relaxed else hi):
                natural.add(p)
        elif old!=draft and lo:
            (revokes if draft==mask_id else edits).add(p)
    for group in groups.values():
        if not any(p in natural for p,_,_ in group):
            fallback.add(max(group,key=lambda v:v[1])[0])
    real={p for group in groups.values() for p,_,draft in group if draft!=mask_id}
    hits,forced=natural & real, fallback & real
    require(hits|forced==set(e["admitted"]) and edits==set(e["edited"])
            and revokes==set(e["revoked"]),"wrong first-stage or fallback decision")
    return Counter(natural=len(hits),fallback=len(forced),fallback_groups=len(fallback),
                   edit=len(edits),revoke=len(revokes))


def paired(high,low,mask_id,trial="low"):
    ledger={}
    for cutoff,r in [('high',high),(trial,low)]:
        c=Counter()
        origin=r.get("prefill_Q",0)
        seen=set(range(origin))
        for e in r['trace']:
            base=32*e["block"]
            # Reconstruct from earlier paid physical projections only, not the planner flag.
            warm=all(p in seen for p in range(max(origin,base),min(origin+CAP,base+32)))
            if cutoff=="warm_head":
                require(e["admission_head_warm_before"]==warm,"incorrect pre-forward warm state")
            c.update(decisions(e,mask_id,cutoff,warm))
            seen.update(e["rows"])
        require(c['natural']+c['fallback']==r['row_visits']['m2t'],"admission bill mismatch")
        ledger[cutoff]=dict(c)
    first=None
    for a,b in zip(high['trace'],low['trace']):
        require(all(a.get(k)==b.get(k) for k in ('tick','block','rows','sampled','proposals',
                    'dirty_before','edited','revoked')),"trajectory changed before admission")
        old,new=set(a['admitted']),set(b['admitted'])
        if old!=new:
            require(old < new and set(b['dirty_after'])-set(a['dirty_after'])==new-old,
                    "first difference is not additional measured M2T")
            first=dict(tick=a['tick'],Q=a['q'],high=len(old),low=len(new),more=len(new-old))
            if trial=="warm_head":
                require(b["admission_head_warm_before"] and all(p//32==b["block"] for p in new-old),
                        "intervention outside warm head")
            break
        require(all(a[k]==b[k] for k in ('written_absolute','dirty_after','retired_blocks')),
                "trajectory changed with identical token writes")
    if first is None:
        require(all(high[k]==low[k] for k in ('output_ids','F','Q','correct','failure')),
                "outputs differed without an admission intervention")
    return first,ledger


def run(a):
    import torch
    import math_verify
    import transformers
    from bdlm_serving.methods.t2t import llada_budget_request
    from . import llada_fixed_family_eval as fixed
    from .cross_race_llada import load_gpu
    require(torch.cuda.is_available() and torch.cuda.device_count()==1
            and sha256(a.samples.read_bytes()).hexdigest()==SAMPLES
            and sha256(Path(scoring.__file__).read_bytes()).hexdigest()==SCORER_SHA,"environment")
    tasks=load_tasks(a.samples)
    require([r['index'] for r in tasks]==list(range(90)),"unexpected cohort")
    device=os.environ["CUDA_VISIBLE_DEVICES"]
    require(device.startswith("GPU-"),"use a physical device")
    prior=None
    if a.trial=="warm_head":
        require(a.paired_history is not None,"supply same-device high/low receipts")
        prior=json.loads((a.paired_history/"run.json").read_bytes())
        require(prior["offer"]==a.offer and prior["device"]==device and prior["samples"]==SAMPLES
                and prior["scorer"]==SCORER_SHA and prior["protocol"]=="llada-admission-v1"
                and prior["checkpoint"]==str(a.checkpoint),"wrong same-device control")
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/"rows").mkdir()
    protocol="llada-admission-v1" if a.trial=="low" else "llada-warm-admission-v1"
    meta=dict(protocol=protocol,offer=a.offer,samples=SAMPLES,scorer=SCORER_SHA,
              device=device,checkpoint=str(a.checkpoint),order=ORDER,torch=str(torch.__version__),
              transformers=transformers.__version__,sources={Path(p).name:sha256(Path(p).read_bytes()).hexdigest()
                    for p in (__file__,llada_budget_request.__file__,fixed.__file__)})
    if prior is not None:
        meta["paired_history"]=str(a.paired_history)
        meta["control_registration"]=prior["sources"]
    (a.out/"run.json").write_text(json.dumps(meta,sort_keys=True)+"\n")
    def ready():
        if a.handoff is not None:
            (a.handoff/'ready').touch(exist_ok=False)
            while not (a.handoff/'go').exists():
                require(time.time()<a.deadline,"handoff expired")
                time.sleep(.2)
    tok,model=load_gpu(a,ready)
    for serial,i in enumerate(ORDER):
        task=tasks[i]
        native=json.loads((a.reference/f"{i:02d}-native32.json").read_bytes())
        require(native['index']==i and native['generation_cap']==CAP,"wrong native prompt")
        q=a.offer
        opts=dict(cross=("head",False)) if q<=32 else dict(cross=("quarter",False),coverage="head")
        rows={}
        arms=("high",a.trial) if serial%2==0 else (a.trial,"high")
        for arm in arms:
            require(time.time()<a.deadline,"deadline")
            rows[arm]=evaluate_fixed(model,tok,task,native["prompt_ids"],q,Law(),**opts,
                                     admission=.5 if arm=="low" else .9, audit_admission=True,
                                     admission_scope="warm_head" if arm=="warm_head" else "global")
        high,low=rows['high'],rows[a.trial]
        first,counts=paired(high,low,tok.convert_tokens_to_ids("<|mask|>"),a.trial)
        anchor=None
        if i in ANCHORS:
            path=a.history/{16:f"rows/c16-noborrow-{i:02d}.json",
                            32:f"llada-low-dose-a800-gpu2/rows/c32-p1of8-{i:02d}.json",
                            64:f"llada-hfull-paired/rows/c64-full-quarter-{i:02d}.json"}[q]
            blob=path.read_bytes()
            ref=json.loads(blob)
            require(ref['samples_sha256']==SAMPLES and ref['law']==[1,8] and all(ref[k]==high[k]
                    for k in (*KEYS,'output_ids','input_sha256','correct','N','F','Q')),"control replay failed")
            anchor=sha256(blob).hexdigest()
        out=dict(**{k:task[k] for k in KEYS},offer=q,device=device,replay=anchor,
                 input=high['input_sha256'],first=first,arms={})
        for arm,r in rows.items():
            require(all(r[k]==task[k] for k in KEYS) and r["input_sha256"]==high["input_sha256"]
                    and 0<r['max_decoder_Q']<=q and r['Q']==r['decoder_Q'] and r['encoder_Q']==0,"wrong bill")
            out['arms'][arm]=dict(**{k:r[k] for k in ("N","F","Q","correct","failure","max_decoder_Q")},
                                  ids=sha256(json.dumps(r['output_ids']).encode()).hexdigest(),**counts[arm])
        if prior is not None:
            blob=(a.paired_history/"rows"/f"{i:02d}.json").read_bytes()
            old=json.loads(blob)
            require(all(old[k]==out[k] for k in (*KEYS,"input","offer","device","replay"))
                    and old['arms']['high']==out['arms']['high'],"high cutoff did not reproduce")
            out['control_sha256']=sha256(blob).hexdigest()
        path=a.out/"rows"/f"{i:02d}.json"
        tmp=path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out,sort_keys=True)+"\n")
        tmp.replace(path)
        print(i,q,[(k,rows[k]['correct'],rows[k]['F']) for k in ("high",a.trial)],first,flush=True)
        if a.trial=="warm_head" and serial+1==len(ANCHORS):
            anchors=[json.loads((a.out/"rows"/f"{j:02d}.json").read_bytes()) for j in ANCHORS]
            require(all(r['replay'] for r in anchors) and any(r['first'] for r in anchors),
                    "No qualified warm-head exposure; do not expand")


def report(root,manifest):
    meta=json.loads((root/"run.json").read_bytes())
    q=meta["offer"]
    trial="warm_head" if meta['protocol']=="llada-warm-admission-v1" else "low"
    require(q in (16,32,64) and meta['protocol'] in ('llada-admission-v1','llada-warm-admission-v1')
            and meta['order']==list(ORDER) and meta['samples']==manifest['samples_sha256']==SAMPLES
            and meta['scorer']==SCORER_SHA,"wrong registration")
    rows={}
    for p in (root/"rows").glob("*.json"):
        r=json.loads(p.read_bytes())
        i=r['index']
        require(i not in rows and p.name==f"{i:02d}.json" and all(
            r[k]==manifest['samples'][i][k] for k in KEYS) and r['offer']==q and r["device"]==meta["device"]
            and bool(r['replay'])==(i in ANCHORS),"wrong identity")
        require(set(r['arms'])=={'high',trial},"wrong paired arms")
        if trial=="warm_head":
            require(len(r["control_sha256"])==64 and bool(meta.get("paired_history")),"missing exact control")
        for a in r['arms'].values():
            require(0<a['F']<=a['Q']<=a['F']*q and 0<a['max_decoder_Q']<=q and a['N']>=0
                    and type(a['correct']) is bool and not (a['failure'] and a['correct']),"incorrect bill")
        rows[i]=r
    cohorts={}
    for name,ix in [('qualification',ANCHORS),('original30',range(30)),('new60',range(30,90)),('frozen90',range(90))]:
        subset=[rows[i] for i in ix if i in rows]
        c=cohorts[name]=dict(expected=len(ix),paired=len(subset))
        if len(subset)!=len(ix):
            continue
        for arm in ('high',trial):
            cases=[r['arms'][arm] for r in subset]
            s={k:sum(x[k] for x in cases) for k in ('N','F','Q','correct','natural','fallback','edit','revoke')}
            s.update(macro_TPF=mean(x['N']/x['F'] for x in cases),TE=s['N']/s['Q'],
                     failed=sum(bool(x['failure']) for x in cases))
            c[arm]=s
        c['first_extra_admissions']=sum(r['first']['more'] for r in subset if r['first'])
        c['intervened']=sum(r['first'] is not None for r in subset)
        c['rescues']=sum(r['arms'][trial]['correct'] and not r['arms']['high']['correct'] for r in subset)
        c['losses']=sum(r['arms']['high']['correct'] and not r['arms'][trial]['correct'] for r in subset)
    return dict(registration=meta,cohorts=cohorts)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    modes=ap.add_subparsers(dest="mode",required=True)
    ev=modes.add_parser("run")
    for name in ("out","reference","history","samples","checkpoint"):
        ev.add_argument("--"+name,type=Path,required=True)
    ev.add_argument("--offer",choices=(16,32,64),type=int,required=True)
    ev.add_argument("--trial",choices=("low","warm_head"),default="low")
    ev.add_argument("--paired-history",type=Path)
    ev.add_argument("--handoff",type=Path)
    ev.add_argument("--deadline",type=int,required=True)
    agg=modes.add_parser("report")
    agg.add_argument("root",type=Path)
    agg.add_argument("manifest",type=Path)
    a=ap.parse_args()
    if a.mode=="run":
        run(a)
    else:
        print(json.dumps(report(a.root,json.loads(a.manifest.read_bytes())),sort_keys=True,indent=2))
