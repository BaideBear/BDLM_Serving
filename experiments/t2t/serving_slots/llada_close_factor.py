"""Three fixed-B arms isolate terminal row coverage from one terminal T2T pass."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import mean
import time

from bdlm_serving.evaluation import scoring
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.methods.t2t.budget_family import Law
from .llada_fixed_family_eval import CAP, KEYS, ORDER, SCORER_SHA, evaluate_fixed, require


ARMS = ("unchanged","full_write","edit_once")
ANCHORS = (0,10,20)
SAMPLES = "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed"
WIDTH = 32


def proof(rows, origin):
    stamps = {}
    for arm,r in rows.items():
        seen=set()
        marks=[]
        for e in r['trace']:
            if e['phase'] not in ('close_edit_once','close_full_write'):
                continue
            block=e['block']
            positions=list(range(max(origin,WIDTH*block),min(origin+CAP,WIDTH*(block+1))))
            require(block not in seen and e['head_mask_at_plan']==0 and e['q']==len(positions)<=WIDTH
                    and e['rows']==positions and not e['cold'] and not e['mask'] and not e['admitted']
                    and set(e['dirty_before'])<=set(positions) and e['dirty_before']
                    and e['sampled']==(positions if arm=='edit_once' else [])
                    and e['phase']==('close_edit_once' if arm=='edit_once' else 'close_full_write')
                    and e['dirty_after']==e['written_absolute'], "invalid terminal event")
            seen.add(block)
            marks.append(e)
        if arm=='unchanged':
            require(not marks,"changed control")
        else:
            require(marks or r['failure'],"no terminal factor exposure")
        stamps[arm]=marks
    if stamps['full_write'] and stamps['edit_once']:
        a,b=stamps['full_write'][0],stamps['edit_once'][0]
        t=a['tick']
        keys=('rows','sampled','dirty_before','dirty_after','written_absolute','retired_blocks')
        require(b['tick']==t and t<len(rows['unchanged']['trace']) and all(
            [[e[k] for k in keys] for e in rows['unchanged']['trace'][:t]]==
            [[e[k] for k in keys] for e in rows[m]['trace'][:t]] for m in ARMS[1:]),"different common prefix")
        ctrl=rows['unchanged']['trace'][t]
        require(ctrl['rows']==ctrl['dirty_before']==a['dirty_before']==b['dirty_before'] and
                not ctrl['sampled'] and a['rows']==b['rows'] and a['block']==b['block']==ctrl['block'],
                "first contrast did not isolate coverage and sample")
    return stamps


def run(a):
    import torch
    import transformers
    from .cross_race_llada import load_gpu
    from bdlm_serving.methods.t2t import llada_budget_request
    from . import llada_fixed_family_eval
    require(torch.cuda.is_available() and torch.cuda.device_count()==1 and
            sha256(a.samples.read_bytes()).hexdigest()==SAMPLES and
            sha256(Path(scoring.__file__).read_bytes()).hexdigest()==SCORER_SHA,"unregistered environment")
    import math_verify  # Fail before launching if the frozen math scorer is unavailable.
    tasks=load_tasks(a.samples)
    require([r['index'] for r in tasks]==list(range(90)),"wrong cohort")
    device=os.environ["CUDA_VISIBLE_DEVICES"]
    require(device.startswith("GPU-"),"require a physical UUID")
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/"rows").mkdir()
    code={Path(p).name:sha256(Path(p).read_bytes()).hexdigest() for p in
          (__file__,llada_budget_request.__file__,llada_fixed_family_eval.__file__)}
    meta=dict(protocol="llada-fixed-B-close-v1",arms=ARMS,offer=WIDTH,indices=list(ORDER),
              sources=code,samples=SAMPLES,scorer=SCORER_SHA,device=device,torch=str(torch.__version__),
              transformers=transformers.__version__,checkpoint=str(a.checkpoint))
    (a.out/"run.json").write_text(json.dumps(meta,sort_keys=True)+"\n")
    tok,model=load_gpu(a,lambda:None)
    for i in ORDER:
        task=tasks[i]
        native=json.loads((a.reference/f"{i:02d}-native32.json").read_bytes())
        require(native["index"]==i and native["generation_cap"]==CAP,"wrong registered prompt")
        prompt=native["prompt_ids"]
        rows={}
        for arm in ARMS[i%3:]+ARMS[:i%3]:
            require(time.time()<a.deadline,"deadline")
            r=evaluate_fixed(model,tok,task,prompt,WIDTH,Law(),cross=("head",False),close_policy=arm)
            require(r['q']==WIDTH and r['max_decoder_Q']<=WIDTH and r['decoder_Q']==r['Q'] and
                    r['encoder_Q']==0 and all(r[k]==task[k] for k in KEYS),"unexpected arm receipt")
            rows[arm]=r
        stamps=proof(rows,len(prompt))
        replay=None
        if i in ANCHORS:
            src=a.anchors/f"c32-p1of8-{i:02d}.json"
            blob=src.read_bytes()
            ref=json.loads(blob)
            replay=sha256(blob).hexdigest()
            require(ref['law']==[1,8] and ref['samples_sha256']==SAMPLES and
                    all(ref[k]==rows['unchanged'][k] for k in (*KEYS,"input_sha256","output_ids","correct","N","F","Q")),
                    "Q32 control differs from registered 1/8")
        data={}
        for arm,r in rows.items():
            marks=stamps[arm]
            data[arm]=dict(**{k:r[k] for k in ("N","F","Q","correct","failure","max_decoder_Q")},
                ids=r["output_ids"],blocks=len(marks),terminal_Q=sum(e['q'] for e in marks),
                terminal_writes=sum(len(e['written_absolute']) for e in marks),
                terminal_edits=sum(len(e['edited']) for e in marks),terminal_revokes=sum(len(e['revoked']) for e in marks),
                phases=r["phase_F"],drain_F=sum(not e['sampled'] for e in r['trace']))
        receipt=dict(**{k:task[k] for k in KEYS},input=rows['unchanged']['input_sha256'],
                     device=device,replay_sha256=replay,arms=data)
        dest=a.out/"rows"/f"{i:02}.json"
        tmp=dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(receipt,sort_keys=True)+"\n")
        tmp.replace(dest)
        print(i,[(x,data[x]['correct'],data[x]['F'],data[x]['terminal_edits']) for x in ARMS],flush=True)


def report(root, manifest):
    meta=json.loads((root/"run.json").read_bytes())
    require(meta["samples"]==manifest["samples_sha256"]==SAMPLES and meta['scorer']==SCORER_SHA
            and meta['protocol']=='llada-fixed-B-close-v1' and meta['arms']==list(ARMS) and meta['offer']==WIDTH
            and meta["indices"]==list(ORDER),"wrong registration")
    rows={}
    for path in (root/"rows").glob("*.json"):
        r=json.loads(path.read_bytes())
        i=r["index"]
        require(i not in rows and path.name==f"{i:02}.json" and r['device']==meta['device']
                and all(r[k]==manifest['samples'][i][k] for k in KEYS)
                and bool(r['replay_sha256'])==(i in ANCHORS),"identity/replay mismatch")
        for v in r['arms'].values():
            require(0<v['F']<=v['Q']<=WIDTH*v['F'] and 0<v['max_decoder_Q']<=WIDTH and
                    len(v['ids'])>=v['N']>=0 and sum(v['phases'].values())==v['F'] and
                    v['terminal_writes']==v['terminal_edits']+v['terminal_revokes'] and
                    type(v['correct']) is bool and (not v['failure'] or not v['correct']),"invalid bill")
        rows[i]=r
    out={}
    for label,indices in [('qualification',ANCHORS),('original30',range(30)),('new60',range(30,90)),('frozen90',range(90))]:
        idx=sorted(rows.keys() & set(indices))
        c=out[label]=dict(paired=len(idx),expected=len(indices))
        if len(idx)!=len(indices):
            continue
        c['arms']={}
        for arm in ARMS:
            rs=[rows[i]['arms'][arm] for i in idx]
            z={k:sum(v[k] for v in rs) for k in ('correct','N','F','Q','blocks','terminal_Q','terminal_edits','terminal_revokes','drain_F')}
            z.update(macro_TPF=mean(v['N']/v['F'] for v in rs),token_efficiency=z['N']/z['Q'],
                     failures=sum(bool(v['failure']) for v in rs))
            c['arms'][arm]=z
        for a,b in zip(ARMS,ARMS[1:]):
            v=[(rows[i]['arms'][a],rows[i]['arms'][b]) for i in idx]
            c[a+'_to_'+b]=dict(rescues=sum(not x['correct'] and y['correct'] for x,y in v),
                    losses=sum(x['correct'] and not y['correct'] for x,y in v),
                    changed_ids=sum(x['ids']!=y['ids'] for x,y in v))
    return dict(registration=meta,cohorts=out)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    modes=ap.add_subparsers(dest="mode",required=True)
    ev=modes.add_parser("run")
    for name in ("out","reference","anchors","samples","checkpoint"):
        ev.add_argument("--"+name,type=Path,required=True)
    ev.add_argument("--deadline",type=int,required=True)
    agg=modes.add_parser("report")
    agg.add_argument("root",type=Path)
    agg.add_argument("manifest",type=Path)
    args=ap.parse_args()
    if args.mode=="run":
        run(args)
    else:
        print(json.dumps(report(args.root,json.loads(args.manifest.read_bytes())),indent=2,sort_keys=True))
