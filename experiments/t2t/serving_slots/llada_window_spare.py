"""Test otherwise idle Q on a fixed window: interior KV or a waiting MASK."""
import argparse
from dataclasses import replace
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
from .llada_seat_report import paired
from .llada_window_geometry import window_class


def spare(plan, sampled, masks, resident, hi, q):
    if not sampled or not masks or plan.drain_only or len(plan.rows)>=q:
        return plan,()
    pool=(resident & set(range(min(masks)+1,hi)))-masks-set(plan.rows)
    extra=tuple(sorted(pool)[:q-len(plan.rows)])
    return replace(plan,rows=tuple(sorted((*plan.rows,*extra)))),extra


def waiting(plan,sampled,masks,resident,hi,q):
    if plan.drain_only or not plan.cold or not sampled or len(plan.rows)>=q:
        return plan,sampled,()
    candidates=sorted((masks & resident)-set(plan.rows)-set(plan.dirty))
    if not candidates:
        return plan,sampled,()
    p=candidates[0]
    require(p<hi and p//32==plan.cold[0]//32,"not a resident waiting MASK in this head")
    return (replace(plan,rows=tuple(sorted((*plan.rows,p))),mask=tuple(sorted((*plan.mask,p)))),
            tuple(sorted((*sampled,p))),(p,))


def prefix_spare(plan,sampled,prefix,q):
    if not sampled or plan.drain_only:
        return plan,()
    extra=tuple(p for p in prefix if p not in plan.rows)[:min(2,q-len(plan.rows))]
    return replace(plan,rows=tuple(sorted((*plan.rows,*extra)))),extra


def with_spare(control,factor="interior"):
    class ReadInterior(control):
        def _cross_plan(self,q):
            p,s,phase,cap=super()._cross_plan(q)
            self.extra=()
            if s and len(p.rows)<q:
                require(self.cross=="head" and not self.borrow and not self.stopping,"not an active single head")
                masks,prefix,_=self._inputs(32)
                hi=min(self.head+32,self.end)
                if factor=="waiting":
                    p,s,self.extra=waiting(p,s,masks,self.bank.initialized,hi,q)
                elif factor=="prefix":
                    p,self.extra=prefix_spare(p,s,prefix,q)
                else:
                    p,self.extra=spare(p,s,masks,self.bank.initialized,hi,q)
            return p,s,phase,cap

        def _step(self,q):
            e=super()._step(q)
            if self.extra:
                e["spare_"+("read" if factor=="interior" else factor)]=list(self.extra)
            return e
    return ReadInterior


def analyze(a,b,q,*,factor="interior",mask_id=None):
    first=None
    calls=extra_q=accepted=0
    key="spare_"+("read" if factor=="interior" else factor)
    decoded=set()
    for i,y in enumerate(b["trace"]):
        extra=y.get(key,[])
        if extra:
            extra_q+=len(extra)
            calls+=1
            before=y["pre_state"]
            require(len(set(extra))==len(extra) and set(extra)<=set(y["rows"])
                    and set(extra).isdisjoint(y["cold"]+y["dirty"]+y["prefix"])
                    and max(extra)<min(before["last_end"],before["head"]+32) and min(extra)>=before["head"]
                    and 0<y["q"]-len(extra)<y["q"]<=q,"extra row not a spare resident read")
            if factor=="waiting":
                require(len(extra)==1 and y["cold"] and set(extra)<=set(y["mask"]) & set(y["sampled"])
                        and mask_id is not None and len(y["sampled"])==len(y["proposals"]) and
                        y["proposals"][y["sampled"].index(extra[0])][0]==mask_id,"not a paid unresolved row")
                accepted+=int(extra[0] in y["admitted"])
            else:
                require(set(extra).isdisjoint(y["sampled"]+y["mask"]),"resident read altered sampling")
                if factor=="prefix":
                    lo=max(before["head"],before["origin"]); hi=min(before["head"]+32,before["end"])
                    front=next((r for r in range(lo,hi) if r not in decoded),None)
                    candidates=() if front is None else tuple(p for p in range(front-1,lo-1,-1)
                        if p<before["last_end"] and p not in set(y["rows"])-set(extra))
                    require(tuple(extra)==candidates[:min(2,q-y["q"]+len(extra))],
                            "not the exact right edge of the physical decoded prefix")
        if factor=="prefix":
            decoded.update(y["admitted"]); decoded.difference_update(y["revoked"])
        if first is None:
            require(i<len(a["trace"]),"unexplained extra call")
            x=a["trace"][i]
            if not extra:
                require(x==y,"changed before extra read")
                continue
            added=set(extra) if factor=="waiting" else set()
            require(set(y["rows"])==set(x["rows"])|set(extra) and set(extra).isdisjoint(x["rows"])
                    and all(set(y[k])==set(x[k])|added for k in ("sampled","mask"))
                    and all(x[k]==y[k] for k in ("pre_state","phase","cold","dirty",
                                                 "prefix","visible_kv","sample_clock"))
                    and x["layer_physical_q"]==[x["q"]]*20 and y["layer_physical_q"]==[y["q"]]*20,
                    "first read changed plan or sampler")
            first=dict(tick=i,added=extra,old_q=x["q"],new_q=y["q"])
    if first is None:
        require(len(a["trace"])==len(b["trace"]) and all(a[k]==b[k] for k in
                ("output_ids","score","failure","Q","N","F")),"unexposed outcome changed")
    return dict(first=first,calls=calls,extra_Q=extra_q,**({"extra_admits":accepted} if factor=="waiting" else {}))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def run(a):
    import torch
    from .cross_race_llada import load_gpu,save
    from .llada_hfull_single_factor import request_classes
    from . import llada_window_geometry as geometry
    refmeta=json.loads((a.parent/"run.json").read_bytes())
    uuid=os.environ["CUDA_VISIBLE_DEVICES"]
    gb=Path(geometry.__file__).read_bytes()
    extra_mask=a.factor=="waiting"; interior_swap=a.factor=="interior_swap"
    head_orig=a.factor=="prefix"; h0=extra_mask or head_orig or interior_swap
    protocol="llada-window-paid-mask-v1" if h0 else "llada-fixed-width-window-v1"
    contested=a.factor=="contested" or interior_swap
    old,new=("old","swap") if interior_swap else ("paid","waiting") if extra_mask else ("old","prefix") if head_orig else ("window","swap" if contested else "read")
    # Older controls differ only by not entering this branch at Q4.
    hashes={sha256(b).hexdigest() for b in (gb,gb.replace(b"q in (4,8,16)",b"q in (8,16)"))}
    require(a.q in (8,16) and torch.cuda.is_available() and torch.cuda.device_count()==1
            and uuid.startswith("GPU-") and digest(a.samples)==SAMPLES and digest(Path(scoring.__file__))==SCORER_SHA
            and refmeta["device"]==uuid and refmeta["samples"]==SAMPLES and refmeta["scorer"]==SCORER_SHA
            and refmeta["protocol"]==protocol and refmeta["geometry_sha256"] in hashes
            and refmeta["q"]==a.q and refmeta["checkpoint"]==str(a.checkpoint) and refmeta["order"]==list(ORDER)
            and sorted(p.name for p in (a.parent/"rows").glob("*.json"))==[f"{i:02}.json" for i in range(90)],
            "missing exact completed fixed-window control")
    tasks=load_tasks(a.samples)
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/"rows").mkdir()
    reg=dict(protocol="llada-window-spare-"+("read" if a.factor=="interior" else a.factor)+"-v1",q=a.q,device=uuid,order=ORDER,
             samples=SAMPLES,scorer=SCORER_SHA,source=digest(Path(__file__)),
             parent=str(a.parent),parent_sha=digest(a.parent/"run.json"),
             geometry=sha256(gb).hexdigest(),parent_geometry=refmeta["geometry_sha256"],pid=os.getpid())
    control=window_class(request_classes()[1])
    if h0:
        from . import llada_window_sample as paid
        sealed=json.loads(Path(__file__).with_name("controls").joinpath("llada-slot-allocation","c8-paid-frozen90.json").read_bytes())
        require(a.q==8 and refmeta==sealed["registration"] and digest(Path(paid.__file__))==refmeta["sampling_sha256"]
                and sha256(b"".join(bytes.fromhex(digest(p)) for p in sorted((a.parent/"rows").glob("*.json")))).hexdigest()==sealed["rows_sha256"],
                "not the sealed same-device paid sampler")
        control=paid.variants(request_classes()[1])[int(extra_mask)]
        reg.update(sampling=refmeta["sampling_sha256"],parent_rows=sealed["rows_sha256"])
    if contested:
        from . import llada_contested
        reg["exchange_source"]=digest(Path(llada_contested.__file__))
    save(a.out/"run.json",reg)
    def ready():
        if a.handoff is not None:
            (a.handoff/"ready").touch(exist_ok=False)
            while not (a.handoff/"go").exists():
                require(time.time()<a.deadline,"handoff expired")
                time.sleep(.2)
    tok,model=load_gpu(a,ready)
    mask_id=tok.convert_tokens_to_ids("<|mask|>") if extra_mask or contested else None
    arms={old:control,new:llada_contested.variant(control,interior=interior_swap) if contested else with_spare(control,a.factor)}
    exposed=0
    for j,i in enumerate(ORDER):
        require(time.time()<a.deadline-60,"stop")
        ref=json.loads((a.reference/f"{i:02d}-native32.json").read_bytes())
        require(ref["index"]==i and ref["generation_cap"]==CAP,"wrong input")
        res={k:evaluate_fixed(model,tok,tasks[i],ref["prompt_ids"],a.q,Law(),cross=("head",False),
                              request_cls=arms[k],audit_admission=extra_mask or contested) for k in (tuple(arms) if j%2==0 else tuple(reversed(arms)))}
        r=(llada_contested.analyze(res[old],res[new],a.q,mask_id,interior=interior_swap) if contested else
           analyze(res[old],res[new],a.q,factor=a.factor,mask_id=mask_id))
        orig_path=a.parent/"rows"/f"{i:02d}.json"
        orig=json.loads(orig_path.read_bytes())
        for k,v in res.items():
            require(all(tasks[i][t]==v[t]==orig[t] for t in KEYS) and v["input_sha256"]==orig["input"]
                    and 0<v["max_decoder_Q"]<=a.q and v["Q"]==sum(e["q"] for e in v["trace"]),"wrong bill/input")
            r[k]={t:v[t] for t in ("N","Q","F","failure","correct","max_decoder_Q")}
            r[k]["output_digest"]=sha256(json.dumps(v["output_ids"]).encode()).hexdigest()
        require(all(orig[old][k]==v for k,v in r[old].items()),"same-device window did not reproduce")
        r.update({k:tasks[i][k] for k in KEYS},input=orig["input"],reference_sha=digest(orig_path))
        save(a.out/"rows"/f"{i:02d}.json",r)
        exposed+=bool(r["first"])
        print(i,r["first"],r["calls"],r["extra_Q"],flush=True)
        if j+1==len(ANCHORS):
            require(exposed>0,"no genuine test in three unchanged anchors")


def report(root,manifest):
    reg=json.loads((root/"run.json").read_bytes())
    extra_mask=reg["protocol"]=="llada-window-spare-waiting-v1"
    head_orig=reg["protocol"]=="llada-window-spare-prefix-v1"
    interior_swap=reg["protocol"]=="llada-window-spare-interior_swap-v1"
    contested=reg["protocol"]=="llada-window-spare-contested-v1" or interior_swap
    old,new=("old","swap") if interior_swap else ("paid","waiting") if extra_mask else ("old","prefix") if head_orig else ("window","swap" if contested else "read")
    if interior_swap:
        require(reg["q"]==8 and "parent_rows" in reg and "exchange_source" in reg,"invalid Q8 swap parent")
    require((reg["protocol"]=="llada-window-spare-read-v1" or contested or (extra_mask or head_orig) and reg["q"]==8)
            and reg["q"] in (8,16) and reg["order"]==list(ORDER)
            and reg["samples"]==SAMPLES==manifest["samples_sha256"] and reg["scorer"]==SCORER_SHA,"wrong run")
    rows={}
    h=sha256()
    for p in sorted((root/"rows").glob("*.json")):
        blob=p.read_bytes()
        h.update(sha256(blob).digest())
        r=json.loads(blob);i=r["index"]
        require(p.name==f"{i:02d}.json" and all(r[t]==manifest["samples"][i][t] for t in KEYS)
                and bool(r["first"])==bool(r["swapped_seats"] if contested else r["extra_Q"])==bool(r["calls"])
                and (not contested or r["extra_Q"]==0 and 0<=r["mask_admits"]<=r["calls"]==r["swapped_seats"])
                and (r["first"] or r[old]==r[new]) and
                (not extra_mask or 0<=r["extra_admits"]<=r["calls"]==r["extra_Q"])
                and all(0<v["F"]<=v["Q"]<=reg["q"]*v["F"] and 0<v["max_decoder_Q"]<=reg["q"]
                        and (not v["failure"] or not v["correct"]) for v in (r[old],r[new])),"wrong receipt")
        rows[i]=r
    require(set(rows)==set(ORDER[:len(rows)]),"unexpected task slots")
    cohorts={}
    for name,ids in (("original30",range(30)),("new60",range(30,90)),("frozen90",range(90))):
        if not set(ids)<=rows.keys(): continue
        part=[rows[i] for i in ids]
        c=cohorts[name]=dict(n=len(part),exposed=sum(bool(r["first"]) for r in part),
            extra_Q=sum(r["extra_Q"] for r in part),rescues=sum(r[new]["correct"] and not r[old]["correct"] for r in part),
            losses=sum(r[old]["correct"] and not r[new]["correct"] for r in part),ci95=paired(part,old,new))
        if extra_mask:
            c["extra_admits"]=sum(r["extra_admits"] for r in part)
        if contested:
            c.update({k:sum(r[k] for r in part) for k in ("swapped_seats","mask_admits")})
        for k in (old,new):
            vals=[r[k] for r in part]
            s={t:sum(v[t] for v in vals) for t in ("Q","F","N","correct")}
            c[k]=dict(**s,macro_TPF=mean(v["N"]/v["F"] for v in vals),TE=s["N"]/s["Q"],failed=sum(bool(v["failure"]) for v in vals))
    return dict(registration=reg,finished=len(rows),rows_sha256=h.hexdigest(),cohorts=cohorts)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="mode",required=True)
    ev=sub.add_parser("run")
    for k in ("out","samples","reference","checkpoint","parent"): ev.add_argument("--"+k,type=Path,required=True)
    ev.add_argument("--q",type=int,choices=(8,16),required=True)
    ev.add_argument("--factor",choices=("interior","waiting","prefix","contested","interior_swap"),default="interior")
    ev.add_argument("--handoff",type=Path)
    ev.add_argument("--deadline",type=int,required=True)
    re=sub.add_parser("report");re.add_argument("root",type=Path);re.add_argument("manifest",type=Path)
    a=ap.parse_args()
    if a.mode=="run": run(a)
    else: print(json.dumps(report(a.root,json.loads(a.manifest.read_bytes())),indent=2,sort_keys=True))
