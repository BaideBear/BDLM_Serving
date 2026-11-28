"""Fixed small offers: spend one cold seat on the earliest resident unresolved MASK."""
import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import time

from bdlm_serving.evaluation import scoring
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.methods.t2t.budget_family import Law
from .llada_close_factor import ANCHORS, SAMPLES
from .llada_fixed_family_eval import CAP, KEYS, ORDER, SCORER_SHA, evaluate_fixed, require


def exchange(plan, sampled, warm_pending):
    """Keep at least one cold row and remove only the rightmost new row (no KV hole)."""
    if plan.drain_only or len(plan.cold)<2 or not warm_pending or sampled!=plan.cold:
        return plan,sampled,None
    waiting=sorted(set(warm_pending)-set(plan.rows))
    if not waiting or waiting[0]>=plan.cold[0]:
        return plan,sampled,None
    take,drop=waiting[0],plan.cold[-1]
    def swap(rows):
        return tuple(sorted(set(rows)-{drop}|{take}))
    return (replace(plan,rows=swap(plan.rows),cold=plan.cold[:-1],
                    mask=tuple(sorted((*plan.mask,take)))),swap(sampled),(drop,take))


def interior_exchange(plan,sampled,candidates):
    if plan.drain_only or plan.cold or len(plan.mask)<2:
        return plan,sampled,None
    waiting=sorted(set(candidates)-set(plan.rows))
    if not waiting:
        return plan,sampled,None
    drop,take=max(plan.mask),waiting[0]
    if drop not in sampled:
        raise ValueError("Cannot displace an unmeasured MASK proposal")
    return (replace(plan,rows=tuple(sorted(set(plan.rows)-{drop}|{take})),
                    mask=tuple(p for p in plan.mask if p!=drop)),
            tuple(p for p in sampled if p!=drop),(drop,take))


def classes(factor="frontier"):
    from .llada_hfull_single_factor import request_classes
    _,control,_=request_classes()
    if factor=="successor":
        from .llada_successor_seats import variants
        return variants(control)
    if factor=="window":
        from .llada_window_geometry import window_class
        return control,window_class(control)
    if factor=="paid":
        from .llada_window_sample import variants
        return variants(control)
    class Exchange(control):
        def _cross_plan(self,q):
            plan,sampled,phase,cap=super()._cross_plan(q)
            self.swap=None
            if plan.cold and self.cross=="head" and not self.borrow and q in (4,8):
                masks,_,_=self._inputs(32)
                plan,sampled,self.swap=exchange(plan,sampled,(masks & self.bank.initialized)-self.dirty)
            return plan,sampled,phase,cap

        def _step(self,q):
            event=super()._step(q)
            if self.swap:
                event["frontier_swap"]=list(self.swap)
            return event
    if factor=="frontier":
        return control,Exchange
    class InteriorRead(control):
        edit_interior=False
        def _cross_plan(self,q):
            plan,sampled,phase,cap=super()._cross_plan(q)
            self.swap=None
            if self.cross=="head" and not self.borrow and q in (4,8) and len(plan.mask)>=2 and not plan.cold:
                start,end=max(self.head,self.origin),min(self.head+32,self.end)
                if self.bank.last_end>=end:
                    masks,_,_=self._inputs(32)
                    pool=set(range(max(start,min(masks)+1),end))-masks-self.dirty
                    plan,sampled,self.swap=interior_exchange(plan,sampled,pool)
                    if self.swap and self.edit_interior:
                        sampled=tuple(sorted((*sampled,self.swap[1])))
            return plan,sampled,phase,cap

        def _step(self,q):
            event=super()._step(q)
            if self.swap:
                event["interior_swap"]=list(self.swap)
            return event

    class InteriorEdit(InteriorRead):
        edit_interior=True
    return control,InteriorRead,InteriorEdit


def first_difference(a,b):
    for x,y in zip(a["trace"],b["trace"]):
        if "frontier_swap" not in y:
            require(x==y,"unexplained difference before first exchange")
            continue
        drop,take=y["frontier_swap"]
        require(x["pre_state"]==y["pre_state"] and len(x["cold"])>=2
                and x["cold"][-1]==drop and take<x["cold"][0] and take not in x["rows"]
                and take<x["pre_state"]["last_end"] and drop>=x["pre_state"]["last_end"]
                and x["phase"]==y["phase"] and x["prefix"]==y["prefix"] and x["dirty"]==y["dirty"]
                and x["q"]==y["q"] and x["layer_physical_q"]==y["layer_physical_q"]==[x["q"]]*20
                and y["cold"]==x["cold"][:-1] and y["mask"]==sorted(x["mask"]+[take])
                and x["sample_clock"]==y["sample_clock"]
                and all(set(y[k])==set(x[k])-{drop}|{take} for k in ("rows","sampled")),
                "first exchange changed a second factor")
        return dict(tick=x["tick"],drop=drop,take=take,width=x["q"])
    require(all(a[k]==b[k] for k in ("failure","F","N","Q","output_ids","score")),
            "unexplained difference without intervention")
    return None


def compact(r):
    extras={}
    tail=[e for e in r["trace"] if "supplemental_tail" in e]
    if tail:
        extras=dict(supp_calls=len(tail),supp_Q=sum(len(e["supplemental_tail"]) for e in tail),
                    supp_T2T=sum(len(set(e["supplemental_tail"]) & set(e["edited"])) for e in tail),
                    supp_revoke=sum(len(set(e["supplemental_tail"]) & set(e["revoked"])) for e in tail))
    if any("window_geometry" in e for e in r["trace"]):
        interventions=[e for e in r["trace"] if "window_geometry" in e]
        extras=dict(window_calls=len(interventions),
                    window_interior=sum(len(e["window_geometry"]["interior"]) for e in interventions),
                    window_edits=sum(len(set(e["window_geometry"]["interior"]) & set(e["edited"]+e["revoked"]))
                                     for e in interventions))
    paid=[e for e in r["trace"] if e.get("paid_waiting_mask")]
    if paid:
        extras.update(paid_waiting_calls=len(paid),paid_waiting_rows=sum(len(e["paid_waiting_mask"]) for e in paid),
                      paid_waiting_admits=sum(len(set(e["paid_waiting_mask"]) & set(e["admitted"])) for e in paid))
    return dict(**extras,**{k:r[k] for k in ("N","F","Q","failure","correct","row_visits","max_decoder_Q")},
                output_digest=sha256(json.dumps(r["output_ids"]).encode()).hexdigest(),
                actions=sum("frontier_swap" in e for e in r["trace"]),
                swapped_admits=sum(e["frontier_swap"][1] in e["admitted"]
                                   for e in r["trace"] if "frontier_swap" in e),
                interior_swaps=sum("interior_swap" in e for e in r["trace"]),
                interior_writes=sum(e["interior_swap"][1] in e["written_absolute"]
                                    for e in r["trace"] if "interior_swap" in e))


def first_interior(old,new,*,compare_sampling):
    first=None
    for a,b in zip(old["trace"],new["trace"]):
        swap=b.get("interior_swap")
        if swap is None:
            require(a==b,"trajectory changed before the registered intervention")
            continue
        drop,take=swap
        require(a["pre_state"]==b["pre_state"] and a["q"]==b["q"] and a["q"]>1
                and a["layer_physical_q"]==b["layer_physical_q"]==[a["q"]]*20
                and a["dirty"]==b["dirty"] and a["prefix"]==b["prefix"] and
                not a["cold"] and not b["cold"] and a["phase"]==b["phase"] and
                a["sample_clock"]==b["sample_clock"] and a["visible_kv"]==b["visible_kv"],
                "unmatched warm interior exchange")
        if not compare_sampling:
            require(take not in a["rows"] and drop in a["mask"] and drop in a["sampled"] and
                    set(b["rows"])==(set(a["rows"])-{drop}|{take}) and
                    set(b["sampled"])==set(a["sampled"])-{drop} and
                    set(b["mask"])==set(a["mask"])-{drop},"not a one-seat read-only exchange")
            return dict(tick=b["tick"],drop=drop,take=take,width=b["q"])
        require(a.get("interior_swap")==swap and a["rows"]==b["rows"] and
                a["mask"]==b["mask"] and take not in a["sampled"] and
                set(b["sampled"])==set(a["sampled"])|{take} and a["admitted"]==b["admitted"],
                "edit arm changed another decision")
        if first is None:
            first=dict(tick=b["tick"],drop=drop,take=take,width=b["q"],first_write=None)
        if a["written_absolute"]!=b["written_absolute"]:
            require(set(b["written_absolute"])==set(a["written_absolute"])|{take} and
                    take in set(b["edited"]+b["revoked"]),"unexplained write")
            first["first_write"]=b["tick"]
            return first
    require(all(old[k]==new[k] for k in ("failure","F","N","Q","output_ids","score")),
            "unexplained outcome difference")
    return first


def qualify_extension(factor, old, new):
    if factor=="paid":
        require(old["protocol"]=="llada-window-paid-mask-v1"
                and new["protocol"]=="llada-window-paid-mask-v1"
                and old["q"]==new["q"] and old["q"] in (8,16)
                and len(old["order"])==30 and len(new["order"])==90
                and old["order"]==new["order"][:30]
                and old["geometry_sha256"]==new["geometry_sha256"]
                and old["sampling_sha256"]==new["sampling_sha256"],
                "paid extension must inherit the same window control and sampling registration")
        require(all(old[k]==new[k] for k in ("device","checkpoint","samples","scorer")),
                "paid extension must inherit the same device, checkpoint, samples and scorer")
        return "paid"
    sources={"frontier":"04618ba8a8075dd7980f3e006dc5d8f64d1b8b36b2205e7aed7ba1895fc86cb9",
             "window":"5a986abee4a9461c07c4811014d978ece7fa6575a48fbe26c015e88d9fbf1254"}
    require(factor in sources and old["planner"]==sources[factor]
            and new["q"] in ((8,) if factor=="frontier" else (8,16))
            and len(new["order"])==90 and old["order"]==new["order"][:30]
            and all(old[k]==new[k] for k in ("protocol","q","device","checkpoint","samples","scorer")),
            "extension must inherit the registered original30 on the same device")
    if factor=="window":
        legacy="258c9a8be716798a0728e2dc0adbd1b1bc159d83b3a16eca13db63774040fde1"
        # The second digest only enables offer 4 in the dispatch condition.
        require(old["geometry_sha256"]==legacy and new["geometry_sha256"] in
                (legacy,"c3fe36a6fe092dc63bac8a22d5c7dfacdf0f93d64ac06d6cf09221c9ed53bcd9"),
                "changed window selection since original30")
    return "exchange" if factor=="frontier" else "window"


def run(a):
    import torch
    from .cross_race_llada import load_gpu, save
    opts=(64,) if a.factor=="successor" else (4,8,16) if a.factor in ("window","paid") else (4,8)
    device=os.environ["CUDA_VISIBLE_DEVICES"]
    require(a.offer in opts and device.startswith("GPU-") and torch.cuda.device_count()==1
            and sha256(a.samples.read_bytes()).hexdigest()==SAMPLES
            and sha256(Path(scoring.__file__).read_bytes()).hexdigest()==SCORER_SHA,
            "unregistered device or inputs")
    tasks=load_tasks(a.samples)
    require([t["index"] for t in tasks]==list(range(90)),"wrong sample order")
    order=[i for i in ORDER if i<a.limit]
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/"rows").mkdir()
    protocol={"frontier":"llada-frontier-one-seat-v1","interior":"llada-interior-one-seat-v1",
              "window":"llada-fixed-width-window-v1","successor":"llada-warm-successor-q64-v1",
              "paid":"llada-window-paid-mask-v1"}[a.factor]
    registration=dict(protocol=protocol,order=order,q=a.offer,device=device,
                      checkpoint=str(a.checkpoint),samples=SAMPLES,scorer=SCORER_SHA,
                      planner=sha256(Path(__file__).read_bytes()).hexdigest())
    prior={}
    if a.factor in ("window","paid"):
        registration["geometry_sha256"]=sha256(Path(__file__).with_name("llada_window_geometry.py").read_bytes()).hexdigest()
    if a.factor=="paid":
        registration["sampling_sha256"]=sha256(Path(__file__).with_name("llada_window_sample.py").read_bytes()).hexdigest()
    if a.factor=="successor":
        registration["successor_sha256"]=sha256(Path(__file__).with_name("llada_successor_seats.py").read_bytes()).hexdigest()
    replay = a.factor=="paid" and a.offer==4
    if replay:
        require(a.paired_window is not None and a.limit==90 and not a.extend_from,"Q4 requires a sealed control")
        sealed=json.loads(Path(__file__).with_name("controls").joinpath("llada-slot-allocation","c4-window-frozen90.json").read_bytes())
        reg=json.loads((a.paired_window/"run.json").read_bytes())
        digest=sha256()
        for i in range(90):
            digest.update(sha256((a.paired_window/"rows"/f"{i:02}.json").read_bytes()).digest())
        require(reg==sealed["registration"] and reg["protocol"]=="llada-fixed-width-window-v1"
                and all(reg[k]==registration[k] for k in ("order","q","device","checkpoint","samples","scorer","geometry_sha256"))
                and digest.hexdigest()==sealed["rows_sha256"],"wrong fixed Q4 source")
        registration.update(paired_window=str(a.paired_window),paired_window_rows=sealed["rows_sha256"])
    if a.extend_from:
        old_reg=json.loads((a.extend_from/"run.json").read_bytes())
        inherited_arm=qualify_extension(a.factor,old_reg,registration)
        prior={i:json.loads((a.extend_from/"rows"/f"{i:02}.json").read_bytes()) for i in range(30)}
        require(all(all(r[k]==tasks[i][k] for k in KEYS) for i,r in prior.items()),"wrong inherited identity")
        registration.update(inherited=str(a.extend_from),inherited_planner=old_reg["planner"],
                            inherited_replay=list(ANCHORS))
    save(a.out/"run.json",registration)
    def ready():
        if a.handoff is not None:
            (a.handoff/"ready").touch(exist_ok=False)
            while not (a.handoff/"go").exists():
                require(time.time()<a.deadline,"handoff expired")
                time.sleep(.2)
    tok,model=load_gpu(a,ready)
    controls=classes(a.factor)
    counts=Counter()
    for j,i in enumerate(order):
        require(time.time()<a.deadline-60,"deadline")
        if i in prior and i not in ANCHORS:
            save(a.out/"rows"/f"{i:02d}.json",prior[i])
            continue
        ref=json.loads((a.reference/f"{i:02d}-native32.json").read_bytes())
        require(i==ref["index"] and ref["generation_cap"]==CAP and ref["prompt_ids"],"wrong prompt")
        policy=(dict(cross=("quarter",False),coverage="head",warm_head_sample=True)
                if a.factor=="successor" else dict(cross=("head",False)))
        arms=[evaluate_fixed(model,tok,tasks[i],ref["prompt_ids"],a.offer,Law(),
                             **policy,request_cls=cls) for cls in controls]
        old,new=arms[:2]
        require(all(old[k]==r[k] for r in arms[1:] for k in (*KEYS,"input_sha256")),"unpaired input")
        if a.factor=="frontier":
            first=first_difference(old,new)
            extra=dict(exchange=compact(new))
        elif a.factor=="window":
            from .llada_window_geometry import first_geometry
            first=first_geometry(old,new)
            extra=dict(window=compact(new))
        elif a.factor=="paid":
            from .llada_window_sample import first as first_paid
            first=first_paid(old,new)
            extra=dict(paid=compact(new))
        elif a.factor=="successor":
            from .llada_successor_seats import first as first_tail
            first=first_tail(old,new,edit=False)
            edit=first_tail(new,arms[2],edit=True)
            require(bool(first)==bool(edit),"missing matched successor opportunity")
            extra=dict(read=compact(new),edit=compact(arms[2]),edit_first=edit)
        else:
            first=first_interior(old,new,compare_sampling=False)
            edit=first_interior(new,arms[2],compare_sampling=True)
            require(bool(first)==bool(edit),"missing matched read/edit opportunity")
            extra=dict(read=compact(new),edit=compact(arms[2]),edit_first=edit)
        for r in arms:
            require(0<r["max_decoder_Q"]<=a.offer and r["Q"]==sum(e["q"] for e in r["trace"])
                    and all(e["q"]<=a.offer and e["layer_physical_q"]==[e["q"]]*20
                            for e in r["trace"]),"invalid physical bill")
        row=dict(**{k:old[k] for k in KEYS},input=old["input_sha256"],first=first,old=compact(old),**extra)
        if replay:
            control=json.loads((a.paired_window/"rows"/f"{i:02}.json").read_bytes())
            require(all(row[k]==control[k] for k in (*KEYS,"input"))
                    and row["old"]==control["window"],"historical active-window arm did not reproduce")
            row["replayed_window"]=True
        if i in prior:
            refrow=prior[i]
            require(row["input"]==refrow["input"] and row["first"]==refrow["first"] and
                    all(all(row[arm][k]==v for k,v in refrow[arm].items()) for arm in ("old",inherited_arm)),
                    "original control/treatment anchor replay differs, do not extend")
        save(a.out/"rows"/f"{i:02d}.json",row)
        counts["exposed"]+=bool(first)
        print(i, bool(first),[(v["F"],v["correct"]) for v in arms],flush=True)
        if j+1==len(ANCHORS):
            require(order[:j+1]==list(ANCHORS) and counts["exposed"]>=(1 if a.factor=="paid" else len(ANCHORS)),
                    "unqualified pilot; do not expand")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("reference","samples","checkpoint","out"):
        ap.add_argument("--"+k,type=Path,required=True)
    ap.add_argument("--offer",choices=(4,8,16,64),type=int,required=True)
    ap.add_argument("--limit",choices=(30,90),type=int,default=30)
    ap.add_argument("--factor",choices=("frontier","interior","window","successor","paid"),default="frontier")
    ap.add_argument("--extend-from",type=Path)
    ap.add_argument("--paired-window",type=Path)
    ap.add_argument("--handoff",type=Path)
    ap.add_argument("--deadline",type=int,required=True)
    run(ap.parse_args())
