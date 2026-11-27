"""Experiment only: native stop on consecutive one-resident Q512 dense head samples."""
from collections import Counter
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location

from .gemma_fullstop import FullHeadStop, HeadTick
from .gemma_joint_verify import B, JointRequest, check, digest


def eligible(q, resident, route):
    return q == 2*B and resident == 1 and route == "dense_single"


def measured(e, origin):
    start = origin+B*e["ordinal"]
    return (eligible(e["offer"], e["single512"]["resident"], e["route"])
            and len(e["parts"]) == 1 and e["parts"][0]["phase"] == "sample"
            and e["parts"][0]["sampled"] == list(range(B))
            and e["positions"] == e["sampled_absolute"] == list(range(start,start+B)))


def paid_after_single_stop(trace, t, origin):
    if t < 2:
        return False
    p, h, v = trace[t-2:t+1]
    return (p["ordinal"] == h["ordinal"] == v["ordinal"]
            and p["tick"]+2 == h["tick"]+1 == v["tick"]
            and all(measured(e,origin) and e["single512"]["epoch"]==origin+B*v["ordinal"] for e in (p,h))
            and h["single512"]["hit"] and h["single512"]["applied"])


class SingleStopRequest(JointRequest):
    def __init__(self,model,data,*,seed,q,law,max_new_tokens,ablation,single512_stop):
        check(type(single512_stop) is bool, "Expected shadow or apply")
        super().__init__(model,data,seed=seed,q=q,law=law,max_new_tokens=max_new_tokens,
                         ablation=ablation,joint_verify=True)
        cfg, unused = model._prepare_generation_config(None,max_new_tokens=max_new_tokens)
        check(not unused and (cfg.stability_threshold,cfg.confidence_threshold)==(1,.005),
              "Wrong native stopping configuration")
        self.fullstop = single512_stop
        self.single_observer = FullHeadStop.from_model(model,cfg)
        check(self.single_observer.criterion is not None, "Missing native criterion")

    def step(self,q=None):
        check(q is None or type(q) is int and q==2*B, "Fixed Q512 only")
        return super().step(q)

    def _apply(self,states,pos,sample,prefix,cold,offer,route):
        h = states[0]
        current = HeadTick(self.tick,h.ordinal,int(self.cache.get_seq_length()))
        check(current.epoch==self.origin+B*self.ordinal and h.ordinal==self.ordinal and
              len(states)==1+int(self.tail is not None), "Incorrect current epoch or residency")
        scoped = eligible(offer,len(states),route)
        if scoped:
            expected = tuple(range(current.epoch,current.epoch+B))
            check(not self.rebuild and pos==expected and sample==set(expected) and h.can_sample,
                  "Expected all fresh head rows")
        else:
            self.single_observer.observe(current,phase="absent")
        # The pinned parent's consume callback supplies only this tick's processed
        # scores. Detaching it on all other routes also excludes sampled rebuilds.
        self.stop_observer = self.single_observer if scoped else None
        try:
            e = super()._apply(states,pos,sample,prefix,cold,offer,route)
        finally:
            self.stop_observer = None
        native = e.get("fullstop",{})
        e["single512"] = dict(epoch=current.epoch,resident=len(states),qualified=scoped,
                              hit=native.get("hit",False),applied=native.get("latched",False))
        return e


def make_auditor():
    from . import gemma_fixed_dual_eval as src
    spec = spec_from_file_location(__package__+"._single512_audit",src.__file__)
    private = module_from_spec(spec)
    spec.loader.exec_module(private)
    private.AblationRequest = SingleStopRequest
    return private.audited


def inspect(trace, origin, on):
    hits, resets, chain = [], Counter(), sha256()
    qualified = pairs = 0
    previous = resident_tail = None
    for t,e in enumerate(trace):
        m = e["single512"]
        if e["opened"] is not None:
            check(resident_tail is None and e["opened"]==e["ordinal"]+1,"Duplicate tail admission")
            resident_tail = e["opened"]
        yes = measured(e,origin)
        identity = (e["ordinal"],origin+B*e["ordinal"])
        check(m["epoch"]==identity[1] and m["resident"]==1+int(resident_tail is not None)
              and e["tick"]==t and e["offer"]==2*B and m["qualified"]==yes
              and ("fullstop" in e)==yes, "Observer/replay disagreement")
        two = yes and previous==identity
        native = e.get("fullstop",{})
        check(m["hit"]==native.get("hit",False) and m["applied"]==(on and m["hit"])
              and (not m["hit"] or two), "Stop outside a consecutive eligible pair")
        qualified += int(yes)
        pairs += int(two)
        if not yes:
            resets[e["route"]+("_resident2" if resident_tail is not None else "_resident1")] += 1
        chain.update(digest({k:e[k] for k in ("route","positions","parts","written_absolute","encoded",
                                             "verified_digest") if k in e}).encode())
        if m["hit"]:
            hp=e["parts"][0]
            masks=e["head_mask_before"]-len(hp["admitted"])+len(hp["revoked"])
            actionable=bool(masks and hp["age_after"]<48)
            check(t+1<len(trace) and trace[t+1]["ordinal"]==e["ordinal"], "Hit without a following tick")
            nxt=trace[t+1]
            if on:
                check(nxt["route"] in ("verify","verify_paid") and not nxt["sampled_absolute"] and
                      nxt["positions"]==list(range(identity[1],identity[1]+B)) and
                      not nxt["dirty_after_absolute"] and "verified_digest" in nxt and
                      nxt["dirty_before_absolute"]==e["dirty_after_absolute"],
                      "Stopped without physically verifying/paying the complete head")
            elif actionable:
                check(bool(nxt["sampled_absolute"]),"Actionable shadow hit without further sampling")
            hits.append(dict(tick=t,ordinal=e["ordinal"],epoch=m["epoch"],age=hp["age_after"],
                             masks=masks,actionable=actionable,dirty=len(e["dirty_after_absolute"]),
                             prefix=chain.hexdigest(),next_route=nxt["route"]))
        previous = identity if yes else None
        if e["encoded"] or e["discarded"]:
            resident_tail = None
    return dict(decoder_ticks=len(trace),qualified=qualified,pairs=pairs,hits=hits,resets=dict(resets),
                positives=len(hits),actionable=sum(h["actionable"] for h in hits),
                applied=len(hits) if on else 0,trajectory=chain.hexdigest())
