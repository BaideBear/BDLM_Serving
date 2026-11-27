"""Q512 resident-tail terminal writeback. Only the head is verified."""
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
import json

from bdlm_serving.methods.t2t.budget_family import Law
from .gemma_fixed_dual import AblationRequest, B, FULL


def check(ok, message):
    if not ok:
        raise ValueError(message)


def digest(x):
    return sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def validate_plan(req, states, pos, sample, q):
    h,t = states
    base = req.origin+B*req.ordinal
    expected = tuple(range(base,base+B))+tuple(base+B+i for i in sorted(t.rows.dirty))
    check(req.joint_verify and req.sample_full_rebuild and req.fuse_first_verify and q==2*B
          and req.top_gate=="half" and req.tail is t and h.ordinal+1==t.ordinal and not req.rebuild
          and not req.cursor and not h.can_sample and h.sampler.phase=="sample" and h.rows.initialized==FULL
          and (h.rows.dirty or t.rows.dirty) and not sample and pos==expected and req.bank.causal,
          "Not a debt-covered block-causal joint verify")


def validate_tick(e, origin, partial, debt, epoch_rows, measured, req):
    base=origin+B*e["ordinal"]
    head=set(range(base,base+B))
    tail=debt-head
    hp,*tp=e["parts"]
    check(req.joint_verify and req.sample_full_rebuild and e["offer"]==2*B and not partial
          and bool(debt) and head <= epoch_rows and head|tail == set(measured)
          and e["resident_at_close"]==e["ordinal"]+1 and hp["phase"]=="verify"
          and (not e["head_mask_before"] or hp["age_before"]==48)
          and hp["rows"]==list(range(B)) and not e["sampled_absolute"] and not e["written_absolute"]
          and not e["dirty_after_absolute"] and len(tp)==bool(tail)
          and all(p["phase"]=="drain" and p["rows"]==p["dirty_before"]
                  and p["age_before"]==p["age_after"] and not p["dirty_after"] for p in tp),
          "Joint terminal receipt diverged from measured head+tail debt")


class JointRequest(AblationRequest):
    def __init__(self,model,data,*,seed,q,law,max_new_tokens,ablation,joint_verify):
        check(q==2*B and ablation=="00" and type(joint_verify) is bool and law==Law(),"Fixed 2B only")
        self.joint_verify=joint_verify
        super().__init__(model,data,seed=seed,q=q,law=law,max_new_tokens=max_new_tokens,ablation=ablation,
                         top_gate="half",fuse_first_verify=True,sample_full_rebuild=True,receipt_tail_ids=True)

    def _apply(self,states,pos,sample,prefix,cold,offer,route):
        h=states[0]
        mark=None
        if (self.tail is not None and not self.rebuild and not self.cursor and not h.can_sample
                and h.rows.initialized==FULL and h.sampler.phase=="sample" and
                (h.rows.dirty or self.tail.rows.dirty)):
            check(route==("verify_joint_paid" if self.joint_verify else "drain"),"Missed opportunity")
            t=self.tail
            def state(c):
                return (c.sampler.ids.tolist(),sorted(c.rows.accepted), sorted(c.rows.dirty),
                        c.sampler.rng.get_state().tolist(),c.sampler.age,sorted(c.sampler.sc))
            mark=dict(tick=self.tick,ordinal=h.ordinal,head_dirty=sorted(h.rows.dirty),
                      tail_dirty=sorted(t.rows.dirty),state=digest([state(h),state(t)]))
        e=super()._apply(states,pos,sample,prefix,cold,offer,route)
        if mark:
            e.update(opportunity=mark,resident_at_close=mark["ordinal"]+1)
        if route in ("verify_joint_paid","verify","verify_paid"):
            e["verified_digest"]=digest([h.sampler._sealed[r] for r in range(B)])
        return e


def make_auditor():
    from . import gemma_fixed_dual_eval as src
    spec=spec_from_file_location(__package__+"._joint_verify_audit",src.__file__)
    private=module_from_spec(spec)
    spec.loader.exec_module(private)
    private.AblationRequest=JointRequest
    return private.audited


def alignment(trace,on,origin):
    """Hash shared prehistory and exact selected positions before the first difference."""
    chain=sha256()
    hits=[]
    for t,e in enumerate(trace):
        if "opportunity" in e:
            m=dict(e["opportunity"])
            h=e["ordinal"]
            base=origin+B*h
            head=list(range(base,base+B))
            debt=sorted([base+i for i in m["head_dirty"]]+[base+B+i for i in m["tail_dirty"]])
            joint=head+[base+B+i for i in m["tail_dirty"]]
            v=e if on else trace[t+1]
            check(e["positions"]==(joint if on else debt) and
                  ((on and e["route"]=="verify_joint_paid") or
                   (not on and e["route"]=="drain" and v["route"]=="verify" and
                    v["positions"]==head and v["ordinal"]==h)),"First or later row plan mismatch")
            hits.append(dict(**m,prefix=chain.hexdigest(),sealed=v["verified_digest"],
                             drain_rows=digest(debt),verify_rows=digest(head),
                             joint_rows=digest(joint),applied=on))
        if not hits:
            chain.update(digest({k:e[k] for k in ("route","positions","parts","encoded","promoted",
                                                  "sampled_absolute","written_absolute")}).encode())
    return hits
