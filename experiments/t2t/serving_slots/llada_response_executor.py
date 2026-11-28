"""Serial LLaDA Live response fork with one explicit whole-successor admission exception.

A nominal offer is *not* a hard cap on an admission tick. Ordinary ticks,
including payments after an oversize admission, retain the strict Live contract.
No weights, KV reconstruction forwards, or prefill are copied by inference.
"""
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from threading import current_thread

import torch

from bdlm_serving.methods.t2t.llada_budget_request import B
from bdlm_serving.methods.t2t.serving_profiles import open_strict_llada
from .llada_frontier_accept import PROFILE

MODES = ("strict", "full_block")


def admission_start(origin, end, resident_end, cold):
    """The first block containing generated rows is the bootstrap, not a successor."""
    first = origin // B
    new = {p//B*B for p in cold if first < p//B and p//B*B >= resident_end}
    if not new:
        return None
    if len(new) != 1 or min(new) != resident_end or resident_end >= end:
        raise RuntimeError("Noncontiguous or multiple new successor blocks")
    return min(new)


class _Admission:
    def _switch_plan(self, q):
        plan,sampled,phase,cap = super()._switch_plan(q)
        start = admission_start(self.origin,self.end,self.bank.last_end,plan.cold)
        self._response_build = None
        if self._response_mode == "strict" or start is None:
            return plan,sampled,phase,cap
        block = tuple(range(start,min(start+B,self.end)))
        if (self.stopping or not sampled or plan.drain_only or self.bank.initialized.intersection(block)
                or not self.dirty.issubset(plan.rows)
                or not bool((self.x[0,start:start+len(block)]==self.mask_id).all())):
            raise RuntimeError("Admission needs a pristine MASK block and a plan paying all prior debt")
        old = plan.rows
        rows = tuple(sorted(set(old).union(block)))
        # Preserve the parent's head sampler seats at 2B, causal block mask and all debt.
        sampled = tuple(sorted(set(sampled).union(block)))
        plan = replace(plan,rows=rows,cold=tuple(sorted(set(plan.cold).union(block))))
        self._response_build = (start,block,old)
        return plan,sampled,"full_block_admission",max(q,len(rows))


@lru_cache(maxsize=None)
def _adapt(base):
    return type("ResponseLive",(_Admission,base),{})


class ResponseHandle:
    """`step(offer)`, `fork(mode=...)`, `status`, `result()`, `close()` on one serial lane."""
    offers = (4,8,16,32,64)

    def __init__(self, req):
        self._request = req
        self._status = "ready"
        self._result = None
        self.opened_prefill_q = req.origin
        self._owner = current_thread()

    def _check_lane(self):
        if current_thread() is not self._owner:
            raise RuntimeError("Use the owning serial inference lane")

    @property
    def status(self):
        return self._status

    @torch.inference_mode()
    def fork(self, *, mode=None):
        self._check_lane()
        req=self._request
        if self.status!="ready":
            raise ValueError("Fork requires a ready request")
        if mode is None: mode=req._response_mode
        if mode not in MODES:
            raise ValueError("Fork a ready request and specify a registered mode")
        # The engine is a stateless _execute binding to shared eval weights and
        # projections. All mutable request tensors, incl. cache, votes and IDs,
        # must belong to the child. Cloning here is D2D, never an extra model call.
        branch=deepcopy(req,{id(req.model):req.model,id(req.engine):req.engine})
        branch._response_mode=mode
        return ResponseHandle(branch)

    def step(self, q):
        self._check_lane()
        if self.status!="ready" or type(q) is not int or q not in self.offers:
            raise ValueError("Expected a ready response and a supported nominal offer")
        req=self._request
        self._status = "advancing"
        try:
            e=req.step(q)
            build=req._response_build
            width=e["physical_q"]
            if (width!=e["q"] or width!=len(e["rows"]) or
                    e["layer_physical_q"]!=[width]*len(req.engine.projections) or
                    e["F"]!=1 or e["offer"]!=q or (width>q and not build)):
                raise RuntimeError("Invalid response projection bill")
            if build:
                start,block,old=build
                if (set(e["rows"])!=set(old).union(block) or not set(block)<=set(e["sampled"])
                        or not set(block)<=req.bank.initialized):
                    raise RuntimeError("Incomplete physical successor admission")
            if req.status == "done":
                self._result, self._request = tuple(req.tokens), None
            elif req.status != "ready":
                raise RuntimeError(f"Unexpected underlying request state: {req.status}")
            self._status = req.status
            return dict(e, response_mode=req._response_mode, admission_start=build[0] if build else None,
                        admission_rows=list(build[1]) if build else [],
                        parent_plan_rows=list(build[2]) if build else list(e["rows"]),
                        prefix_share=(req.law.num, req.law.den),
                        nominal_offer=q,over_offer_Q=max(0,width-q),decoder_F=1,decoder_Q=width,
                        history_Q=width,encoder_F=0,encoder_Q=0,generation_F=1,physical_generation_Q=width)
        except BaseException:
            req.status=self._status="failed"
            self._request=None
            raise

    def result(self):
        if self.status!="done":
            raise RuntimeError(f"No completed output: {self.status}")
        return self._result

    def close(self):
        self._check_lane()
        if self.status=="ready":
            self._request.status=self._status="cancelled"
            self._request=None
        elif self.status=="advancing":
            raise RuntimeError("Close only at a tick boundary")


def open_llada_response(model,input_ids,*,seed,mask_id,eos_ids,max_new_tokens=2048,
                        mode="full_block",audit_admission=False):
    """Create the exact Live v1 parent once, then opt in to its successor planner."""
    if mode not in MODES or type(audit_admission) is not bool:
        raise ValueError("Invalid response mode or telemetry option")
    handle=open_strict_llada(model,input_ids,seed=seed,mask_id=mask_id,eos_ids=eos_ids,
                             max_new_tokens=max_new_tokens,profile=PROFILE)
    req=handle._request
    req.__class__=_adapt(type(req))
    req.audit_admission=audit_admission
    req._response_mode=mode
    req._response_build=None
    return ResponseHandle(req)
