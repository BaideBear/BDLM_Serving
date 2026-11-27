"""Fixed Q128 only: registered 0.10 vs 0.20 joint entropy budget after full initialization."""
from importlib.util import module_from_spec, spec_from_file_location

from bdlm_serving.methods.t2t.budget_family import Law
from .gemma_fixed_dual import AblationRequest, B


class WarmEntropyRequest(AblationRequest):
    def __init__(self,model,data,*,seed,q,law,max_new_tokens,ablation,warm_entropy):
        if q!=128 or ablation!="00" or law!=Law(1,4) or type(warm_entropy) is not bool:
            raise ValueError("Unregistered warm entropy experiment")
        self.warm_entropy=warm_entropy
        super().__init__(model,data,seed=seed,q=q,law=law,max_new_tokens=max_new_tokens,ablation=ablation,
                         top_gate="off",fuse_first_verify=True)

    def _apply(self,states,pos,sample,prefix,cold,offer,route):
        h,=states
        if offer!=128 or self.tail is not None:
            raise ValueError("Fixed single-head Q128 only")
        warm=bool(sample) and route=="sparse" and len(h.rows.initialized)==B
        if warm and (cold or len(h.sampler.sc)!=B or not h.can_sample):
            raise RuntimeError("Warm sample without full live SC / measured rows")
        h.sampler.entropy_bound=.20 if warm and self.warm_entropy else .10
        e=super()._apply(states,pos,sample,prefix,cold,offer,route)
        e["warm_sample"]=warm
        return e


def auditor():
    from . import gemma_fixed_dual_eval as base
    spec=spec_from_file_location(__package__+"._warm_entropy_audit",base.__file__)
    mod=module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.AblationRequest=WarmEntropyRequest
    return mod.audited
