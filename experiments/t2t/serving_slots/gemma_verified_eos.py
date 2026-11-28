"""Only a measured EOS in a paid, ordered terminal chunk may end a sparse request."""
from .gemma_fixed_dual import AblationRequest, B
from bdlm_serving.methods.t2t.budget_family import Law


def verified_end(sealed, start, end, eos):
    if not 0 <= start < end <= B or set(sealed) != set(range(end)):
        raise RuntimeError("Terminal verification is not a contiguous prefix")
    return next((i+1 for i in range(start,end) if sealed[i] in eos), None)


class SwitchableVerifiedEOSRequest(AblationRequest):
    """Only the head-alone close path may shorten. Never unpark or drop a tail."""
    early = True
    fixed_fixture = False

    def _apply(self, states, pos, sample, prefix, cold, offer, route,
               prefix_pool=None):
        # Let the unchanged executor physically pay, seal and bill the entire chunk.
        start = self.cursor
        solo = len(states)==1 and self.tail is None and not self.rebuild
        e = super()._apply(states,pos,sample,prefix,cold,offer,route,
                           prefix_pool=prefix_pool)
        end = start+len(pos)
        if not self.early or route not in ("verify","verify_paid") or end >= B:
            return e
        if not solo and not self.fixed_fixture:
            return e
        h, = states
        base = self.origin+B*h.ordinal
        if (self.status!="ready" or e["encoded"] or e["encoder_F"] or e["encoder_Q"]
                or h.ordinal!=self.ordinal or self.cache.get_seq_length()!=base
                or self.tail is not None or h.rows.dirty or self.rebuild or sample
                or h.sampler.phase!="closing" or h.can_sample or self.cursor!=end
                or tuple(pos)!=tuple(range(base+start,base+end)) or len(pos)!=offer
                or len(self.tokens)!=B*self.ordinal):
            raise RuntimeError("Unexpected debt, epoch, sampling or partial terminal chunk")
        seal = h.sampler._sealed
        stop = verified_end(seal,start,end,self.eos)
        if stop is not None:
            if any(seal[i] in self.eos for i in range(start)):
                raise RuntimeError("Earlier sealed EOS was missed")
            self.tokens.extend(seal[i] for i in range(stop))
            e["verified_eos_end"] = stop  # One past measured EOS, local to this head.
            self.status = "done"
            self.lane = self.bank = self.rows = self.sampler = None
        return e


class VerifiedEOSRequest(SwitchableVerifiedEOSRequest):
    fixed_fixture = True

    def __init__(self, model, data, *, seed, q, early, prefix_order='right'):
        if type(q) is not int or q not in (32,64,128) or type(early) is not bool:
            raise ValueError("Requires a fixed sparse offer and explicit arm")
        if prefix_order not in ("right", "spread") or q > 64 and prefix_order == "spread":
            raise ValueError("Unsupported verified-EOS prefix identity")
        self.early = early
        super().__init__(model,data,q=q,seed=seed,max_new_tokens=2048,ablation="00",
                         law=Law(1,4 if q==128 else 8),top_gate="off",
                         prefix_order=prefix_order,fuse_first_verify=True)

    def step(self, q=None):
        if q is not None and (type(q) is not int or q != self.fixed_q):
            raise ValueError("No switching in the fixed experiment")
        return super().step(self.fixed_q)
