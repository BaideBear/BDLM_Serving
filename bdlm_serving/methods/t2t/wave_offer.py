"""Request-local whole-canvas offers. Projection and sampling have separate widths."""
from dataclasses import dataclass


TRACES={
    "wave_bridge_256":(256,), "wave_bridge_512":(512,), "wave_guard_512":(512,),
    "wave_guard_hhll":(512,512,256,256), "wave_guard_llhh":(256,256,512,512),
}


@dataclass(frozen=True)
class WaveTick:
    tick: int
    head: int
    resident: int
    mandatory: tuple
    may_open: bool


def widths(block, offer, debts):
    if type(offer) is not int or offer not in (block, 2*block) or len(debts) not in (1,2):
        raise ValueError("Expected one or two live canvases and a whole-canvas offer")
    sampling = min(offer//block,len(debts))
    # Native packed execution has no persistent tail decoder KV. Flush an
    # outstanding tail write once, without advancing its sampler under low Q.
    projection = 2 if len(debts)==2 and debts[1] else sampling
    return projection,sampling


def periodic(offers):
    values=tuple(offers)
    if not values:
        raise ValueError("Empty offer sequence")
    return lambda state:values[state.tick % len(values)]


def audit(trace, physical, block, offers):
    if not trace or not physical or physical[0][0]!="encoder":
        raise ValueError("Missing prompt prefill or request")
    expect=[]
    pending={}
    head=0
    prior_ages={}
    promoted=None
    for j,e in enumerate(trace):
        q=offers[j%len(offers)]
        h,r=e["head"],e["active"]
        if h!=head or e["tick"]!=j or e["offer"]!=q:
            raise ValueError("Wrong head, clock or external offer")
        ords=list(range(h,h+r))
        before=[pending.get(o,[]) for o in ords]
        projected,sampled=widths(block,q,before)
        if (e["width"]!=projected or e["sample_width"]!=sampled or e["q"]!=block*projected or
            e["dirty_before"]!=before or len(e["dirty_after"])!=r or
            e["refresh_only"]!=projected-sampled or e["ages_before"]!=[prior_ages.get(o,0) for o in ords]
            or len(e["ages_after"])!=r or promoted is not None and (promoted!=h or
                e["ages_after"][0]!=e["ages_before"][0]+1)):
            raise ValueError("Work or mandatory frontier disagrees")
        promoted=None
        for i,o in enumerate(ords):
            after=e["dirty_after"][i]
            adds,edits=e["admitted"][i],e["strict_edits"][i]
            if (after!=sorted(set(adds)|set(edits)) or set(adds)&set(edits) or
                len(set(adds))+len(set(edits))!=len(adds)+len(edits) or
                any(type(p) is not int or not 0<=p<block for p in after) or
                i>=projected and before[i] or i>=sampled and after or
                e["ages_after"][i]-e["ages_before"][i] not in ((0,1) if i<sampled else (0,))):
                raise ValueError("Unprojected mandatory row or advanced parked state")
            pending[o]=after
            prior_ages[o]=e["ages_after"][i]
        expect.append(("decoder",block*projected))
        for o in e["encoded"]:
            if o!=head:
                raise ValueError("Out-of-order encoder commit")
            expect.append(("encoder",block))
            pending.pop(head,None)
            head+=1
        if e["promoted"]:
            if e["promoted"]!=[head] or head not in ords or e["encoded"]!=[head-1]:
                raise ValueError("Promotion bypassed encoder or real head")
            promoted=head
    if promoted is not None:
        raise ValueError("Terminal request lacks promised authoritative head step")
    if [tuple(a) for a in physical[1:]]!=expect:
        raise ValueError("Observed encoder/decoder work disagrees with external trace")
