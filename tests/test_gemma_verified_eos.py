"""No model or logits generated: verify guards on a synthetic already-sealed tape."""
from types import SimpleNamespace as NS

import pytest

from experiments.t2t.serving_slots.gemma_fixed_dual import AblationRequest
from experiments.t2t.serving_slots.gemma_verified_eos import VerifiedEOSRequest, verified_end
from experiments.t2t.serving_slots.gemma_verified_eos_report import theory


@pytest.mark.parametrize("row,saved",[(0,7),(31,7),(32,6),(223,1),(224,0),(255,0)])
def test_whole_chunk_billed(row,saved):
    assert theory([9]*256+[9]*row+[106],{1,106,50},32)==saved
    end=(row//32+1)*32
    seal={i:(106 if i==row else 9) for i in range(end)}
    assert verified_end(seal,end-32,end,{1,106,50})==row+1


def test_missing_or_future_row_and_no_eos():
    assert verified_end({j:7 for j in range(64)},32,64,{1}) is None
    assert theory([9]*2048,{1},32)==0
    for seal in ({0:1,2:1},{0:1,1:1,2:1}):
        with pytest.raises(RuntimeError,match="contiguous"):
            verified_end(seal,0,2,{1})
    with pytest.raises(ValueError):
        theory([9]*4,{1},32)


@pytest.mark.parametrize("case",["paid","off","draft","no_eos","dirty","gap","tail","would_encode"])
def test_after_normal_seal_only(monkeypatch,case):
    r=object.__new__(VerifiedEOSRequest)
    r.early=case!="off"
    r.origin,r.ordinal,r.cursor=13,1,0
    r.cache=NS(get_seq_length=lambda:269)
    r.eos,r.tokens,r.status={1,106},[7]*256,"ready"
    r.tail=object() if case=="tail" else None
    r.rebuild=False
    h=NS(ordinal=1,can_sample=False,rows=NS(dirty={0} if case=="dirty" else set()),
         sampler=NS(_sealed={},phase="closing"))
    def paid_step(self,states,pos,sample,prefix,cold,offer,route):
        assert self.cursor==0 and len(pos)==32
        h.sampler._sealed={j:(1 if j==7 and case!="no_eos" else 8) for j in range(32)}
        if case=="gap":
            del h.sampler._sealed[4]
        self.cursor+=32
        return dict(encoded=case=="would_encode",encoder_F=int(case=="would_encode"),encoder_Q=0)
    monkeypatch.setattr(AblationRequest,"_apply",paid_step)
    def run():
        return r._apply([h],tuple(range(269,301)),set(),(),(),32,"sparse" if case=="draft" else "verify_paid")
    if case in ("dirty","gap","tail","would_encode"):
        with pytest.raises(RuntimeError):
            run()
    else:
        e=run()
        if case=="paid":
            assert e["verified_eos_end"]==8 and r.tokens==[7]*256+[8]*7+[1] and r.status=="done"
        else:
            assert "verified_eos_end" not in e and r.status=="ready" and r.tokens==[7]*256
