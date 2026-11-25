import gzip
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from bdlm_serving.evaluation.frozen import MANIFEST
from bdlm_serving.methods.t2t.planner import History, allocate
from tools.analyze_t2t import replay


def test_disjoint_budget_and_fallback():
    h=History([True,True,True,False,False],[.1,.7,.4,None,None],[0,2,4,0,0],{0},{0})
    assert allocate(h,range(5),3,1,1)==[0,2,3]
    assert allocate(h,range(5),4,0,4)==[0,2,3,4]
    assert allocate(h,[4,4],8,1,2)==[4]
    assert allocate(h,range(5),0,0,0)==[]


def simulate(model):
    ll = model=="llada21"
    b = 32 if ll else 256
    def step(before,after,accepted,clock):
        return dict(block=0,clock=clock,before=before,after=after,accepted=accepted,
                    draft=after,certainty=list(range(b)))
    x=list(range(b))
    y=[a+1 for a in x]
    valid=[True,True,True]+[False]*(b-3)
    z=y.copy()
    for i in (0,1,2,3): z[i]+=4
    second=valid.copy()
    if not ll: second[2]=False   # revocation to noise, not a T2T edit
    meta=json.loads(MANIFEST.read_text())["samples"][0]
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/"trace.gz"
        with gzip.open(p,"wt") as f:
            for row in ([meta]+([dict(prompt_length=1)] if ll else [])+
                        [step(x,y,valid,1 if ll else 48),step(y,z,second,2 if ll else 47)]):
                f.write(json.dumps(row)+"\n")
        stats=defaultdict(lambda:[0,0,0,0])
        result=replay(p,model,stats)
    assert result["t2t"]==2
    assert result["revoked"]==(0 if ll else 1)
    assert stats["all","all_resolved","uncertain",1][:3]==[1,1,2]
    assert abs(stats["all","all_resolved","uniform_expected",1][0]-(1 if ll else 2/3))<1e-8


def test_prompt_excluded():
    simulate("llada21")


def test_renoising_not_a_positive():
    simulate("gemma")
