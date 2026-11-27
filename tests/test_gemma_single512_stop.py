import pytest
from experiments.t2t.serving_slots.gemma_single512_stop import B,eligible,paid_after_single_stop,inspect


def tick(t):
    rows=list(range(B))
    return dict(tick=t,offer=512,ordinal=0,route="dense_single",parts=[dict(phase="sample",sampled=rows)],
                positions=rows,sampled_absolute=rows,opened=None,encoded=False,discarded=[],
                single512=dict(epoch=0,resident=1,qualified=True,hit=False,applied=False),
                fullstop=dict(hit=False,latched=False))


@pytest.mark.parametrize("q,r,route",[(256,1,"dense_single"),(512,2,"dense_single"),(512,2,"dense_joint"),
    (512,1,"rebuild"),(512,1,"rebuild_sample"),(512,2,"dense_head_debt"),(512,1,"verify_paid")])
def test_excludes(q,r,route):
    assert not eligible(q,r,route)


@pytest.mark.parametrize("change",["route","epoch","resident","sample","gap","ordinal","none"])
def test_fused_terminal_requires_previous_measured_pair(change):
    a,b,c=[tick(i) for i in range(3)]
    b["single512"].update(hit=True,applied=True)
    if change=="route": a["route"]="rebuild_sample"
    if change=="epoch": b["single512"]["epoch"]=B
    if change=="resident": a["single512"]["resident"]=2
    if change=="sample": b["sampled_absolute"]=list(range(B-1))
    if change=="gap": a["tick"]=-1
    if change=="ordinal": a["ordinal"]=1
    assert paid_after_single_stop([a,b,c],2,0) == (change=="none")


def test_no_hit_on_first_or_on_parked_tail():
    a=tick(0)
    a["fullstop"].update(hit=True,latched=True)
    a["single512"].update(hit=True,applied=True)
    with pytest.raises(ValueError):
        inspect([a],0,True)
    a["opened"]=1
    with pytest.raises(ValueError):
        inspect([a],0,True)
