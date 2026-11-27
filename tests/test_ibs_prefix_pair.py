from copy import deepcopy

import pytest

from experiments.t2t.serving_slots import ibs_prefix_pair as study


def row(quota):
    def step(tick, original, new, ready, after, *, wrote=(), before=()):
        return dict(absolute_positions=new, q_tokens=len(new), phase="optimus",
                    physical_forwards=1, physical_cap=8, offer_tick=tick,
                    block=1, cached_prefix_before=ready, cached_prefix_after=after,
                    candidate_local=[i-32 for i in original], dirty_before=list(before),
                    selected_local=[i-32 for i in original], selected_masked=[True]*len(original),
                    visible_kv=max(original)+1,written_absolute=list(wrote),edited_positions=[])
    pre=dict(phase="prefill",physical_forwards=1,absolute_positions=list(range(32)),q_tokens=32)
    full=list(range(32,40))
    tail=[40,41]
    events=[pre,step(0,full,full,0,0,wrote=full),step(1,full,full,0,8,before=full),
            step(2,tail,[38,39]+tail if quota else tail,8,8)]
    q=18+quota
    return dict(model="llada21",policy=study.LABELS[quota],dataset="toy",index=0,
                task_id="test",seed=10,prompt_ids=[2]*32,phase="formal",generation_cap=2048,
                events=events, calls=[dict(rows=e["q_tokens"],physical_rows=e["q_tokens"],
                    kv_rows=max(e["absolute_positions"])+1,batch=1) for e in events],
                forward_inputs=[[3]*e["q_tokens"] for e in events], output_tokens=1,
                generation_forwards=3, generation_query_rows=q,generation_physical_padded_rows=q,
                tpf=1/3,token_efficiency=1/q,prefill_rows=32,max_q=8)


def test_spare_rows_do_not_change_original_inputs():
    p=study.experiment_pair(dict(row(0),output_ids=[4]),dict(row(2),output_ids=[5]))
    assert p["refresh"]["first"]==2 and p["refresh"]["prefix_Q"]==2
    assert p["control"]["Q"]==18 and p["refresh"]["Q"]==20
    assert not p["same_ids"]


@pytest.mark.parametrize("field,value",[
    ("visible_kv",43),("written_absolute",[38]),("selected_local",[6,7,8,9]),
    ("absolute_positions",[32,33,40,41]),("dirty_before",[38]),
])
def test_refuse_invalid_extra_rows(field,value):
    r=deepcopy(row(2))
    r["events"][-1][field]=value
    with pytest.raises(ValueError):
        study.audit(r,2)
