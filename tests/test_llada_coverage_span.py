import pytest
from experiments.t2t.serving_slots.coverage_span import select
from experiments.t2t.serving_slots.coverage_span_report import TIERS, decide


def test_debt_has_priority_and_may_be_parked():
    blocks=[(0,8,{3,4,5})]
    init=set(range(8))|{35,36}
    plan,sample=select(2,blocks,init,{35,36})
    assert plan.rows==(35,36) and not sample
    plan,sample=select(4,blocks,init,{35,36})
    assert plan.rows==(3,4,35,36) and sample==(3,4)


def test_prefix_quota_and_spare_capacity_are_different():
    blocks=[(0,16,set(range(8,16))),(16,24,set(range(16,24)))]
    init=set(range(8))
    plan,_=select(8,blocks,init,set())
    assert plan.rows==tuple(range(7,15))
    plan,_=select(24,blocks,init,set())
    assert plan.rows==tuple(range(24))
    for q in TIERS:
        plan,s=select(q,blocks,init,{4,5})
        assert {4,5} <= set(plan.rows) and set(s)<=set(plan.rows)
        assert len(plan.rows)==len(set(plan.rows))<=q


def test_pilot_needs_high_tier_gain_and_does_not_mix_arms():
    def row(tpf=1, score=9):
        return dict(macro_TPF=tpf, correct=score, new_failures=0)
    data={q:{"cohorts":{"all":dict(n=9,live=row(),span=row(2 if q<32 else 1),covered=row())}} for q in TIERS}
    assert decide(data)["expand"] is None
    data[32]["cohorts"]["all"]["span"]=row(2)
    assert decide(data)["expand"]=="span"
    data[16]["cohorts"]["all"]["span"]=row(2,7)
    assert decide(data)["expand"] is None
    with pytest.raises(ValueError):
        decide({32:data[32]})


def test_confirmation_uses_common_unseen_ids_and_rejects_unsealed_results(tmp_path,monkeypatch):
    import json
    from experiments.t2t.serving_slots import coverage_confirmation as c
    pilot={q:dict(registration=dict(offer=q),receipt_sha256=str(q)) for q in TIERS}
    monkeypatch.setattr(c,"audit",lambda *a:pilot)
    monkeypatch.setattr(c,"decide",lambda _:{"expand":"covered"})
    meta=[dict(index=i,dataset="test",task_id=f"test/{i}",seed=i) for i in range(90)]
    m=tmp_path/"manifest.json"
    m.write_text(json.dumps(dict(samples=meta)))
    order=[i+j for i in range(30) for j in (0,30,60) if i+j not in c.ANCHORS]
    for q in TIERS:
        p=tmp_path/f"llada-coverage-span-v1-q{q}"
        d=tmp_path/f"llada-coverage-span-v1-confirm-covered-q{q}"
        p.mkdir(); d.mkdir()
        (d/"run.json").write_text(json.dumps(dict(offer=q,extension=True,winner="covered",confirm_indices=order,
            anchors_excluded=c.ANCHORS,pilot_receipt_sha256=str(q),code_sha256="f"*64,checkpoint="test")))
        for i in (*c.ANCHORS,*order[:2 if q>4 else 1]):
            a=dict(correct=True,failure=None,N=11,F=4,Q=9,max_decoder_Q=3,two_block_F=0,output_digest="a"*64)
            (p if i in c.ANCHORS else d).joinpath(f"{i:02}.json").write_text(json.dumps(dict(**meta[i],input=f"{i:064x}",
                    live=a,covered=dict(a,N=8,F=2,Q=5))))
    out=c.report(tmp_path,m,interim=True)
    assert not out["complete"] and out["cohort_indices"]=={"interim_common":[0]}
    g=out["results"][4]["interim_common"]
    assert g["delta_macro_TPF"]==1.25 and g["output_length_term"]<0<g["forward_count_term"]
    with pytest.raises(ValueError,match="not sealed"):
        c.report(tmp_path,m)
    assert c.report(tmp_path,m,sealed_tiers=True)["results"]=={}
    # Q64 completes first. Its full cohorts must not implicitly upgrade the other offers.
    for i in order[2:]:
        (d/f"{i:02}.json").write_text(json.dumps(dict(**meta[i],input=f"{i:064x}",live=a,covered=a)))
    (d/"complete.json").write_text(json.dumps(dict(q=64,indices=order,winner="covered")))
    monkeypatch.setattr(c,"paired",lambda *args: {})
    completed=c.report(tmp_path,m,sealed_tiers=True)
    assert completed["sealed_offers"]==[64] and set(completed["results"])=={64} and not completed["complete"]
    assert completed["results"][64]["frozen90"]["n"]==90 and completed["results"][64]["confirm81"]["n"]==81
    row=d/"00.json"
    row.write_text(row.read_text().replace(f'{0:064x}','e'*64))
    with pytest.raises(ValueError,match="identity mismatch"):
        c.report(tmp_path,m,interim=True)
