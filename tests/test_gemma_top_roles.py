import pytest

from experiments.t2t.serving_slots.gemma_top_roles import ROLES, classify
from experiments.t2t.serving_slots.plot_fixed_gemma_profile import collect, verified_profile


def part(ordinal, phase, rows, *, dirty=(), sampled=(), admitted=(), edited=(), prefix=()):
    return dict(ordinal=ordinal, phase=phase, rows=rows, dirty_before=dirty, sampled=sampled,
                admitted=admitted, edited=edited, revoked=[], prefix=prefix)


def test_roles_and_debt_in_mixed_phases():
    def step(t, left, rows, parts, before, after):
        return dict(tick=t, offer=512, ordinal=0, route="joint", head_mask_before=left,
                    positions=rows, decoder_Q=len(rows), parts=parts, dirty_before_absolute=before,
                    written_absolute=after, dirty_after_absolute=after)
    events=[
        step(0,256,[1,2,257],[part(0,"dense",[0,1],sampled=[0,1],admitted=[0,1]),
                              part(1,"dense",[0],sampled=[0],admitted=[0])],[],[1,2,257]),
        step(1,254,[1,2,257],[part(0,"verify",[0,1],dirty=[0,1]),
                              part(1,"dense",[0],dirty=[0],sampled=[0],edited=[0])],[1,2,257],[257]),
        step(2,254,[1,2,257],[part(0,"dense",[0,1],prefix=[0]),
                              part(1,"rebuild",[0],dirty=[0])],[257],[]),
    ]
    tape=dict(trace=events,physical_trace=[dict(kind="encoder",paid=False,positions=[0])]+[
        dict(kind="decoder",paid=True,tick=e["tick"],positions=e["positions"]) for e in events])
    st=classify(tape)
    assert dict(st["roles"])==dict(unresolved=3,verify=2,mandatory=1,prefix=1,rebuild=1,**{"clean-other":1})
    assert st["dirty_positions"]["verify"]==2 and st["edit_positions"]["mandatory"]==1
    assert st["placement"]=={"successor_Q":3,"head_Q":6}
    events[-1]["parts"][1]["dirty_before"]=[]
    with pytest.raises(ValueError,match="projection/debt"):
        classify(tape)


def test_all_five_profile_role_bills_are_disjoint_and_additive():
    rows=[x for x in verified_profile(collect()) if x["method"]=="Verified EOS request"]
    assert len(rows)==15
    for x in rows:
        assert sum(x["Q_"+r] for r in ROLES)==x["history_Q"]
        assert x["token_efficiency"]==pytest.approx(x["N"]/x["history_Q"],abs=1e-8)
    for q in (32,64,128,256,512):
        z={r["cohort"]:r for r in rows if r["budget"]==q}
        for key in ("n","N","history_Q",*("Q_"+r for r in ROLES)):
            assert z["frozen90"][key]==z["original30"][key]+z["new60"][key]
