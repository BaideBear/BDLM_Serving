from experiments.t2t.serving_slots.llada_risk_ribbon import ANCHORS,select


def test_one_risk_limit_across_unresolved_positions():
    vals={0:(True,.8),3:(True,.6),4:(True,.875),5:(True,.875)}
    assert select([0,3,4,5],vals,{3}) == [0,4]
    assert select([0,2,3],vals,{3}) == [0]


def test_missing_or_mask_proposal_cannot_be_crossed():
    assert select([0,1],{0:(False,1.),1:(True,1.)},{0})==[]
    assert select([5,6],{6:(True,1.)},set())==[]
    assert sum(i<30 for i in ANCHORS)==3 and len(set(ANCHORS))==9
