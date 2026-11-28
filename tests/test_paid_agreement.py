import pytest
from bdlm_serving.methods.t2t.paid_agreement import EvidenceLedger


def see(ledger, tick, pos=(32,35), prop=(17,19), probs=(.6,.7),
        native=(False,False), block=1):
    return ledger.observe(block=block, tick=tick, positions=pos, proposals=prop,
                          probabilities=probs, native=native, excluded_ids={0,1})


def test_only_measured_twice_and_not_already_accepted():
    r = EvidenceLedger()
    assert see(r,0) == ()
    assert see(r,1,pos=(35,32),prop=(19,17),native=(False,True)) == (35,)
    assert see(r,2,pos=(32,36),prop=(17,17)) == ()


def test_new_epoch_gap_and_absence_expire_vote():
    for kwargs in (dict(tick=3),dict(tick=2,block=2),
                   dict(tick=2,pos=(33,34)),dict(tick=2,prop=(19,17))):
        r = EvidenceLedger()
        see(r,1)
        assert see(r,**kwargs) == ()
    r = EvidenceLedger()
    see(r,0)
    assert see(r,1,(),(),(),()) == see(r,2) == ()


def test_strict_floor_special_ids_and_request_isolation():
    a, b = EvidenceLedger(), EvidenceLedger()
    see(a,0,probs=(.5,.9),prop=(17,1))
    assert see(a,1,probs=(.99,.98),prop=(17,1)) == ()
    assert see(b,2) == ()
    assert see(a,2,probs=(.5,.9)) == ()


def test_errors_do_not_advance_state():
    r = EvidenceLedger()
    see(r,0)
    for kw in (dict(pos=(32,32)),dict(probs=(float("nan"),.8)),
               dict(probs=(.9,)),dict(native=(False,2))):
        with pytest.raises(ValueError):
            see(r,1,**kw)
    assert see(r,1) == (32,35)
    with pytest.raises(ValueError):
        see(r,1)
