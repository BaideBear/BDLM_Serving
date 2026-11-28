from tools.llada_ibs_eos_q16_risk import trace
from bdlm_serving.methods.t2t.capped_agreement import CappedEvidenceLedger


def event(tick, probabilities, *, block=2, native=None, proposals=None):
    positions = (77, 70, 73, 80, 82, 83)
    return dict(block=block, tick=tick, positions=positions,
                proposals=proposals or (11, 11, 11, 99, 11, 11),
                probabilities=probabilities,
                native=(False,) * 6 if native is None else native)


def test_risk_rank_tie_exact_boundary_and_v1_order():
    t = [event(4, (.875, .75, .75, 1., .5, 1.)),
         event(5, (1., .75, .75, 1., 1., 1.), native=(False, False, False, False, False, True))]
    _, full, kept, ranked, spent = list(trace(t, {99}, cls=CappedEvidenceLedger))[-1]
    assert full == (77, 70, 73)
    assert kept == (77, 70)
    assert ranked == [(.125, 77), (.25, 70), (.25, 73)]
    assert spent == .375
    t[0]["probabilities"] = (.75, .75, .75, 1., .5, 1.)
    _, _, kept, ranked, spent = list(trace(t, {99}, cls=CappedEvidenceLedger))[-1]
    assert ranked == [(.25, 70), (.25, 73), (.25, 77)]
    assert kept == (70, 73) and spent == .5
    t[0]["probabilities"] = (.875, .9375, .5625, 1., .5, 1.)
    _, _, kept, ranked, _ = list(trace(t, {99}, cls=CappedEvidenceLedger))[-1]
    assert ranked[:2] == [(.125, 77), (.25, 70)]
    assert kept == (77, 70)
    t[1]["probabilities"] = (1., 1., .75, 1., 1., 1.)
    _, _, kept, ranked, _ = list(trace(t, {99}, cls=CappedEvidenceLedger))[-1]
    assert ranked[:2] == [(.0625, 70), (.125, 77)] and kept == (77, 70)


def test_skipped_tick_block_transition_and_changed_id():
    p = (1.,) * 6
    t = [event(1, p), event(3, p), event(4, p, block=3),
         event(5, p, block=3, proposals=(12, 11, 11, 99, 11, 11))]
    assert [entry[2] for entry in trace(t, {99}, cls=CappedEvidenceLedger)] == [
        (), (), (), (70, 73, 82, 83)]
