from experiments.t2t.serving_slots.llada_window_fallback import decision


def event(cold, warm, wc, cc, *, waiting=True):
    return dict(sampled=[cold,warm],cold=[cold],paid_waiting_mask=[warm] if waiting else [],
                proposals=[(99,1,cc,cc>.5,cc>.9),(99,2,wc,wc>.5,wc>.9)])


def test_only_fallback_changes_not_high_confidence_admissions():
    e=event(5,4,.70,.55)
    assert decision(e,99,False)==[4] and decision(e,99,True)==[5]
    for wc,cc,expected in ((.94,.60,[4]),(.96,.95,[4,5]),(.53,.55,[5])):
        e=event(5,4,wc,cc)
        assert decision(e,99,False)==decision(e,99,True)==expected
    e=event(5,4,.70,.55,waiting=False)
    assert decision(e,99,True)==decision(e,99,False)==[4]
