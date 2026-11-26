import pytest

from bdlm_serving.methods.t2t.slot_trace import RoundInfo
from experiments.t2t.serving_slots.expanded_handoff import STAGES
from experiments.t2t.serving_slots.wide_gate import POLICIES, REFERENCES, choice


@pytest.mark.parametrize("model,width", [("llada21", 32), ("gemma", 256)])
def test_registered_midpoints_and_two_start_phases(model, width):
    fixed, groups = (qs for m, _, qs in STAGES if m == model)
    assert len(set(fixed)) == 3 and all(0 < q < width for q in fixed)
    assert set(groups) == set(REFERENCES[model])
    for low in groups:
        for phase in ("low", "high"):
            for t in range(8):
                for mask in (0, width//4, width//4 + 1, width):
                    i = RoundInfo(t, width, width, mask, width-mask)
                    cap, all_offer = choice("wide_all_"+phase, low, i)
                    assert (cap == width) == ((t % 4 < 2) == (phase == "high"))
                    assert all_offer.q == cap
                    cap2, tail = choice("wide_tail_"+phase, low, i)
                    assert cap2 == cap and tail.q <= cap
                    assert tail.native_full == (cap == width and mask <= width//4)
                    flat_cap, flat = choice("wide_fixed_"+phase, low, i)
                    assert flat_cap == flat.q == (width if phase == "high" else low)


def test_no_unregistered_gate():
    assert len(set(POLICIES)) == 6
    with pytest.raises(ValueError):
        choice("wide_tail_other", 8, RoundInfo(0,0,32,4,0))
