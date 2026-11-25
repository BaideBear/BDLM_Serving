from random import Random

from bdlm_serving.methods.t2t.frontier import select_hard
from bdlm_serving.methods.t2t.protocol import Arm


def test_matched_counts_and_mask_positions_across_states():
    rng = Random(715)
    for _ in range(800):
        width = rng.randrange(4, 36)
        q = rng.randrange(1, width+1)
        elig = list(range(rng.randrange(0, width//2), width))
        resolved = [rng.choice((True, False)) for _ in range(width)]
        fresh = {i for i in elig if resolved[i] and rng.random() < .5}
        strict = {i for i in fresh if rng.random() < .5}
        w = [4 if i in fresh else rng.randrange(-1, 4) for i in range(width)]
        t = [rng.randrange(-1, 5) for _ in range(width)]
        e = rng.randrange(1, q+1)
        for mode in ("hard_slack", "hard_trigger"):
            a, ar = select_hard(elig, resolved, fresh, w, t, 5, q, e, mode, 7, previous_strict=strict)
            b, br = select_hard(elig, resolved, fresh, w, t, 5, q, e, mode+"_placebo", 7, previous_strict=strict)
            assert len(a) == len(b) and len(ar) == len(br)
            assert fresh <= set(a) and fresh <= set(b)
            assert [i for i in a if not resolved[i]] == [i for i in b if not resolved[i]]
            assert all(resolved[i] and i not in fresh for i in br)


def test_trigger_placebo_can_choose_outside_gate_and_never_force_an_empty_gate():
    r = [True, True, False, False, True, True, True, True, True]
    w, t = [3]+[-1]*8, [3]*9
    def pick(mode, seed, prev):
        return select_hard(range(9), r, {0}, w, t, 4, 3, 1, mode, seed, previous_strict=prev)
    assert pick("hard_trigger", 1, {0})[1] == [1]
    assert any(pick("hard_trigger_placebo", s, {0})[1][0] > 4 for s in range(40))
    assert pick("hard_trigger_placebo", 1, set())[1] == []
    for width in (32, 256):
        assert Arm.parse("4:0:3:hard_trigger_placebo", width).edit == 1
