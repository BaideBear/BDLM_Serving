from bdlm_serving.methods.t2t.frontier_agreement import FrontierEvidenceLedger, before_measured_hole


def test_measured_hole_only_no_imputed_positions_or_sort_of_output():
    assert before_measured_hole([12, 1, 9, 14, 7], [0, 1, 0, 0, 0],
                                (12, 9, 7)) == (12, 9, 7)
    assert before_measured_hole([12, 1, 9, 15, 7], [0, 1, 0, 0, 0],
                                (12, 15, 7)) == (7,)
    assert before_measured_hole([12, 1], [0, 1], (12,)) == (12,)
    assert before_measured_hole([], [], ()) == ()


def test_blocked_row_retains_its_real_vote_and_can_become_eligible():
    ledger = FrontierEvidenceLedger()

    def step(tick, p, mid, *, block=0):
        return ledger.observe(block=block, tick=tick, positions=iter((9, 3)),
                              proposals=(8, mid), probabilities=(.9, p),
                              native=iter((False, False)), excluded_ids={0})

    assert step(1, .2, 1) == ()
    assert step(2, .9, 4) == ()  # 3 changed; 9 agrees but sits beyond it
    assert step(3, .9, 4) == (9, 3)
    assert step(4, .9, 4, block=1) == ()
    assert step(6, .9, 4, block=1) == ()


def test_native_row_does_not_block_but_rejected_special_does():
    def step(ledger, tick, native):
        return ledger.observe(block=0, tick=tick, positions=(1, 6, 9),
                              proposals=(99, 100, 101), probabilities=(1, .8, .8),
                              native=(native, False, False), excluded_ids={99})
    a, b = FrontierEvidenceLedger(), FrontierEvidenceLedger()
    assert step(a, 1, False) == step(b, 1, True) == ()
    assert step(a, 2, False) == ()
    assert step(b, 2, True) == (6, 9)
