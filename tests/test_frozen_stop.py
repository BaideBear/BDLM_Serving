import pytest

from bdlm_serving.methods.t2t.frozen_stop import StopDecision, frozen_stop


def step(**changes):
    return frozen_stop(**dict(dict(start=32, width=32, ready=12, generated_from=38,
                                   generated_until=70, eos_position=41, debt=[], offer=4),
                              **changes))


def test_unpaid_suffix_is_not_free_or_rounded_to_offer():
    assert step(debt={48, 49}) == StopDecision("pay", "suffix_debt", (48, 49))
    assert step().action == "finish"
    assert step(debt={48, 49}, offer=1) == StopDecision("continue", "debt_exceeds_offer")


def test_partial_and_ordinary_are_distinct():
    assert step() == StopDecision("finish", "frozen_eos")
    assert step(ready=32) == StopDecision("finish", "ordinary_block_end")
    assert step(ready=32, eos_position=None).action == "continue"
    assert step(ready=9).action == "continue"  # EOS is the first still-editable row.


@pytest.mark.parametrize("change", [dict(eos_position=33), dict(eos_position=64),
    dict(eos_position=47, generated_until=47), dict(debt=[41]),
    dict(debt=[43], eos_position=None), dict(debt=[65]), dict(debt=[48,48]),
    dict(ready=33), dict(offer=0)])
def test_reject_bad_certificate_even_without_terminal_action(change):
    with pytest.raises(ValueError):
        step(**change)
