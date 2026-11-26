from types import SimpleNamespace

import pytest

from experiments.t2t.serving_slots.llada_all_slots import AllSlotKernel
from experiments.t2t.serving_slots.llada_quote import quote
from experiments.t2t.serving_slots.llada_spawn import SpawnGatedKernel
from test_llada_top_transition import CPUModel, create


@pytest.mark.parametrize("threshold", [True, float("nan"), -0.1, 1.1])
def test_reject_bad_threshold(threshold):
    with pytest.raises(ValueError, match="spawn threshold"):
        SpawnGatedKernel(CPUModel().eval(), spawn_threshold=threshold)


def test_only_current_head_net_mask_progress_is_used():
    kernel = SpawnGatedKernel(CPUModel().eval(), spawn_threshold=.8)
    def decide(before, after):
        event = {"block_counts": {"4": dict(masked_before=before, masked_after=after),
                                  "5": dict(masked_before=32, masked_after=0)}}
        return kernel.admit_new_tail(SimpleNamespace(resident=True, events=[event], block=4))
    assert decide(5, 1) and not decide(5, 2)
    assert not decide(0, 0) and not decide(4, 5)


def test_deferral_prices_actual_work_and_never_parks_resident_tail():
    m = CPUModel().eval()
    k = SpawnGatedKernel(m, spawn_threshold=.8)
    s = create(k, prompt=43, length=83)
    assert not k.admit_new_tail(s)
    quotes = []
    for offer in (64, 64, 8, 64, 16, 4, 32, 64):
        if s.phase == "done":
            break
        bill = quote(k, s, offer)
        e = k.step(s, offer)
        assert (bill["action"], bill["physical_q"]) == (e["top_action"], e["physical_q"])
        assert e["q_budget"] == offer
        quotes.append((bill["action"], e.get("tail_admission")))
    assert quotes[:2] == [("head_bootstrap", "defer"), ("fused_growth", "open")]
    assert quotes[2][1] is None


@pytest.mark.parametrize("prompt,length", [(32, 64), (43, 87), (47, 1)])
def test_eager_metadata_only(prompt, length):
    kernels = [AllSlotKernel(CPUModel().eval()),
               SpawnGatedKernel(CPUModel().eval(), spawn_threshold=None)]
    runs = []
    for k in kernels:
        s = create(k, prompt=prompt, length=length)
        offers = (64, 4, 8, 64, 32, 16)
        k.run(s, lambda tick: offers[tick.tick % len(offers)])
        assert quote(k, s, 64)["physical_q"] == 0
        runs.append((k.output(s).tolist(), [{a:b for a,b in e.items() if a != "tail_admission"}
                                             for e in s.events]))
    assert runs[0] == runs[1]
