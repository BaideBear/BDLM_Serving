"""One external decision clock, using model-specific reference executors."""
from dataclasses import replace

import pytest
import torch

from experiments.t2t.serving_slots.request_handle import open_llada, open_gemma
from experiments.t2t.serving_slots.llada_all_slots import AllSlotKernel
from test_llada_all_slots import TapedCPUModel, CFG
from test_gemma_request import B, inputs, model_factory, plan, reference


LL_OFFERS = (64, 4, 8, 16, 32)


def make_ll(kernel, seed):
    return open_llada(kernel, torch.full((1, 34 if seed == 1 else 28), 5), CFG,
                      62, seed, 0, ())


def advance_ll(kernel, h, offer):
    m = kernel.model
    before = len(m.calls)
    menu = h.menu()
    assert h.menu() is menu and len(m.calls) == before
    price = menu[offer]
    step = h.advance(price)
    tape = [len(pos) for pos, _ in m.calls[before:]]
    assert tape == ([step.prefill_q] if step.prefill_q else []) + [step.main_q]
    assert len(tape) == step.total_f and step.paid_f == price.paid_f_max == 1
    assert step.paid_q == price.paid_q_max
    return step, m.calls[before:]


def solo(seed):
    k = AllSlotKernel(TapedCPUModel().eval())
    h = make_ll(k, seed)
    trace = []
    while h.status == "ready":
        trace.append(advance_ll(k, h, LL_OFFERS[len(trace) % 5]))
        assert len(trace) < 300
    return h.result(), trace


@pytest.mark.parametrize("cancel", [False, True])
def test_llada_two_requests_on_one_kernel(cancel):
    refs = [solo(s) for s in (1, 2)]
    k = AllSlotKernel(TapedCPUModel().eval())
    a, b = make_ll(k, 1), make_ll(k, 2)
    assert a.opened_prefill_q == b.opened_prefill_q == 0
    aa, bb = a.menu()[64], b.menu()[64]
    assert aa.prefill_q == 32 and bb.prefill_q == 0
    for bad in (64, bb, replace(aa)):
        with pytest.raises(ValueError):
            a.advance(bad)
    assert not k.model.calls
    for tick in range(max(len(t) for _,t in refs)):
        for h, (output, trace) in zip((a,b), refs):
            if h.status != "ready":
                continue
            step, work = advance_ll(k, h, LL_OFFERS[tick%5])
            assert (step, work) == trace[tick]
            if h.status == "done":
                assert h.result() == output
            if tick == 1 and h is a and cancel:
                before = len(k.model.calls)
                h.close()
                h.close()
                assert h.status == "cancelled" and len(k.model.calls) == before
                with pytest.raises(RuntimeError):
                    h.result()
            with pytest.raises(ValueError):
                h.advance(aa)
    assert b.status == "done" and a.status == ("cancelled" if cancel else "done")


def test_llada_failed_forward_poisoned_not_replayed():
    k = AllSlotKernel(TapedCPUModel().eval())
    a,b = make_ll(k,1),make_ll(k,2)
    price = a.menu()[64]
    def fail(*_):
        raise ArithmeticError("injected projection failure")
    # A later layer fails after the first layer already wrote request-local KV.
    with k.projections[-1].register_forward_pre_hook(fail):
        with pytest.raises(ArithmeticError):
            a.advance(price)
    assert a.status == "failed"
    with pytest.raises(ValueError):
        a.advance(price)
    ids, trace = solo(2)
    for tick, want in enumerate(trace):
        assert advance_ll(k,b,LL_OFFERS[tick%5]) == want
    assert ids == b.result()


def test_gemma_handle_reconciles_open_and_optional_encoder(model_factory):
    path = (32,512,64,128,256)
    data = inputs(5)
    ids, trace = reference(model_factory(), data, 23, path)
    model = model_factory()
    h = open_gemma(model, data, plan(2*B), seed=23)
    assert h.opened_prefill_q == 5 and len(model.encoder_calls) == 1
    n_post = 0
    for i, ev in enumerate(trace):
        dc, ec = len(model.decoder_calls), len(model.encoder_calls)
        price = h.menu()[path[i%5]]
        step = h.advance(price)
        assert (step.tick,step.main_q,step.action) == (i,ev["q"],ev["phase"])
        assert step.post_q == len(ev["encoded"])*B <= price.post_q_max
        assert len(model.decoder_calls)-dc == 1
        assert len(model.encoder_calls)-ec == len(ev["encoded"])
        assert step.total_f == step.paid_f == 1 + len(ev["encoded"])
        n_post += len(ev["encoded"])
    assert n_post and h.result() == tuple(ids) and h.status == "done"
