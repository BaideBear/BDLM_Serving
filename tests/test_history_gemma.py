"""CPU regressions for Gemma history admission and endpoint gating."""
from copy import deepcopy

import torch

from bdlm_serving.methods.t2t.history_gemma import Rows, _bounds, _eligible_sampler
from tools.run_history_gemma import endpoint_case, quantiles


def test_cold_priority_dirty_age_and_epoch_reset():
    rows = Rows(8)
    unresolved = [False] * rows.width
    assert rows.select(2, unresolved, False) == ()
    assert rows.select(2, unresolved, True) == (0, 1)
    assert rows.refresh((0, 1), 0) == []
    rows.write((1,), 0)
    assert rows.select(2, [True, True, *unresolved[2:]], True) == (1, 2)
    rows.write((1, 0), 1)
    assert rows.select(1, unresolved, True) == (1,)
    assert rows.refresh((1,), 2) == [2]
    assert rows.dirty == {0: 1}
    rows.write((1,), 2)
    assert rows.select(1, unresolved, True) == (0,)
    rows.invalidate()
    assert not rows.valid and not rows.last_read
    assert rows.select(4, unresolved, True) == (0, 1, 2, 3)
    assert rows.select(8, unresolved, True) == rows.select(16, unresolved, True)
    assert rows.select(2, unresolved, True, force_full=True) == tuple(range(8))


def test_two_distinct_legal_ranges_and_empty_history():
    mask = torch.tensor([[[False, True, True, False], [False, False, False, False]]])
    bounds, counts = _bounds(mask)
    assert counts == (2, 0)
    assert bounds[0, 0].tolist() == [1, 3]
    assert bounds[0, 1, 0] == bounds[0, 1, 1]
    assert _bounds(mask[..., :0]) == (None, (0, 0))


def test_cold_sampler_preserves_native_rng_and_warm_edits():
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
        EntropyBoundSampler, EntropyBoundSamplerConfig)

    canvas = torch.tensor([[0, 1, 2, 3]])
    proposal = torch.tensor([[4, 4, 4, 4]])
    logits = torch.zeros((1, 4, 5))
    config = EntropyBoundSamplerConfig(100.)
    native, guarded = (EntropyBoundSampler(config, 4, 5, 8) for _ in range(2))

    def stop(*_):
        raise AssertionError("Partly cold canvas must not update the native stopper")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        native.renoise_canvas(native.accept_canvas(canvas, proposal, logits, 7), 7)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(17)
        with _eligible_sampler(guarded, canvas, [1, 3], stop) as guard:
            accepted = guarded.accept_canvas(canvas, proposal, logits, 7)
            assert accepted.tolist() == [[0, 4, 2, 4]]
            assert guarded.accepted_token_mask.tolist() == [[False, True, False, True]]
            guarded.renoise_canvas(accepted, 7)
            assert not guard(proposal, logits).item()
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert "accept_canvas" not in vars(guarded)


def test_numeric_and_trajectory_gates_are_independent():
    trace = dict(q=256, history_q=256, reused=0, close_excess=0, block=0,
                 clock=1, canvas="same", draft="same", stop_candidate=True,
                 closed=True, encoder_version=1, invalidated=True)
    probe = dict(max_abs=0., rel_rmse=0.)
    result = dict(index=0, trace=[trace], probes=[dict(attention={0: probe}, logits=probe)],
                  events=[["decoder", 256]], output_ids=[1], decoder_steps_per_canvas=[1],
                  generation_forwards=1, eos=True, valid_output_rows_in_last_canvas=1)
    native, full, double = (deepcopy(result) for _ in range(3))
    native["trace"][0]["draft"] = "different"
    case = endpoint_case(native, full, double)
    assert case["numerical_pass"] and case["saturated_endpoints_equal"]
    assert not case["native_trajectory_equal"] and case["first_divergent_round"] == 0
    full["probes"][0]["logits"]["max_abs"] = 9
    assert not endpoint_case(native, full, double)["numerical_pass"]
    assert quantiles([32, 32, 256]) == dict(p50=32, p95=256, max=256)
