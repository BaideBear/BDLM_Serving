import json

import numpy as np
import pytest

from bdlm_serving.evaluation.frozen import sha
from bdlm_serving.methods.t2t.embedding_context import (
    CALIBRATION, DESIGN, ContextRanker, read_context_weights)
from bdlm_serving.methods.t2t.hazard import choose
from bdlm_serving.methods.t2t.llada_sparse import LLaDASparseKernel
from bdlm_serving.methods.t2t.protocol import Arm
from test_t2t_llada_sparse import _GateToy, _state


def test_bounded_arms_and_artifact(tmp_path):
    for q, e in ((4, 1), (8, 1), (16, 2)):
        assert Arm.parse(f'{q}:0:{q-e}:hard_context_fixed', 32).edit == e
    for block, spec in ((32, '8:0:6'), (32, '4:1:2'), (32, '32:0:31'), (256, '4:0:3')):
        with pytest.raises(ValueError):
            Arm.parse(spec + ':hard_context_fixed', block)
    with pytest.raises(ValueError):
        read_context_weights(None)
    artifact = dict(**DESIGN, weights=[0.] * 28, calibration_sha256=sha(CALIBRATION))
    path = tmp_path / 'weights.json'
    path.write_text(json.dumps(artifact))
    assert len(read_context_weights(path)) == 28
    for bad in (dict(radius=7), dict(weights=[0.] * 27), dict(weights=[np.nan] * 28),
                dict(calibration_sha256='wrong')):
        path.write_text(json.dumps({**artifact, **bad}))
        with pytest.raises(ValueError):
            read_context_weights(path)


def test_exact_fixed_allocator_when_ranks_equal():
    rng = np.random.default_rng(13)
    for _ in range(160):
        valid = list(range(int(rng.integers(0, 31)), 32))
        resolved = rng.integers(0, 2, size=32).tolist()
        fresh = {i for i in valid if resolved[i] and rng.random() < .7}
        w = rng.integers(-3, 4, 25)
        ranker = ContextRanker(lambda _: np.ones(16) / 4, np.r_[w, 0., 0., 0.])
        for q, e in ((4, 1), (8, 1), (16, 2)):
            args = valid, resolved, fresh, [-1]*32, fresh, 4, q, e
            expected = choose(*args, w, 0.)
            actual = choose(*args, None, 0., score_rows=lambda old, x:
                            ranker.scores([1]*32, fresh, old, x))
            assert actual == expected


def test_live_canvas_only_gathered_rows_and_last_accepted_writes():
    model = _GateToy().eval()
    live = []
    class Recorder:
        def scores(self, before, fresh, old, position):
            live.append((before.copy(), set(fresh), list(old), len(model.embedding_rows)))
            return old  # predictable highest-index reserve; no label or model access

    kernel = LLaDASparseKernel(model, 4, mask_slots=3, policy='hard_context_fixed',
                               context_ranker=Recorder())
    state = _state(kernel, gen_length=33)
    kernel.step(state)  # prefill
    state.x[0, 34:40] = 1
    previous_fresh = set()
    selected = 0
    while state.phase != 'done':
        start = 32 * state.block
        before = state.x[0, start:start+32].tolist()
        num_calls = len(model.embedding_rows)
        n = len(live)
        phase = kernel.step(state)
        if phase != 'denoise':
            assert len(live) == n
            previous_fresh = set()
            continue
        event = state.events[-1]
        picked = {i - start for i in event['absolute_positions']}
        assert event['F'] == 1 and event['layer_physical_q'] == [len(picked)] * 2
        assert len(model.embedding_rows) == num_calls + 1 and model.embedding_rows[-1] == len(picked)
        after = state.x[0, start:start+32].tolist()
        assert all(before[i] == after[i] for i in set(range(32)) - picked)
        if len(live) != n:
            assert event['stage'] == 'selected'
            ids, fresh, old, at = live[-1]
            assert ids == before and fresh == previous_fresh and at == num_calls
            assert all(ids[i] != state.mask_id and i+start >= state.prompt_length for i in old)
            assert fresh.isdisjoint(old) and fresh <= picked
            assert max(old) in picked and max(old)+start in event['reserved_extra_edit_positions']
            selected += 1
        previous_fresh = {i for i in picked if before[i] != after[i] and after[i] != state.mask_id}
    assert selected > 0
