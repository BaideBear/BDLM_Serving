"""Model-free contracts for the Gemma cold-first-retry factor."""
from types import SimpleNamespace

import pytest

pytest.importorskip("transformers.models.diffusion_gemma.generation_diffusion_gemma")
from experiments.t2t.serving_slots.gemma_cold_first_retry import (
    ColdFirstRetryRequest, RecordingVerifiedEOSRequest, retry_sample)
from experiments.t2t.serving_slots.gemma_cold_first_retry_eval import (
    cohort, first_difference, summarize)
from experiments.t2t.serving_slots.gemma_verified_eos import VerifiedEOSRequest


def canvas(*, dirty=(), initialized=(), accepted=(), can_sample=True, phase="sample", ordinal=0):
    return SimpleNamespace(
        ordinal=ordinal,
        rows=SimpleNamespace(dirty=set(dirty), initialized=set(initialized),
                             accepted=set(accepted)),
        can_sample=can_sample,
        sampler=SimpleNamespace(phase=phase),
    )


def test_retry_samples_only_unresolved_dirty_rows_on_the_first_cold_drain():
    head = canvas(dirty={0, 1}, initialized={0}, accepted={0})
    assert retry_sample([head], (0, 1), set(), "drain", tail=None,
                        ordinal=0, origin=0, retried=set()) == {1}


def test_retry_requires_a_pure_cold_drain_before_full_initialization():
    head = canvas(dirty={0, 1}, initialized={0}, accepted={0})
    base = dict(tail=None, ordinal=0, origin=0, retried=set())
    assert retry_sample([head], (0, 1), {1}, "drain", **base) is None
    assert retry_sample([head], (0, 1), set(), "sparse", **base) is None
    assert retry_sample([head], (0,), set(), "drain", **base) is None
    assert retry_sample([head], (0, 1), set(), "drain", tail=object(), ordinal=0,
                        origin=0, retried=set()) is None
    assert retry_sample([head], (0, 1), set(), "drain", tail=None,
                        ordinal=0, origin=0, retried={0}) is None
    warm = canvas(dirty={0, 1}, initialized=set(range(256)), accepted={0})
    assert retry_sample([warm], (0, 1), set(), "drain", **base) is None
    stopped = canvas(dirty={0, 1}, initialized={0}, accepted={0}, can_sample=False)
    assert retry_sample([stopped], (0, 1), set(), "drain", **base) is None
    closing = canvas(dirty={0, 1}, initialized={0}, accepted={0}, phase="closing")
    assert retry_sample([closing], (0, 1), set(), "drain", **base) is None
    resolved = canvas(dirty={0, 1}, initialized={0}, accepted=set(range(256)))
    assert retry_sample([resolved], (0, 1), set(), "drain", **base) is None


def event(*, route, sampled, age_after, written, dirty_after):
    return dict(tick=0, ordinal=0, route=route, offer=32, q=32, positions=list(range(32)),
                dirty_before_absolute=list(range(32)), written_absolute=written,
                dirty_after_absolute=dirty_after, sampled_absolute=sampled,
                parts=[dict(phase="sample" if sampled else "drain", age_before=0,
                            age_after=age_after, sampled=sampled, admitted=sampled,
                            edited=[], revoked=[], input_changed=[])])


def test_first_difference_accepts_only_the_matched_cold_retry():
    control_req = SimpleNamespace(origin=0, events=[event(
        route="drain", sampled=[], age_after=0, written=[], dirty_after=[])],
        pre_states=[dict(ordinal=0, phase="sample", tail=False,
                         accepted=set(range(32)) - {5})])
    treatment_req = SimpleNamespace(origin=0, events=[event(
        route="cold_first_retry", sampled=[5], age_after=1, written=[5], dirty_after=[5])])
    control = dict(output_ids=[1], F=1)
    treatment = dict(output_ids=[1], F=1)
    first = first_difference(control_req, treatment_req, control, treatment)
    assert first["added"] == [5] and first["admitted"] == [5]

    bad = SimpleNamespace(origin=0, events=[event(
        route="drain", sampled=[5], age_after=1, written=[5], dirty_after=[5])],
        pre_states=control_req.pre_states)
    with pytest.raises(Exception):
        first_difference(control_req, bad, control, treatment)


def test_cohort_uses_macro_tpf_and_history_q():
    row = dict(first=dict(tick=1), cold_F=1, warm_F=2, new_admits=3, new_edits=0,
               new_revoked=0, new_input_changed=0, treatment_drain_F=4,
               control=dict(correct=True, N=10, F=2, decoder_Q=40, encoder_Q=0),
               treatment=dict(correct=True, N=11, F=2, decoder_Q=40, encoder_Q=0))
    c = cohort({0: row}, [0])
    assert c["complete"] and c["exposed"] == 1
    assert c["control"]["macro_TPF"] == 5.0
    assert c["treatment"]["macro_TPF"] == 5.5
    assert c["control"]["N_per_history_Q"] == 10 / 40
    assert c["cold_F"] == 1 and c["warm_F"] == 2 and c["new_admits"] == 3
    s = summarize({0: row}, 32, "GPU-test", {}, "samples", "scorer")
    assert s["cohorts"]["pilot3"]["present"] == 1
    assert not s["cohorts"]["pilot3"]["complete"]
    assert not s["cohorts"]["all90"]["complete"]


def test_request_classes_are_opt_in_subclasses():
    assert issubclass(ColdFirstRetryRequest, RecordingVerifiedEOSRequest)
    assert issubclass(RecordingVerifiedEOSRequest, VerifiedEOSRequest)
