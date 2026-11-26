from copy import deepcopy
from types import SimpleNamespace
import pytest

from experiments.t2t.serving_slots.unified90 import ARMS, paired, work_row


def record(arm, *, exposed=False):
    first = ARMS[arm][0][0]
    step = dict(tick=0, offer=first, q=2, head=0, phase="sample", mandatory_q=1, overflow=0,
                encoded=[], promoted=[], prefix_selected=[], refresh_only=[], strict_edits=[[]],
                selected=[11, 12], sampled=[11,12], visible=[11,12] + ([267] if exposed else []),
                ages_before=[1], dirty_before=[[0]])
    obs = dict(all_projected_widths_match=True, actual_decoder_q=[2], F=1, Q=2, reencodes=0,
               decoder_layers=30, projection_modules=85, prefill=11)
    task = dict(index=0, dataset="demo", task_id="demo/0", seed=4)
    tok = SimpleNamespace(all_special_ids=[0,3], decode=lambda ids, **kw: "answer")
    cfg = SimpleNamespace(eos_token_id=[3], pad_token_id=0)
    return work_row(task, arm, [7,3], [step], obs, tok, cfg, 2048, "input")


def test_numerator_counts_delivered_tokens_not_edits_or_q():
    row = record("q512")
    assert (row["output_tokens"], row["tpf"], row["token_efficiency"]) == (1, 1., .5)
    assert row["first_opportunity"] is None


def test_identical_unexposed_request():
    on, off = record("replay"), record("replay_open")
    paired(on, off)
    off["output_ids"] = [12,3]
    with pytest.raises(ValueError, match="unexposed"):
        paired(on, off)


def test_after_first_visibility_divergence_is_not_filtered():
    on, off = record("replay", exposed=True), record("replay_open", exposed=True)
    assert on["first_opportunity"] == 0
    off["output_ids"] = [10,3]
    paired(on, off)
    bad = deepcopy(off)
    bad["first_geometry_sha256"] = "wrong"
    with pytest.raises(ValueError, match="geometry"):
        paired(on, bad)


def test_different_offers_are_not_a_mask_comparison():
    with pytest.raises(ValueError, match="mask-only"):
        paired(record("q512"), record("replay_open"))


def test_first_rank_change_may_not_remove_a_mask_or_dirty_row():
    from experiments.t2t.serving_slots.unified90 import check_rank_change
    a,b=record("q64_k16_random"),record("q64_k16_oldest")
    x=dict(optional=[5],selected=[5,10,12],pre=[[10]],phase="sparse",edits=[[]],
           head=0,ages=[2],prefix_end=9)
    y=dict(x,optional=[4],selected=[4,10,12])
    assert check_rank_change(a,b,[x,x],[x,y])==1
    assert check_rank_change(a,b,[x],[x]) is None
    with pytest.raises(ValueError,match="width"):
        check_rank_change(a,b,[x],[dict(y,selected=[4,10,11])])
    with pytest.raises(ValueError,match="prefix witness"):
        check_rank_change(a,b,[x],[dict(y,optional=[3])])
    with pytest.raises(ValueError,match="No rank intervention"):
        check_rank_change(a,dict(b,generation_forwards=7),[x],[x])
    with pytest.raises(ValueError,match="inputs differ"):
        check_rank_change(a,dict(b,input_sha256="other"),[x],[x])
