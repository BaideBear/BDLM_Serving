"""Receipt validation on synthetic counts only; no model inference."""
from collections import Counter
import json
from itertools import product

import pytest

from experiments.t2t.serving_slots.strict_handle_qualify import offer_cycle, reference_path
from experiments.t2t.serving_slots.strict_handle_report import summarize


@pytest.mark.parametrize("tiers", [(4, 8, 16, 32, 64), (32, 64, 128, 256, 512)])
def test_wraparound_covers_each_pair_once(tiers):
    path = offer_cycle(tiers)
    assert path[0] == tiers[0]
    assert Counter(zip(path, path[1:] + path[:1])) == Counter(product(tiers, repeat=2))


def test_new_profiles_check_their_own_fixed_controls(tmp_path):
    for q in (32,64,128,256,512):
        path = reference_path(tmp_path,q,10,"fixed90_fused")
        expected = f"C{q}-profile-fuse1-10.json" if q in (128,512) else f"C{q}-fuse1-10.json"
        assert path.name == expected
        dense = reference_path(tmp_path,q,10,"fixed90_dense_fused")
        assert dense.name == ("C256-capacity1-10.json" if q==256 else expected)
        phase = reference_path(tmp_path,q,10,"phase_high_fused")
        assert phase.name == ("C256-density-high-10.json" if q==256 else expected)
        stop = reference_path(tmp_path,q,10,"dense_b_stop_fused")
        assert stop.name == ("C256-dense-stop1-10.json" if q==256 else expected)
        combined = reference_path(tmp_path,q,10,"dense_stop_promotion_fused")
        assert combined == (tmp_path/"promotion-1128"/"C512-promotion1-10.json" if q==512 else stop)
        joint = reference_path(tmp_path,q,10,"dense_stop_joint_verify")
        assert joint == (tmp_path/"joint-verify-1128"/"C512-joint1-10.json" if q==512 else combined)
        top = reference_path(tmp_path,q,10,"dense_stop_joint_single512")
        assert top == (tmp_path/"single512-stop-1128"/"C512-single1-10.json" if q==512 else joint)


@pytest.fixture
def cohort(tmp_path):
    tiers = (4, 8, 16, 32, 64)
    tasks = [dict(index=i, seed=i, dataset="unit", task_id=str(i)) for i in (0,10,20)]
    for t in tasks:
        for suite, path in [
            *((f"fixed{q}", (q, q)) for q in tiers),
            ("switch25", offer_cycle(tiers)+(tiers[0],)),
            ("park", (64, 4)),
        ]:
            row = dict(**t, suite=suite, schema="strict-handle-llada-qualification-v1",
                       samples_sha256="digest", F=len(path), decoder_F=len(path),
                       encoder_F=0, encoder_Q=0, decoder_Q=sum(path),
                       decoder_projection_modules=20, downshifts_with_debt=0,
                       parked_rows_paid_on_downshift=0, gpu="test", identical=True,
                       directed_pairs=dict(Counter(f"{a}>{b}" for a,b in zip(path, path[1:]))))
            (tmp_path/f"{suite}-{t['index']:02}.json").write_text(json.dumps(row))
    return tmp_path, dict(samples_sha256="digest", samples=tasks)


def test_count_only_consecutive_steps_in_each_request(cohort):
    path, manifest = cohort
    result = summarize(path, "llada", manifest)["arms"]
    assert result["park"]["directed_pairs"] == {"64>4": 3}
    assert sum(result["switch25"]["directed_pairs"].values()) == 75


def test_verified_draft_compares_through_eos_and_requires_complete_canvas():
    from experiments.t2t.serving_slots.gemma_draft_shadow import compare_block
    full=[10]*256
    a=full.copy()
    a[10]=1
    b=a.copy()
    b[11]=7
    assert compare_block(a,b,{1})==dict(changed=1,first_change=11,draft_eos=10,verify_eos=10,delivered_equal=True)
    b[3]=1
    assert not compare_block(a,b,{1})['delivered_equal']
    assert compare_block(full,full,{1})['verify_eos']==256
    with pytest.raises(ValueError):
        compare_block(a[:32],b[:32],{1})


@pytest.mark.parametrize("invalid", ["missing", "different", "lost-transition"])
def test_reject_unqualified_cohort(cohort, invalid):
    path, manifest = cohort
    f = path/"fixed4-00.json"
    if invalid == "missing":
        f.unlink()
    elif invalid == "different":
        row = json.loads(f.read_text())
        row["identical"] = False
        f.write_text(json.dumps(row))
    else:
        for f in path.glob("switch25-*.json"):
            row = json.loads(f.read_text())
            row["directed_pairs"]["4>4"] += row["directed_pairs"].pop("4>8")
            f.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        summarize(path, "llada", manifest)
