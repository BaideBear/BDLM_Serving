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


def test_quarter_profile_requires_action_exposure_at_the_correct_offer(cohort):
    path, manifest=cohort
    for p in path.glob("*.json"):
        r=json.loads(p.read_text())
        r["policy_checks"]={"fixed16":{"borrow_Q":3},"fixed32":{"full_write_F":1},
                            "fixed64":{"quarter_open_F":1}}.get(r["suite"],{})
        r.update(profile="quarter_fullwrite",downshifts_with_debt=int(r["suite"]=="switch25"))
        p.write_text(json.dumps(r))
    assert summarize(path,"llada",manifest)["profile"]=="quarter_fullwrite"
    p=path/"fixed4-00.json"
    r=json.loads(p.read_text())
    r["policy_checks"]={"borrow_Q":4}
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match="wrong fixed action exposure"):
        summarize(path,"llada",manifest)


@pytest.mark.parametrize("omitted", ["terminal_writes", "warm_sample_rows"])
def test_edit_warm_profile_requires_observed_edits_and_warm_sampling(cohort, omitted):
    path, manifest=cohort
    for p in path.glob("*.json"):
        r=json.loads(p.read_text())
        r["policy_checks"]={"fixed16":{"borrow_Q":2},
                            "fixed32":{"edit_once_F":2,"terminal_writes":1},
                            "fixed64":{"quarter_open_F":1,"warm_sample_rows":2}}.get(r["suite"],{})
        r.update(profile="quarter_edit_warm",downshifts_with_debt=int(r["suite"]=="switch25"))
        p.write_text(json.dumps(r))
    assert summarize(path,"llada",manifest)["profile"]=="quarter_edit_warm"
    for p in path.glob("fixed*.json"):
        r=json.loads(p.read_text())
        r["policy_checks"].pop(omitted,None)
        p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match="wrong fixed action exposure"):
        summarize(path,"llada",manifest)


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


@pytest.mark.parametrize("repeat",[False,True])
def test_switchable_window_requires_real_fixed_changes_and_a_parked_write(cohort,repeat):
    path,manifest=cohort
    profile="quarter_window_repeat64" if repeat else "quarter_edit_window"
    for p in path.glob("*.json"):
        r=json.loads(p.read_text())
        s=r["suite"]
        r.update(profile=profile, downshifts_with_debt=int(s=="switch25"))
        r["policy_checks"]=({"window_F":1} if s in ("fixed4","fixed8","fixed16") else {
            "fixed32":{"edit_once_F":2,"terminal_writes":1},
            "fixed64":{"quarter_open_F":1,"warm_sample_rows":2},
            "park":{"parked_dirty_Q":1}}.get(s,{}))
        if s=="park":
            r.update(directed_pairs={"64>4":1,"4>64":1},F=3,decoder_F=3)
        if repeat and s=="fixed64": r["policy_checks"]["repeat_Q"]=1
        p.write_text(json.dumps(r))
    if repeat:
        with pytest.raises(ValueError,match="live vote"):
            summarize(path,"llada",manifest)
        p=path/"switch25-00.json"
        r=json.loads(p.read_text());r["policy_checks"]["discard_on_downshift"]=1
        p.write_text(json.dumps(r))
    assert summarize(path,"llada",manifest)["profile"]==profile
    for p in path.glob("park-*.json"):
        r=json.loads(p.read_text())
        r["policy_checks"]={}
        p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match="parked-tail"):
        summarize(path,"llada",manifest)


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


def test_paid16_profile_requires_paid_q16_and_a_real_parked_tail(cohort):
    path, manifest = cohort
    profile = "quarter_window_repeat64_paid16"
    for p in path.glob("*.json"):
        r = json.loads(p.read_text())
        s = r["suite"]
        r.update(profile=profile,
                 downshifts_with_debt=int(s in ("switch25", "park", "park16")))
        r["policy_checks"] = {
            "fixed4": {"window_F": 1},
            "fixed8": {"window_F": 1},
            "fixed16": {"window_F": 1, "paid_mask_rows": 2},
            "fixed32": {"edit_once_F": 2, "terminal_writes": 1},
            "fixed64": {"quarter_open_F": 1, "warm_sample_rows": 2, "repeat_Q": 1},
            "switch25": {"discard_on_downshift": 1, "paid_mask_rows": 1},
            "park": {"parked_dirty_Q": 1},
        }.get(s, {})
        if s == "park":
            r.update(directed_pairs={"64>4": 1, "4>64": 1}, F=3, decoder_F=3)
        p.write_text(json.dumps(r))
    for i in (0, 10, 20):
        row = dict(index=i, seed=i, dataset="unit", task_id=str(i),
                   suite="park16", schema="strict-handle-llada-qualification-v1",
                   samples_sha256="digest", F=3, decoder_F=3, encoder_F=0,
                   encoder_Q=0, decoder_Q=96, decoder_projection_modules=20,
                   downshifts_with_debt=1, parked_rows_paid_on_downshift=1,
                   gpu="test", identical=True, profile=profile,
                   directed_pairs={"64>16": 1, "16>64": 1},
                   policy_checks={"parked_dirty_Q": 1, "paid_mask_rows": 1})
        (path / f"park16-{i:02}.json").write_text(json.dumps(row))
    result = summarize(path, "llada", manifest)
    assert result["profile"] == profile
    assert result["arms"]["park16"]["directed_pairs"] == {"64>16": 3, "16>64": 3}
    for p in path.glob("fixed16-*.json"):
        r = json.loads(p.read_text())
        r["policy_checks"].pop("paid_mask_rows", None)
        p.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="wrong fixed action exposure"):
        summarize(path, "llada", manifest)
    for p in path.glob("fixed16-*.json"):
        r = json.loads(p.read_text())
        r["policy_checks"]["paid_mask_rows"] = 2
        p.write_text(json.dumps(r))
    for p in path.glob("park16-*.json"):
        r = json.loads(p.read_text())
        r["policy_checks"] = {"paid_mask_rows": 1}
        p.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="parked-tail"):
        summarize(path, "llada", manifest)


def paid16_cohort(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    tiers = (4, 8, 16, 32, 64)
    tasks = [dict(index=i, seed=i, dataset="unit", task_id=str(i)) for i in (0, 10, 20)]
    for t in tasks:
        for suite, path in [
            *((f"fixed{q}", (q, q)) for q in tiers),
            ("switch25", offer_cycle(tiers) + (tiers[0],)),
            ("park", (64, 4)),
        ]:
            row = dict(**t, suite=suite, schema="strict-handle-llada-qualification-v1",
                       samples_sha256="digest", F=len(path), decoder_F=len(path),
                       encoder_F=0, encoder_Q=0, decoder_Q=sum(path),
                       decoder_projection_modules=20, downshifts_with_debt=0,
                       parked_rows_paid_on_downshift=0, gpu="test", identical=True,
                       directed_pairs=dict(Counter(f"{a}>{b}" for a, b in zip(path, path[1:]))))
            (tmp_path / f"{suite}-{t['index']:02}.json").write_text(json.dumps(row))
    for i in (0, 10, 20):
        row = dict(index=i, seed=i, dataset="unit", task_id=str(i),
                   suite="park16", schema="strict-handle-llada-qualification-v1",
                   samples_sha256="digest", F=3, decoder_F=3, encoder_F=0,
                   encoder_Q=0, decoder_Q=96, decoder_projection_modules=20,
                   downshifts_with_debt=1, parked_rows_paid_on_downshift=1,
                   gpu="test", identical=True,
                   directed_pairs={"64>16": 1, "16>64": 1})
        (tmp_path / f"park16-{i:02}.json").write_text(json.dumps(row))
    for p in tmp_path.glob("*.json"):
        r = json.loads(p.read_text())
        s = r["suite"]
        r.update(profile="quarter_window_repeat64_paid16",
                 downshifts_with_debt=int(s in ("switch25", "park", "park16")))
        r["policy_checks"] = {
            "fixed4": {"window_F": 1},
            "fixed8": {"window_F": 1},
            "fixed16": {"window_F": 1, "paid_mask_rows": 2},
            "fixed32": {"edit_once_F": 2, "terminal_writes": 1},
            "fixed64": {"quarter_open_F": 1, "warm_sample_rows": 2, "repeat_Q": 1},
            "switch25": {"discard_on_downshift": 1, "paid_mask_rows": 1},
            "park": {"parked_dirty_Q": 1},
            "park16": {"parked_dirty_Q": 1},
        }.get(s, {})
        if s == "park":
            r.update(directed_pairs={"64>4": 1, "4>64": 1}, F=3, decoder_F=3)
        p.write_text(json.dumps(r))
    return tmp_path, dict(samples_sha256="digest", samples=tasks)


def test_paid16_report_requires_park16_and_switch25_paid(tmp_path):
    path, manifest = paid16_cohort(tmp_path / "ok")
    assert summarize(path, "llada", manifest)["profile"] == "quarter_window_repeat64_paid16"
    for p in path.glob("park16-*.json"):
        p.unlink()
    with pytest.raises(ValueError, match="missing park16"):
        summarize(path, "llada", manifest)

    path, manifest = paid16_cohort(tmp_path / "foreign")
    for p in path.glob("*.json"):
        r = json.loads(p.read_text())
        r["profile"] = "quarter_window_repeat64"
        p.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="unexpected park16"):
        summarize(path, "llada", manifest)

    path, manifest = paid16_cohort(tmp_path / "nopaid")
    for p in path.glob("switch25-*.json"):
        r = json.loads(p.read_text())
        r["policy_checks"].pop("paid_mask_rows", None)
        p.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="no paid Q16 warm MASK"):
        summarize(path, "llada", manifest)
