"""Synthetic receipt and report tests; no model, inference or scorer."""

import json

import pytest

from experiments.t2t.serving_slots import cross_race_report as report


SNAPSHOT = "a" * 64
BASE = "C512-gateoff-slack0-4-pref1of8"
TRIAL = "C512-gatequarter-slack0-4-pref1of8"
PROVENANCE = dict(source_sha256={f: "a"*64 for f in
                  ("cross_race.py", "gemma_fixed_dual.py", "gemma_fixed_dual_eval.py")},
                  scorer_sha256="b"*64)


def identity(i):
    return dict(index=i, dataset=("gsm8k", "math500", "humaneval")[(i % 30)//10],
                task_id=f"item/{i}", seed=17000+i)


def manifest():
    return dict(samples_sha256=SNAPSHOT, samples=[identity(i) for i in range(90)])


def cross(i, *, trial=False, correct=True, n=6, df=2, dq=10):
    tail = dict.fromkeys(report.TAIL, 0)
    if trial:
        tail.update(tail_open=2, tail_discard=1, tail_touched_positions=4,
                    tail_last_candidate=3, tail_final_matches=2,
                    discarded_tail_Q=2, tail_Q=5)
    categories = dict.fromkeys(report.CATEGORIES, 0)
    categories["unresolved"] = dq
    row = dict(**identity(i), schema="single-factor-cross-v1", samples_sha256=SNAPSHOT,
               input_sha256=f"{i:064x}", budget=512, gate="quarter" if trial else "off",
               slack=False, late_den=4, prefix=[1, 8], N=n, F=df, decoder_F=df,
               decoder_Q=dq, encoder_F=0, encoder_Q=0, max_decoder_Q=dq-df+1,
               history_query_rows=dq, joint_F=int(trial), truncated=False,
               categories=categories, route_F={"sample": df}, route_Q={"sample": dq},
               output_text="must not leak", output_ids=[999], **tail)
    if correct is None:
        row["score_pending"] = "must not leak pending error"
    else:
        row["correct"] = correct
    return row


def summarize(rows, **kwargs):
    return report.summarize(map(report.normalize, rows), comparisons=[(BASE, TRIAL)], **kwargs)


def test_fixed_cohorts_and_metric_denominators():
    rows = [
        cross(0, correct=False, n=4, df=1, dq=10),
        cross(0, trial=True, n=12, df=3, dq=40),
        cross(29, n=9, df=3, dq=30),
        cross(29, trial=True, correct=None, n=8, df=2, dq=10),
        cross(30, n=4),
        cross(30, trial=True, correct=False, n=8),
        cross(89, trial=True),
    ]
    rows[1].update(F=4, encoder_F=1, encoder_Q=256, truncated=True)
    out = summarize(rows, manifest=manifest(), draws=80, seed=17)
    comparison, = out["comparisons"]
    orig, new, full = (comparison["cohorts"][k] for k in report.COHORTS)
    assert out["status"] == comparison["status"] == "incomplete"
    assert out["manifest_verified"] and orig["paired_work"] == 2 and orig["paired_scored"] == 1
    assert orig["paired_score_pending_indices"] == [29]
    assert orig["quality"] == dict(reference_correct=0, trial_correct=1, rescue=1, loss=0,
                                   accuracy_delta=1)
    assert orig["work"]["reference"]["macro_TPF"] == (4/1 + 9/3) / 2
    assert orig["work"]["trial"]["N_per_history_Q"] == (12+8)/(40+10)
    assert orig["work"]["trial"]["physical_Q"] == 306
    assert orig["work"]["trial"]["peak_decoder_Q"] == 38
    assert orig["work"]["trial"]["tail_match_rate"] == 4/6
    assert orig["work"]["trial"]["discarded_tail_Q"] == 4
    assert orig["work"]["trial"]["truncated"] == 1
    assert orig["confidence95"]["macro_TPF_delta"] == [-1, 1]
    assert orig["confidence95"]["tail_match_rate_delta"] is None
    assert "accuracy_delta" not in orig["confidence95"]
    assert new["quality"]["loss"] == 1 and new["paired_work"] == new["trial_only"] == 1
    assert full["quality"]["rescue"] == full["quality"]["loss"] == 1
    assert full["confidence95"]["accuracy_delta"] == [-1, 1]
    assert out["arms"][TRIAL]["cohorts"]["frozen90"]["score_pending_indices"] == [29]
    assert out["arms"][BASE]["cohorts"]["frozen90"]["received"] == 3
    assert out == summarize(list(reversed(rows)), manifest=manifest(), draws=80, seed=17)
    assert "must not leak" not in json.dumps(out)


def test_full_coverage_counts_pending_as_incomplete():
    rows = [cross(i, trial=trial) for i in range(90) for trial in (False, True)]
    out = summarize(rows, manifest=manifest(), draws=0)
    assert out["status"] == "complete"
    assert [v["paired_scored"] for v in out["comparisons"][0]["cohorts"].values()] == [30, 60, 90]
    rows[-1].pop("correct")
    rows[-1]["score_pending"] = ""
    out = summarize(rows, manifest=manifest(), draws=0)
    full = out["comparisons"][0]["cohorts"]["frozen90"]
    assert (out["status"], full["paired_work"], full["paired_scored"]) == ("incomplete", 90, 89)


def test_seed_mismatch_never_pairs_and_can_omit_empty():
    left, right = cross(0), cross(0, trial=True)
    right["seed"] += 1
    out = summarize([left, right], draws=0)
    orig = out["comparisons"][0]["cohorts"]["original30"]
    assert orig["status"] == "no_common"
    assert orig["paired_work"] == 0 and orig["identity_mismatch_indices"] == [0]
    assert "quality" not in orig and "work" not in orig
    omitted = summarize([left, right], omit_empty=True)["comparisons"][0]
    assert omitted["cohorts"] == {} and omitted["omitted_no_common"] == list(report.COHORTS)
    with pytest.raises(ValueError, match="frozen identity mismatch"):
        summarize([left, right], manifest=manifest())


def test_pending_only_pair_has_work_but_no_quality():
    out = summarize([cross(0, correct=None), cross(0, trial=True)], draws=0)
    orig = out["comparisons"][0]["cohorts"]["original30"]
    assert orig["paired_work"] == 1 and orig["paired_scored"] == 0
    assert orig["status"] == "incomplete" and "quality" not in orig
    assert orig["work"]["trial"]["tail_Q"] == 5


@pytest.mark.parametrize("change,message", [
    ({"index": True}, "invalid index"),
    ({"correct": 1}, "Boolean correct"),
    ({"score_pending": "late"}, "ambiguous score_pending"),
    ({"max_decoder_Q": 513}, "decoder cap"),
    ({"max_decoder_Q": 10}, "accounting"),
    ({"history_query_rows": 11}, "accounting"),
    ({"route_Q": {"sample": 2}}, "route_Q"),
    ({"categories": dict.fromkeys(report.CATEGORIES, 0)}, "categories"),
    ({"tail_final_matches": 16}, "tail counters"),
    ({"gate": "not-registered"}, "unregistered arm"),
])
def test_malformed_receipts_fail_closed(change, message):
    with pytest.raises(ValueError, match=message):
        report.normalize(cross(0, trial=True) | change)


def test_duplicate_digest_and_snapshot_rejected():
    left, right = cross(0), cross(0, trial=True)
    with pytest.raises(ValueError, match="duplicate receipt"):
        summarize([left, left])
    with pytest.raises(ValueError, match="input digest mismatch"):
        summarize([left, right | {"input_sha256": "c"*64}])
    with pytest.raises(ValueError, match="multiple samples snapshots"):
        summarize([left, right | {"samples_sha256": "b"*64}])
    with pytest.raises(ValueError, match="distinct arms"):
        report.summarize([], comparisons=[(BASE, "C128-gateoff-slack0-4-pref1of8")])


def test_cross_and_gemma_layout_registry_matches_worker_grids():
    from experiments.t2t.serving_slots import cross_race, layout_sweep

    cross_arms = {report._cross(c, gate, slack, den, dose) for _, c, gate, slack, den, dose
                  in cross_race.grid()}
    sweep_arms = {report._sweep(c, layout, cold) for layout, cold, c, _ in layout_sweep.grid()}
    assert cross_arms == {a for a in report.ARMS.values() if a.kind == "cross"}
    assert sweep_arms == {a for a in report.ARMS.values() if a.kind == "layout_sweep"}
    ranks = {report._rank(c,r,u) for _,c,_,_,_,_,r,u in cross_race.cases("rank")}
    assert ranks == {a for a in report.ARMS.values() if a.kind == "rank"}


def test_rank_only_and_exposure_reconciled():
    base = cross(0)
    base.update(**PROVENANCE, schema="prefix-rank-v1", budget=32, rank="right", reuse=False,
                resample_paid_F=0, prefix_exposure=dict(prefix_seats=2, prefix_edits=0,
                prefix_revoked=0, differing_prefix_seats=0, prefix_distance_sum=3))
    base["categories"].update(unresolved=8,prefix=2)
    alt = base | {"rank":"oldest", "output_ids":[1001], "prefix_exposure":base["prefix_exposure"] | {"differing_prefix_seats":1}}
    r = report.summarize(map(report.normalize,(base,alt)), draws=0,
                         comparisons=[("C32-rankright-reuse0","C32-rankoldest-reuse0")])
    assert r["arms"]["C32-rankright-reuse0"]["budget_axis"] == "decoder_Q"
    assert r["comparisons"][0]["cohorts"]["original30"]["work"]["trial"]["differing_prefix_seats"] == 1
    assert r["comparisons"][0]["cohorts"]["frozen90"]["output_comparison"]["changed_output_indices"] == [0]
    assert r["sources"]["rank"] == PROVENANCE["source_sha256"]
    with pytest.raises(ValueError, match="source digest mismatch"):
        report.summarize(map(report.normalize, [base, alt | {"source_sha256":dict(
            PROVENANCE["source_sha256"], **{"cross_race.py":"c"*64})}]))
    with pytest.raises(ValueError,match="rank exposure"):
        report.normalize(alt | {"rank":"right"})
    with pytest.raises(ValueError,match="another factor"):
        report.normalize(alt | {"prefix":[1,4]})


def test_close_requires_exact_work_delta_as_well_as_equal_tokens():
    base = cross(0, df=3, dq=12) | dict(**PROVENANCE,budget=256,schema="paid-close-v1",fuse=False,
                                           fused_F=0,fused_dirty_Q=0)
    alt = cross(0) | dict(**PROVENANCE,budget=256,schema="paid-close-v1",fuse=True,fused_F=1,
                          fused_dirty_Q=2,route_F={"sample":1,"verify_paid":1})
    pair = [("C256-fuse0","C256-fuse1")]
    def parity(other):
        return report.summarize(map(report.normalize,(base,other)), comparisons=pair,draws=0)[
            "comparisons"][0]["cohorts"]["original30"]["parity"]
    assert all(not v for v in parity(alt).values())
    assert parity(alt | {"output_ids":[312]})["changed_output_indices"] == [0]
    assert parity(alt | {"fused_dirty_Q":3})["wrong_Q_delta_indices"] == [0]


def test_profile_close_has_distinct_pairs_and_fixed_dose_gate():
    from experiments.t2t.serving_slots.cross_race import profile_close_cases
    for c, gate, dose in ((128,"off",(1,4)),(512,"half",(1,8))):
        items = list(profile_close_cases(c))
        assert len(items)==len(set(items))==180
        assert {i for i,*_ in items} == set(range(90))
        for a, b in zip(items[::2],items[1::2]):
            assert a[:-1] == b[:-1] and a[-1]!=b[-1]
            assert a[1:-1]==(c,gate,False,4,dose,"right",False)
        row = cross(0) | dict(**PROVENANCE,schema="profile-paid-close-v1",
                              fuse=False, budget=c, gate=gate,prefix=dose,fused_F=0,fused_dirty_Q=0)
        assert report.normalize(row).arm.name == f"C{c}-profile-fuse0"
        with pytest.raises(ValueError, match="transfer profile"):
            report.normalize(row | {"prefix":(0,8)})


def test_capacity_study_checks_physical_cap_and_comparator(tmp_path):
    from experiments.t2t.serving_slots.cross_race import capacity_cases, replay_capacity
    assert len(set(capacity_cases()))==180
    r = cross(0) | dict(**PROVENANCE,budget=256,schema="capacity-head-v1",fuse=True,
                        full_head_at_b=True,fused_F=0,fused_dirty_Q=0,joint_F=0)
    ref = dict(r, schema="paid-close-v1",budget=512)
    p = tmp_path/"control.json"
    p.write_text(json.dumps(ref))
    r["replay"] = replay_capacity(r,p)
    assert r["replay"]["reference_budget"]==512
    assert report.normalize(r).arm.name=="C256-capacity1"
    p.write_text(json.dumps(dict(ref,max_decoder_Q=257)))
    with pytest.raises(ValueError,match="Unmatched prior"):
        replay_capacity(r,p)
    with pytest.raises(ValueError,match="parity"):
        report.normalize(dict(r,full_head_at_b=False))


def test_factor_publication_only_complete_cohorts_includes_negative_parity():
    from experiments.t2t.serving_slots.factor_tables import tables
    origins = [cross(i) | PROVENANCE | dict(budget=256,schema="paid-close-v1",fuse=False,
                fused_F=0,fused_dirty_Q=0) for i in range(31)]
    right = [r | {"fuse":True} for r in origins[:30]]
    right[0]["output_ids"] = [177]
    r = report.summarize(map(report.normalize, origins+right), manifest=manifest(), draws=0,
                         comparisons=[("C256-fuse0","C256-fuse1")])
    cov, data = tables(r, "close", manifest())
    assert [(t["cohort"], t["n"], t["correct"]) for t in data] == [("original30",30,30)]*2
    assert [c["status"] for c in cov] == ["complete", "no_common", "incomplete"]
    assert all(t["changed_output_indices"] == [0] and t["token_efficiency"] == .6 for t in data)
    with pytest.raises(ValueError, match="manifest"):
        tables(r, "close", manifest() | {"samples_sha256":"b"*64})


def test_density_registered_at_one_offer_and_checks_phases():
    from experiments.t2t.serving_slots.cross_race import density_cases
    cases=list(density_cases())
    assert len(set(cases))==360 and {c[1:8] for c in cases}=={(256,'off',False,4,(1,8),'right',False)}
    r=cross(0) | PROVENANCE | dict(schema="phase-density-v1",budget=256, density="high",fuse=True,
            fused_F=0,fused_dirty_Q=0, route_F={"sparse":2},route_Q={"sparse":10},
            density_F={"low_sparse":2})
    assert report.normalize(r).work["low_dense_F"]==0
    with pytest.raises(ValueError, match="density route"):
        report.normalize(r | {"density_F":{"low_dense_single":2}})
    with pytest.raises(ValueError, match="conservation"):
        report.normalize(r | {"density_F":{"low_sparse":1}})


def test_gemma_published_fixed_offers_and_denominators():
    from experiments.t2t.serving_slots.plot_fixed_gemma_profile import collect, COHORTS, TIERS
    data=collect()
    for cohort,_,n in COHORTS:
        part=[r for r in data if r["cohort"]==cohort]
        assert len(part)==16 and all(r["n"]==n for r in part)
        points=[r for r in part if r["method"].startswith("dense at ")]
        assert len(points)==2 and all(r["budget"]==r["peak_decoder_Q"]==256 for r in points)
        if n==90:
            assert {r["method"]:r["correct"] for r in points}=={"dense at high MASK":83,"dense at low MASK":84}
        assert {r["budget"] for r in part if r["method"]=="Optimus IBS"}==set(TIERS)
        assert next(r for r in part if r["method"]=="Flow W2")["token_efficiency"] is None
        for r in part:
            if r["history_Q"]:
                assert r["token_efficiency"]==pytest.approx(r["N"]/r["history_Q"])


def test_layout_uses_history_cap_not_actual_projection():
    shared = dict(**identity(0), samples_sha256=SNAPSHOT, output_tokens=8,
                  generation_forwards=3, truncated=False, correct=True)
    llada = dict(shared, budget=4, layout="split", cold_share=50, prefix_den=8,
                 generation_physical_padded_rows=96, history_query_rows=39,
                 history_q_per_forward=[4, 3, 32], denoise_history_rows=7,
                 commit_history_rows=32, local_attention_Q=96)
    l = report.normalize(llada)
    l_trial = report.normalize(llada | {"cold_share": 0, "correct": False})
    assert l.work["max_history_Q"] == 32 > l.arm.budget
    gemma = dict(shared, schema="rect-cold-v1", budget=32, layout="halo", cold_share=50,
                 generation_physical_padded_rows=768, history_query_rows=144,
                 local_attention_Q=288, close_history_excess=96, scorer_sha256="f"*64,
                 events=[["encoder", 100], ["decoder", 256], ["encoder", 256], ["decoder", 256]],
                 trace=[dict(q=256, history_q=16, close=False, local_attention_q=32),
                        dict(q=256, history_q=128, close=True, local_attention_q=256)])
    g = report.normalize(gemma)
    g_ref = report.normalize(gemma | {"layout": "split", "local_attention_Q": 512,
                                      "trace": [{**s, "local_attention_q": 256} for s in gemma["trace"]]})
    assert g.work["max_decoder_Q"] == 256 > g.arm.budget
    assert g.work["encoder_Q"] == 256 and g.work["decoder_Q"] == 512
    assert g.work["close_history_excess"] == 96
    out = report.summarize([l,l_trial,g,g_ref], draws=0,
                           comparisons=[(l.arm.name,l_trial.arm.name),(g_ref.arm.name,g.arm.name)])
    assert all(v["budget_axis"] == "history_Q" for v in out["arms"].values())
    l_pair, g_pair = [x["cohorts"]["original30"] for x in out["comparisons"]]
    assert l_pair["quality"]["loss"] == 1
    assert l_pair["work"]["trial"]["commit_history_rows"] == 32
    assert l_pair["work"]["trial"]["over_history_budget_F"] == 1
    assert g_pair["work"]["delta"]["local_attention_Q"] == -224
    assert "tail_Q" not in g_pair["work"]["trial"]
    with pytest.raises(ValueError, match="scorer digest mismatch"):
        report.summarize([g, report.normalize(gemma | {"layout": "split", "local_attention_Q": 512,
                        "scorer_sha256": "0"*64, "trace": [{**s, "local_attention_q": 256}
                                                         for s in gemma["trace"]]})])


def test_cli_incomplete_report_is_stdout_only(monkeypatch, capsys):
    pinned = json.loads(report.MANIFEST.read_text())
    monkeypatch.setattr(report, "_read_trials", lambda paths: iter(()))
    status = report.main(["ignored", "--compare", BASE, TRIAL,
                          "--require-complete", "--omit-empty", "--bootstrap", "0"])
    assert status == 1
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert not captured.err and out["status"] == "incomplete"
    assert out["samples_sha256"] == pinned["samples_sha256"]
    assert out["comparisons"][0]["omitted_no_common"] == list(report.COHORTS)
