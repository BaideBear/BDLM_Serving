import pytest

from experiments.t2t.serving_slots.gemma_warm_gap import ROOT, account, threshold


def test_uniform_fraction_is_macro_and_impossible_targets_are_not_forecasts():
    cases=[dict(N=10,F=10,warm_F=4),dict(N=6,F=4,warm_F=2)]
    assert threshold(cases,(10/8+6/3)/2)==pytest.approx(.5)
    assert threshold(cases,10) is None and threshold(cases,.5)==0


def test_all_phase_sums_reconcile_to_same_frozen_paths():
    rows=account(ROOT,ROOT.parents[3]/"benchmarks/frozen90/manifest.json")
    for q in (32,64):
        a,b,c=[r for r in rows if r["q"]==q]
        assert (a["n"],b["n"],c["n"])==(30,60,90)
        for k in ("F","N","correct","decoder_Q","cold_F","warm_F","verify_F","encoder_F"):
            assert a[k]+b[k]==c[k]
        assert c["macro_TPF"]<c["free_close_drain_ceiling"]<c["capped_IBS_TPF"]
        assert 0<c["uniform_warm_F_reduction_fraction"]<1
