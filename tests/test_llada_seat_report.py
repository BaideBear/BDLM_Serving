import pytest

from experiments.t2t.serving_slots.llada_seat_report import paired, read_ibs, window_rows


def test_actual_window_rows_are_an_exclusive_physical_partition():
    r=dict(Q=16,row_visits=dict(dirty=3,cold=2,clean_unresolved=6,prefix=1),
           window_interior=4)
    assert sum(window_rows(r).values())==16
    with pytest.raises(ValueError,match="unclassified"):
        window_rows(dict(r,Q=17))


def test_stratified_intervals_recover_identical_work_and_reject_missing_domains():
    cases=[dict(dataset=g,a=dict(correct=True,N=i+1,F=2),b=dict(correct=True,N=i+1,F=2))
           for g in ("a","b","c") for i in range(3)]
    assert paired(cases,"a","b")==dict(correct_rate=[0.,0.],macro_TPF=[0.,0.])
    with pytest.raises(ValueError,match="datasets"):
        paired(cases[:6],"a","b")


def test_refuse_reformatted_or_unsealed_external_scores(tmp_path):
    path=tmp_path/"wrong.json"
    path.write_text("{}")
    with pytest.raises(ValueError,match="sealed"):
        read_ibs(path)
