import pytest

from tools.shared_screen_sweep import CHOICES, build, order


BASE = dict(stage="formal90", cap=2048)


def test_phase_changes_only_one_factor_per_model():
    panel = build(BASE, "prefix")
    for model, arms in panel["arms"].items():
        assert tuple(arms) == CHOICES
        assert all(p == dict(prefix_share=d, open_at="1/4") for d,p in arms.items())
    panel = build(BASE, "open", "0", "1/2")
    for model, a in (("llada21", "0"), ("gemma", "1/2")):
        assert all(p == dict(prefix_share=a, open_at=d) for d,p in panel["arms"][model].items())


def test_every_case_gets_all_twenty_offers_in_rotated_order():
    shard = [dict(index=i) for i in (0,12,24)]
    offers = (4,8,16,32,64)
    rows = [(r["index"],q,d) for r,q,d in order(shard, offers)]
    assert len(rows) == len(set(rows)) == 60
    assert set(rows) == {(r["index"],q,d) for r in shard for q in offers for d in CHOICES}
    assert rows[0][1:] != rows[20][1:]


@pytest.mark.parametrize("base,axis,a,b", [
    (dict(stage="screen18",cap=2048), "prefix",None,None),
    (BASE,"prefix","1/2",None), (BASE,"open","1/8",None), (BASE,"open","1/16","1/4"),
])
def test_reject_unregistered_sweep(base,axis,a,b):
    with pytest.raises(ValueError):
        build(base,axis,a,b)
