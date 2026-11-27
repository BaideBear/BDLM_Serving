import pytest
from experiments.t2t.serving_slots.row_wave import Call, pack, waves


def test_physical_shape_not_offer_or_sum():
    widths = [8]*64+[256]*3+[128,128,65,1,600]
    groups = list(pack([Call(i,"decoder",v,None) for i,v in enumerate(widths)]))
    assert sorted(c.lane for g in groups for c in g) == list(range(len(widths)))
    for g in groups:
        paid = max(c.width for c in g)*len(g)
        assert paid <= 512 or (len(g)==1 and g[0].width==600)
        assert 8*sum(c.width for c in g) >= 7*paid
    assert any(len(g)==64 for g in groups)


def test_waves_preserve_lane_order_and_early_retirement():
    def lane(i):
        def run(issue):
            return [issue("decoder",128,(i,t)) for t in range(i%3+1)]
        return run
    seen = []
    def execute(group):
        seen.append([c.lane for c in group])
        return [c.data for c in group]
    got = waves([lane(i) for i in range(9)], execute, max_active=6)
    assert got == [[(i,t) for t in range(i%3+1)] for i in range(9)]
    assert max(map(len,seen)) == 4


def test_failure_propagates_without_replay():
    def fail(_):
        raise RuntimeError("already wrote KV")
    jobs = [lambda issue: issue("decoder",8,0)]*7
    with pytest.raises(RuntimeError,match="already wrote KV"):
        waves(jobs,fail,max_active=4)


def test_native_encoder_is_separate():
    ops = [Call(i,p,256,None) for i,p in enumerate(["encoder","decoder","encoder","decoder"])]
    assert sorted(len(g) for g in pack(ops)) == [1,1,2]
