from tools.audit_agreement_dose import selection, tally
import pytest
from tools.agreement_fixed90_comparison import cohorts, dataset_cost, decompose, paired_intervals


def step(tick, block, votes, probs, native, ids):
    return dict(tick=tick, block=block, extra=votes, positions=list(range(len(ids))),
                probabilities=probs, native=native, proposals=ids)


def test_matched_dose_does_not_buy_special_or_native_rows():
    s = step(0, 0, [3], [.9, .8, .7, .55, .9], [0, 0, 0, 0, 1], [99, 1, 2, 3, 4])
    assert selection(s, {99})[1:] == ({3}, {1})
    with pytest.raises(ValueError):
        selection(dict(s, extra=[3, 3]), {99})


def test_future_labels_count_only_the_next_native_admission_in_the_same_block():
    steps = [step(0, 0, [0], [.6, .8], [0, 0], [3, 4]),
             step(1, 0, [], [.7, .9], [0, 1], [3, 5]),
             step(2, 1, [], [.9, .8], [1, 1], [3, 4])]
    c = tally(steps, set(), full_tape=True)
    assert (c["dose"], c["replaced_rows"], c["repeat_only_unobserved"],
            c["current_only_different"]) == (1, 1, 1, 1)
    assert not any(k.endswith("unobserved") for k in tally(steps, set(), full_tape=False))


def test_ties_break_by_absolute_position_without_separating_same_sets():
    s = step(0, 0, [0, 1], [.8]*3, [0]*3, [3, 4, 5])
    assert tally([s], set(), full_tape=False)["different_set_steps"] == 0


def test_paired_tpf_breakdown_and_failed_requests_are_retained():
    a = dict(N=40, F=8, Q=100, correct=True)
    b = dict(N=30, F=4, Q=90, correct=False)
    n, f = decompose(a, b)
    assert n < 0 < f and n+f == 2.5
    pairs = {i: dict(stock=a, apply=a) for i in range(90)}
    pairs[55] = dict(stock=a, apply=b)
    s = cohorts(pairs)
    assert s["frozen90"]["lost"] == [55] and s["frozen90"]["F_less"] == 1
    assert s["original30"]["lost"] == [] and s["outside84"]["requests"] == 84
    with pytest.raises(ValueError):
        cohorts(dict(list(pairs.items())[:-1]))


def test_paired_intervals_keep_the_two_arms_together_and_preserve_dataset_size():
    ds=[{"index":i,"dataset":str(i%3)} for i in range(90)]
    rows={}
    for i in range(90):
        a=dict(N=i+7,F=3+i%5,Q=33+i,correct=bool(i%2))
        rows[i]=dict(stock=a,apply=a.copy())
    assert all(v==[0.,0.] for v in paired_intervals(rows,ds,count=101)["ci95"].values())
    ds[-1]["dataset"]=ds[-2]["dataset"]
    with pytest.raises(ValueError):
        paired_intervals(rows,ds,count=101)


def test_cost_breakdown_keeps_the_full_long_failed_generation():
    ds = [{"index":i,"dataset":"code"} for i in range(3)]
    ok = dict(F=10,Q=50,N=42,correct=True)
    fail = dict(F=300,Q=600,N=480,correct=False)
    rows = {i:dict(stock=ok,apply=(fail if i==2 else ok)) for i in range(3)}
    out = dataset_cost(rows,ds)
    assert out["datasets"]["code"] == dict(requests=3, stock_correct=3, apply_correct=2,
                                           delta_F=290, delta_N=438, delta_Q=550)
    assert out["greatest_extra_F"][0]["index"] == 2
