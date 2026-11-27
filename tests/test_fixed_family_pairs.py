import pytest

from experiments.t2t.serving_slots.fixed_family_pairs import pair


def examples():
    c=[dict(index=i,dataset=str(i%3),seed=90+i,task_id=f"x/{i}",N=20+i,F=4+i,
            Q=32+i,correct=i%3==0) for i in range(9)]
    b=[dict(x) for x in c]
    b[0].update(correct=False,N=10)
    b[1].update(correct=True,Q=40)
    b[2].update(correct=True,F=2)
    return c,b


def test_good_pairs():
    a,b=examples()
    r=pair(a,b)
    assert (r["requests"],r["rescued"],r["lost"])==(9,2,1)
    assert r["delta_quality"]==pytest.approx(1/9)
    assert r["delta_TPF"]==pytest.approx(sum(y["N"]/y["F"]-x["N"]/x["F"] for x,y in zip(a,b))/9)
    assert r["delta_efficiency"]==pytest.approx(sum(x["N"] for x in b)/sum(x["Q"] for x in b)-
                                                sum(x["N"] for x in a)/sum(x["Q"] for x in a))
    assert r["quality_ci_low"]<r["delta_quality"]<r["quality_ci_high"]


@pytest.mark.parametrize("field",["index","seed","dataset","task_id"])
def test_wrong_identity(field):
    a,b=examples()
    b[0][field]="wrong"
    with pytest.raises((RuntimeError,ValueError,AssertionError)):
        pair(a,b)
