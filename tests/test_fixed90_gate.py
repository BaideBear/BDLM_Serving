from copy import deepcopy
from hashlib import sha256
import json
import pytest
from experiments.t2t.serving_slots import fixed90_gate as g


def fixture(model="llada21", q=16):
    samples=json.loads(g.MANIFEST.read_bytes())["samples"]
    fingerprint=g.LOCK["llada21",16]
    data=dict(model=model,q=q,family=g.FAMILY,manifest_sha256=g.MANIFEST_SHA,complete=True,verified=90,
              native_prompts_verified=True,cohort="frozen90",missing=[],sealed_failures=[],
              cases={},seal_digests={},counts=dict(F=900,Q=1800),N=900,correct=89,TE=.5,TPF=1.)
    for k,v in zip(("executor_sha256","runner_sources_sha256","checkpoint","scorer"),fingerprint):
        data[k]=v
    for s in samples:
        i=s["index"]
        data["cases"][str(i)]=dict(task_id=s["task_id"],seed=s["seed"],correct=i!=88,
               terminal="native_failure" if i==88 else "eos",N=10,F=10,Q=20,decoder_F=9,encoder_F=1,
               over=0,peak=3,build_Q=0)
        data["seal_digests"][str(i)]=sha256(f"{model}:{q}:{i}".encode()).hexdigest()
    return data


def test_no_cohort_or_comparison_imputation():
    result=g.evaluate([])
    assert result["ready_tiers"] == 0 and len(result["tiers"]) == 10 and result["meets_performance"] is None
    result=g.evaluate([fixture()])
    assert result["ready_tiers"] == 1 and result["status"] == "incomplete"
    cell=next(x for x in result["tiers"] if x["status"] == "ready_to_compare")
    assert cell["cohorts"]["new60"]["terminals"]["native_failure"] == 1
    assert cell["comparisons"]["frozen90"]["IBS capped"]["correct"] == 17
    assert cell["cohorts"]["frozen90"]["mean_decoder_Q"] == pytest.approx(20/9)
    assert cell["comparisons"]["frozen90"]["IBS capped"]["extra_history_Q"] == 1800-206927
    assert cell["meets_performance"] is False
    assert g.evaluate([fixture()],refs=[])["ready_tiers"] == 0
    with pytest.raises(ValueError,match="duplicate"):
        g.evaluate([fixture(),fixture()])


@pytest.mark.parametrize("bad",["seed","task","subset","runner","fork","strict","false_success","bill","prefill"])
def test_reject_wrong_identity_or_work(bad):
    d=fixture()
    c=d["cases"]["0"]
    if bad=="seed": c["seed"]+=1
    if bad=="task": c["task_id"]="replacement"
    if bad=="subset": del d["cases"]["89"]
    if bad=="runner": d["runner_sources_sha256"]="0"*64
    if bad=="fork": d["cohort"]="original30"
    if bad=="strict": d["family"]="strict"
    if bad=="false_success": c["terminal"]="native_failure"
    if bad=="bill": c["F"]-=1
    if bad=="prefill": d["native_prompts_verified"]=False
    with pytest.raises(ValueError):
        g.evaluate([d])


def test_ten_complete_offers_can_still_lose_and_do_not_replace_missing_te():
    docs=[fixture(m,q) for m,tiers in g.MODELS.items() for q in tiers]
    locks={(d["model"],d["q"]):g.LOCK["llada21",16] for d in docs}
    refs=deepcopy(g.collect())
    for r in refs:
        if r["method"]=="Flow W1" and r["model"]=="gemma":
            r["token_efficiency"]=None
    result=g.evaluate(docs,lock=locks,refs=refs)
    assert result["ready_tiers"] == 10 and result["status"] == "ready_to_compare"
    assert not result["meets_performance"] and not result["three_metrics_complete"]


def test_peak_evidence_is_not_assumed_from_nominal_budget():
    doc = fixture()
    refs = deepcopy(g.collect())
    for ref in refs:
        ref["peak_decoder_Q"] = None
    def compared():
        cell = next(c for c in g.evaluate([doc], refs=refs)["tiers"] if "comparisons" in c)
        return cell["comparisons"]["frozen90"]["IBS capped"]["both_observed_within_offer"]
    assert compared() is None
    doc["cases"]["0"].update(peak=17, over=1, build_Q=17)
    assert compared() is False
