"""Compare fixed-offer LLaDA endpoints to completed original FlowBlock or Optimus runs."""
import argparse
from pathlib import Path

from bdlm_serving.evaluation.frozen import sha
from experiments.t2t.prefix_window.score_run import link_requests
from experiments.t2t.serving_slots.llada_reference import read, verify_row
from experiments.t2t.serving_slots.expanded_handoff import SAMPLES
from tools.prefix_cohorts import score_files


def endpoints(w, optimus, mode):
    if w in (1,2) and optimus is None and mode is None:
        endpoint, arm = (("llada21-32-matched","fullwidth") if w==1 else
                         ("llada21-rolling-dual64","dual64_noedit"))
        return "llada21-reference90-wave", f"flow_w{w}", endpoint, arm, w*32
    if w is None and optimus in (4,8,16,32) and mode in ("ibs","obs"):
        endpoint, arm = (("llada21-slots-8-16",f"slottrace_{optimus}") if optimus in (8,16) else
                         (f"llada21-{optimus}-matched",f"t2t_q{optimus}_f0_m{optimus}_hard_prefix_zero"))
        return f"llada21-reference90-q{optimus}", f"optimus_{mode}_c{optimus}", endpoint, arm, optimus
    raise ValueError("Select exactly one of --w or --optimus with --mode")


def report(data, original, w, out, optimus=None, mode=None):
    cases=data/"samples.json"
    reference, policy, endpoint, arm, nominal=endpoints(w,optimus,mode)
    wave=data/reference
    digest=sha(cases)
    if digest!=SAMPLES:
        raise ValueError("Unexpected frozen90 sample digest")
    for folder in (wave,data/endpoint):
        meta=read(folder/"run.json")
        if (meta["samples_sha256"],meta["model"],meta["suite"])!=(digest,"llada21","frozen90"):
            raise ValueError("Reference suite mismatch")
    if read(wave/f"complete-{policy}.json")!=dict(policy=policy,requests=90):
        raise ValueError("Incomplete original reference arm")
    if optimus is not None:
        if read(wave/"run.json").get("hard_q_cap") is not False:
            raise ValueError("Expected the original, uncapped Optimus reference")
        if read(data/endpoint/"qualified.json")!=dict(model="llada21",full_width=32,matched=90):
            raise ValueError("Sparse endpoint is missing its full-width qualification")
    keys=[f"{i:02d}-{policy}.json" for i in range(30)]
    if read(wave/f"reproduced30-{policy}.json")!=dict(policy=policy,matched=30,
                                                     reference_sha256=[sha(original/k) for k in keys]):
        raise ValueError("Original30 evidence changed")
    for k in keys:
        verify_row(read(original/k),read(wave/k))
    proof=(("qualified.json", dict(model="llada21",full_width=32,matched=90)) if w==1 else
           (f"complete-{arm}.json",dict(policy=arm,requests=90)))
    if read(data/endpoint/proof[0])!=proof[1]:
        raise ValueError("Endpoint not qualified")
    cache=out.parent/".inputs"
    base=link_requests(wave,policy,cache)
    test=link_requests(data/endpoint,arm,cache)
    # Different runtimes may not expose a nominal_q field; compare their physical ledgers.
    rows,_,_=score_files(base,test,cases,out,baseline_policy=policy,arm_policy=arm,nominal_q=nominal)
    for r in rows:
        if r["scope"]=="all":
            print(r["cohort"],*(r[k] for k in ("baseline_correct","arm_correct","baseline_macro_tpf",
                  "arm_macro_tpf","baseline_pooled_token_efficiency","arm_pooled_token_efficiency",
                  "baseline_physical_Q","arm_physical_Q","rescued","lost")))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("data","original","out"):
        p.add_argument("--"+k,type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--w",type=int,choices=(1,2))
    group.add_argument("--optimus",type=int,choices=(4,8,16,32))
    p.add_argument("--mode",choices=("ibs","obs"))
    report(**vars(p.parse_args()))
