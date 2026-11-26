"""Score one completed 90-task FlowBlock tier against a true equal-capacity endpoint."""
import argparse
from pathlib import Path

from bdlm_serving.evaluation.frozen import sha
from experiments.t2t.prefix_window.score_run import link_requests
from experiments.t2t.serving_slots.llada_reference import read, verify_row
from tools.prefix_cohorts import score_files


def report(data, original, w, out):
    cases=data/"samples.json"
    wave=data/"llada21-reference90-wave"
    policy=f"flow_w{w}"
    endpoint, arm = (("llada21-32-matched","fullwidth") if w==1 else
                     ("llada21-rolling-dual64","dual64_noedit"))
    digest=sha(cases)
    if w not in (1,2):
        raise ValueError("Only W1/W2")
    for folder in (wave,data/endpoint):
        meta=read(folder/"run.json")
        if (meta["samples_sha256"],meta["model"],meta["suite"])!=(digest,"llada21","frozen90"):
            raise ValueError("Reference suite mismatch")
    if read(wave/f"complete-{policy}.json")!=dict(policy=policy,requests=90):
        raise ValueError("Incomplete wave arm")
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
    rows,_,_=score_files(base,test,cases,out,baseline_policy=policy,arm_policy=arm)
    for r in rows:
        if r["scope"]=="all":
            print(r["cohort"],*(r[k] for k in ("baseline_correct","arm_correct","baseline_macro_tpf",
                  "arm_macro_tpf","baseline_pooled_token_efficiency","arm_pooled_token_efficiency",
                  "baseline_physical_Q","arm_physical_Q","rescued","lost")))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("data","original","out"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--w",type=int,required=True,choices=(1,2))
    report(**vars(p.parse_args()))
