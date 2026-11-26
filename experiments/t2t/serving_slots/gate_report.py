"""Paired, per-cohort Gemma gate report; only complete request sets enter."""
import argparse
from collections import Counter
from functools import lru_cache
from math import ceil
from pathlib import Path

from bdlm_serving.evaluation.frozen import sha
from experiments.t2t.prefix_window.score_run import link_requests, snapshot
from experiments.t2t.serving_slots.external_report import read, write
from experiments.t2t.serving_slots.gated import PLANS, THRESHOLD, original, validate_gate
from experiments.t2t.serving_slots.switch_report import first_changed_offer
from tools.prefix_cohorts import load_cases, load_request_set, select_window_cases, _window_row


def report(data, run, out, policies):
    samples=data/"samples.json"
    digest, cases=sha(samples),load_cases(samples)
    meta=read(run/"run.json")
    matrix=[f"gate_{mode}_{phase}" for mode in ("all","tail","early") for phase in PLANS]
    if (meta["model"],meta["block"],meta["batch_size"],meta["samples_sha256"],meta["suite"],
        meta["threshold"],meta["policies"],meta["offers"])!=(
        "gemma",256,1,digest,"frozen90",THRESHOLD,matrix,{p:list(t) for p,t in PLANS.items()}):
        raise ValueError("Different run contract")
    if policies is None:
        policies=matrix
        if read(run/"replay-qualified.json")!=dict(samples=digest,matched_per_phase=90,phases=list(PLANS)):
            raise ValueError("Missing complete replay qualification")
    if not policies or len(set(policies))!=len(policies) or not set(policies)<=set(matrix):
        raise ValueError("Unregistered or duplicate gate")
    for p in set(policies)|{"gate_all_high","gate_all_low"}:
        if read(run/f"complete-{p}.json")!=dict(policy=p,requests=90):
            raise ValueError(f"Incomplete {p}")
    out.mkdir(parents=True,exist_ok=True)
    cache=out/".inputs"
    @lru_cache(maxsize=None)
    def scored(folder,label):
        rows,model,policy,_=load_request_set(folder,cases,"Gemma mask gate",label)
        if (model,policy)!=("gemma",label):
            raise ValueError("Wrong scored policy")
        return {r["index"]:r for r in rows}
    summary,pairs=[],[]
    for p in policies:
        _,mode,phase=p.split("_")
        orig=original(phase)
        refpath=snapshot(data/"gemma-slots-native256-128",orig,digest,cache)
        folder=link_requests(run,p,cache)
        measurements=scored(folder,p)
        controls={orig:refpath}
        for label in ("slottrace_128","slottrace_256n_128_128_128","slottrace_128_256n_128_128"):
            controls[label]=snapshot(data/"gemma-high-thinning",label,digest,cache)
        timing=read(run/f"timing-{p}.json")
        if timing["policy"]!=p or [v["index"] for v in timing["requests"]]!=list(range(90)):
            raise ValueError("Incomplete clock")
        counts=[]
        for i in range(90):
            r=read(folder/f"{i:02d}-{p}.json")
            ref=read(refpath/f"{i:02d}-{orig}.json")
            decisions=r["gate_decisions"]
            work,stages=validate_gate(r,decisions,phase,mode)
            first=first_changed_offer(ref,r)
            if first!=r["first_changed_event"] or (mode=="all" and first is not None):
                raise ValueError("First intervention does not align")
            if mode=="all" and {k:r[k] for k in ref if k!="policy"}!={k:ref[k] for k in ref if k!="policy"}:
                raise ValueError("Ungated callback did not reproduce the qualified execution")
            offered=sum(d[-2]==256 for d in decisions)
            accepted=sum(d[-1]==256 for d in decisions)
            if accepted!=stages["native_offer_F"]:
                raise ValueError("Accepted native offers and physical events differ")
            counts.append(Counter(offered_high=offered,used_high=accepted,declined_high=offered-accepted,
                                  changed_requests=int(first is not None),physical_Q=work["physical_Q"],
                                  F=work["F"],N=work["N"],overflow_rows=work["overflow_rows"],
                                  native_closes=stages["native_closes"],verify_F=stages["verify_F"],
                                  selected_F=stages["selected_F"],strict_T2T=stages["strict_T2T"]))
        for w in (30,60,90):
            cohort=select_window_cases(cases,w)
            idx=[x["index"] for x in cohort]
            total=Counter()
            for i in idx:
                total.update(counts[i])
            for label,root in controls.items():
                pair=_window_row(cohort,scored(root,label),measurements,w,"all","gemma",label,p,None)
                pairs.append(pair)
                if label!=orig:
                    continue
                if any(total[k]!=pair["arm_"+k] for k in ("N","F","physical_Q")):
                    raise ValueError("Paired scoring and physical ledgers disagree")
                walls=sorted(timing["requests"][i]["wall_ns"]/1e6 for i in idx)
                if walls[0]<=0:
                    raise ValueError("Invalid service interval")
                summary.append(dict(model="gemma",policy=p,cohort=pair["cohort"],requests=w,
                    correct=pair["arm_correct"],macro_tpf=pair["arm_macro_tpf"],
                    pooled_token_efficiency=pair["arm_pooled_token_efficiency"],**total,
                    rescued=pair["rescued"],lost=pair["lost"],wall_mean_ms=sum(walls)/w,
                    wall_p95_ms=walls[ceil(.95*w)-1]))
    singular="" if policies==matrix else "-"+"+".join(policies)
    write(out/("gemma-maskgate"+singular+".csv"),summary)
    write(out/("gemma-maskgate-pairs"+singular+".csv"),pairs)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for field in ("data","run","out"):
        p.add_argument("--"+field,type=Path,required=True)
    p.add_argument("--policy",dest="policies",action="append",help="Report only complete policies")
    report(**vars(p.parse_args()))
