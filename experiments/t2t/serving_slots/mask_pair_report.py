"""Queue or score a registered cross-block mask ablation without filtering cases."""
import argparse
import csv
import json
import os
from pathlib import Path
from random import Random
import subprocess
import sys
import time

from experiments.t2t.serving_slots.expanded_handoff import SAMPLES


HERE = Path(__file__).resolve().parent
CONFIG = {
    "gemma": ("prefix90-wave-switch", "gemma-fixed512-mask-pair", "active_w2_mask_on",
              "active_w2_mask_off", "active_w2", 512, "0"),
    "llada21": ("prefix90-llada-cap4", "mask-pair-llada21-v1", "dual64_noedit",
                "dual64_noedit_bidir", "dual64_noedit", 64, "4"),
}


def read(path):
    return json.loads(path.read_text())


def complete(folder, policy):
    if read(folder/f"complete-{policy}.json") != dict(policy=policy,requests=90):
        raise ValueError(f"Unqualified arm: {folder}/{policy}")


def eligible(e, prompt_rows, block=32):
    if e["phase"]=="prefill":
        return False
    end=e["visible_kv"]
    if end % block:
        raise ValueError("Unexpected LLaDA visible width")
    return any(max(prompt_rows,end-2*block)<=p<end-block for p in e["absolute_positions"])


def first_eligible(events, prompt_rows, block=32):
    return next((i for i,e in enumerate(events) if eligible(e,prompt_rows,block)),None)


def reference_opportunities(data, model, out):
    """Descriptive reference geometry, not an intervention or a new quality score."""
    from bdlm_serving.evaluation.frozen90 import load_tasks
    cases=load_tasks(data/"samples.json")
    folder,p=(("gemma-wave-reference90","flow_w2") if model=="gemma" else
              ("llada21-rolling-dual64","dual64_noedit"))
    src=data/folder
    complete(src,p)
    if read(src/"run.json")["samples_sha256"]!=SAMPLES:
        raise ValueError("Unexpected reference tasks")
    counts=[]
    for task in cases:
        raw=read(src/f"{task['index']:02d}-{p}.json")
        if raw["policy"]!=p or any(raw[k]!=task[k] for k in ("index","task_id","seed","dataset")):
            raise ValueError("Reference case mismatch")
        if model=="gemma":
            steps=raw["trace"]
            observed=[q for phase,q in raw["events"][1:] if phase=="decoder"]
            if ([s["q"] for s in steps]!=observed or any(
                s["width"]!=2 or s["active"] not in (1,2) or s["q"]!=512 for s in steps)):
                raise ValueError("Not the packed W2 reference")
            opportunity=[s["active"]==2 for s in steps]
        else:
            prompt=read(data.parent/"frozen90-native"/"llada21"/f"{task['index']:02d}-native32.json")
            if any(task[k]!=prompt[k] for k in ("index","task_id","seed","dataset")):
                raise ValueError("Wrong prompt reference")
            steps=[s for s in raw["events"] if s["phase"]!="prefill"]
            if len(steps)!=raw["generation_forwards"]:
                raise ValueError("Wrong LLaDA forward ledger")
            opportunity=[eligible(s,len(prompt["prompt_ids"])) for s in steps]
        counts.append((task["index"],task["dataset"],len(opportunity),sum(opportunity)))
    rows=[]
    for cohort,lo,hi in (("original30",0,30),("added60",30,90),("all90",0,90)):
        for ds in ("all","gsm8k","math500","humaneval"):
            subset=[x for x in counts if lo<=x[0]<hi and ds in ("all",x[1])]
            rows.append(dict(model=model,reference=p,cohort=cohort,dataset=ds,
                requests=len(subset),requests_with_opportunity=sum(x[3]>0 for x in subset),
                eligible_steps=sum(x[3] for x in subset),steps=sum(x[2] for x in subset),
                denominator="decoder" if model=="gemma" else "post_prefill_model_forwards"))
    with out.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def check_llada_prefix(on, off, prompt_rows):
    """Reconstruct the first eligible forward from either arm's actual Q and KV.

    The pre-trigger event ledger is post-sampling, so all fields must match
    before the trigger, but only input/work fields should match at the trigger.
    This audits observed events, not a bitwise hash of latent KV/RNG state.
    """
    a,b=on["events"],off["events"]
    i=first_eligible(a,prompt_rows)
    rec=off["mask_pair"]
    if i!=first_eligible(b,prompt_rows) or (i is not None)!=(rec["opened_forward_count"]>0):
        raise ValueError("Mask hook disagrees with first eligible forward in independent ledgers")
    if i is None:
        if a!=b:
            raise ValueError("Unexposed request changed its event ledger")
        return False
    if i!=rec["first_trigger"]["event_index"] or a[:i]!=b[:i]:
        raise ValueError("Divergence before first intervention or trigger indexing mismatch")
    # Only compare input geometry here, not post-step edits or sampling outcomes.
    for k in ("phase","stage","physical_q","actual_q","absolute_positions","layer_physical_q","visible_kv"):
        if a[i][k]!=b[i][k]:
            raise ValueError(f"Different physical inputs at first intervention: {k}")
    return True


def efficiency_interval(cases, off, on, draws=4000, seed=17000):
    """Paired percentile interval for the difference of two pooled ratios."""
    strata={}
    for c in cases:
        a,b=off[c["index"]],on[c["index"]]
        if a["physical_Q"]<=0 or b["physical_Q"]<=0:
            raise ValueError("Nonpositive physical Q")
        strata.setdefault(c["dataset"],[]).append((a["N"],a["physical_Q"],b["N"],b["physical_Q"]))
    if not strata or draws<40:
        raise ValueError("Empty paired scope or too few bootstrap draws")
    rng=Random(seed)
    deltas=[]
    for _ in range(draws):
        n0=q0=n1=q1=0
        for ds in sorted(strata):
            rows=strata[ds]
            for a,b,c,d in rng.choices(rows,k=len(rows)):
                n0+=a; q0+=b; n1+=c; q1+=d
        deltas.append(n1/q1-n0/q0)
    deltas.sort()
    return deltas[int(.025*draws)],deltas[int(.975*draws)-1]


def queue(root, model):
    parent,name,_,_,_,_,gpu = CONFIG[model]
    data=root/"prefix90"
    while True:
        check=subprocess.run(["tmux","has-session","-t","="+parent],capture_output=True)
        if check.returncode==1:
            break
        check.check_returncode()
        time.sleep(8)
    env=dict(os.environ,FROZEN90_SAMPLES=str(data/"samples.json"),TRACE_OUT=str(data/name))
    if model=="llada21":
        from experiments.t2t.serving_slots.llada_capped import policies
        for group in ("anchor","endpoints","low","switch"):
            folder=data/f"llada21-optimus-hardcap-{group}"
            if read(folder/"run.json")["samples_sha256"]!=SAMPLES:
                raise ValueError("Unexpected GPU4 predecessor dataset")
            for p in policies(group):
                complete(folder,p)
        env.update(PREFIX_MODE="llrefs",MASK_PAIR_REFERENCE=str(root/"frozen90-native"/"llada21"),
                   LL_BASELINE_POLICIES="dual64_noedit",MASK_PAIR_ARMS="both")
    runner=HERE/f"mask_pair_{'llada' if model=='llada21' else 'gemma'}.py"
    assert (root/"launch_prefix90.py").is_file() and not (data/name).exists()
    # Each entry uses the same preloaded/UUID-checked release and restore.
    with (data/(name+".log")).open("x") as log:
        subprocess.run([sys.executable,"-u",runner,model,gpu],cwd=HERE.parents[2],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def score(data, model, out):
    from bdlm_serving.evaluation.frozen import sha
    from tools.prefix_cohorts import load_cases, normalize_request, select_window_cases, _window_row
    _,name,on,off,canonical,q,_ = CONFIG[model]
    folder=data/name
    cases=load_cases(data/"samples.json")
    meta=read(folder/"run.json")
    if (meta["samples_sha256"]!=SAMPLES or sha(data/"samples.json")!=SAMPLES
            or meta["suite"]!="frozen90"):
        raise ValueError("Dataset identity mismatch")
    # Reuse the original executor's work audit, not its label on the published rows.
    tables, exposure={},{}
    for role,p in (("on",on),("off",off)):
        path=folder/role if model=="llada21" else folder
        complete(path,p)
        table={}
        for c in cases:
            i=c["index"]
            raw=read(path/f"{i:02d}-{p}.json")
            if raw["policy"]!=p:
                raise ValueError("Wrong recorded mask arm")
            if model=="llada21" and role=="off":
                peer=read(folder/"on"/f"{i:02d}-{on}.json")
                prompt=read(Path(meta["prompt_reference"])/f"{i:02d}-native32.json")["prompt_ids"]
                exposure[i]=check_llada_prefix(peer,raw,len(prompt))
            row=normalize_request(dict(raw,policy=canonical),c,role)
            if row["model"]!=model:
                raise ValueError("Wrong model")
            table[i]=dict(row,policy=p)
        tables[role]=table
    if model=="gemma":
        audit=read(folder/"pair-audit.json")["cases"]
        if (read(folder/"qualified.json")["historical_replayed"]!=90 or
                [a["index"] for a in audit]!=list(range(90)) or not all(a["masked_replay_exact"] for a in audit)):
            raise ValueError("Incomplete paired replay")
        exposure={a["index"]:a["first_intervention_step"] is not None for a in audit}
    if len(exposure)!=90 or not any(exposure.values()):
        raise ValueError("No qualified mask intervention on the full frozen90 cohort")
    output=[]
    for w in (30,60,90):
        subset=select_window_cases(cases,w)
        for scope in ("all","gsm8k","humaneval","math500"):
            r=_window_row(subset,tables["off"],tables["on"],w,scope,model,off,on,q)
            chosen=[c for c in subset if scope in ("all",c["dataset"])]
            r["requests_exposed"]=sum(exposure[c["index"]] for c in chosen)
            for label in ("baseline","arm"):
                r[label+"_pooled_token_efficiency"]=r[label+"_N"]/r[label+"_physical_Q"]
            r["pooled_token_efficiency_delta"]=r["arm_pooled_token_efficiency"]-r["baseline_pooled_token_efficiency"]
            r["efficiency_ci_low"],r["efficiency_ci_high"]=efficiency_interval(chosen,tables["off"],tables["on"])
            r["efficiency_ci_method"]="stratified paired percentile bootstrap; 4000 draws, seed 17000; denominator physical_Q"
            output.append(r)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(output[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("queue","score","opportunities"))
    p.add_argument("model",choices=CONFIG)
    p.add_argument("runtime",type=Path)
    args=p.parse_args()
    root=args.runtime.resolve()
    if args.action=="opportunities":
        reference_opportunities(root/"prefix90",args.model,root/"prefix90"/f"{args.model}-mask-opportunities.csv")
    else:
        if args.action=="queue":
            queue(root,args.model)
        score(root/"prefix90",args.model,root/"prefix90"/CONFIG[args.model][1]/"mask-on-vs-off.csv")
