"""Queue or score a registered cross-block mask ablation without filtering cases."""
import argparse
import csv
import json
import os
from pathlib import Path
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
                exposure[i]=raw["mask_pair"]["opened_forward_count"]>0
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
    output=[]
    for w in (30,60,90):
        subset=select_window_cases(cases,w)
        for scope in ("all","gsm8k","humaneval","math500"):
            r=_window_row(subset,tables["off"],tables["on"],w,scope,model,off,on,q)
            r["requests_exposed"]=sum(exposure[c["index"]] for c in subset if scope in ("all",c["dataset"]))
            output.append(r)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(output[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("queue","score"))
    p.add_argument("model",choices=CONFIG)
    p.add_argument("runtime",type=Path)
    args=p.parse_args()
    root=args.runtime.resolve()
    if args.action=="queue":
        queue(root,args.model)
    score(root/"prefix90",args.model,root/"prefix90"/CONFIG[args.model][1]/"mask-on-vs-off.csv")
