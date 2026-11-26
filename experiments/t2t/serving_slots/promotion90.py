"""Validate verified KV promotion with real GPU calls and paired frozen90 outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.t2t.serving_slots.expanded_handoff import SAMPLES, SHA


def read(p):
    return json.loads(p.read_text())


def peers(data,q):
    if q in (8,16):
        return data/"llada21-slots-8-16", f"slottrace_{q}"
    if q in (4,32):
        return data/f"llada21-{q}-matched",f"t2t_q{q}_f0_m{q}_hard_prefix_zero"
    raise ValueError("Unregistered single-block slot")


def check_pair(ref,row):
    fields=("phase","stage","absolute_positions","physical_q","layer_physical_q","accepted",
            "edited","masked_before","masked_after","hard_strict_edits","context_version","block")
    commits=row["logical_commits"]
    if not commits or [c["block"] for c in commits]!=list(range(commits[0]["block"],commits[0]["block"]+len(commits))):
        raise ValueError("Discontinuous logical commit history")
    seen=set()
    for c in commits:
        idx=c["source_forward"]
        if type(idx)!=int or idx in seen or not 0<=idx<len(row["events"]):
            raise ValueError("Invalid source forward")
        seen.add(idx)
        ev=row["events"][idx]
        if (ev["block"]!=c["block"] or ev["context_version"]!=c["context_version"] or
                ev["physical_q"]!=32 or ev["absolute_positions"]!=list(range(c["block"]*32,(1+c["block"])*32))):
            raise ValueError("Invalid source block")
        if c["promoted"]:
            if ev["stage"]!="verify" or ev["edited"] or ev["masked_before"] or ev["masked_after"]:
                raise ValueError("Cannot promote changed or incomplete inputs")
        elif ev["stage"]!="commit":
            raise ValueError("Missing fallback commit")
    promoted={c["block"] for c in commits if c["promoted"]}
    recorded={e["block"] for e in row["events"] if e["stage"]=="commit"}
    if recorded!={c["block"] for c in commits if not c["promoted"]}:
        raise ValueError("Logical and physical commits disagree")
    old=[e for e in ref["events"] if not (e["stage"]=="commit" and e["block"] in promoted)]
    exact_path=(len(old)==len(row["events"]) and all(
        all(l.get(k)==r.get(k) for k in fields) for l,r in zip(old,row["events"])))
    if exact_path and ((ref["generation_forwards"]-row["generation_forwards"])!=len(promoted) or
            ref["generation_physical_padded_rows"]-row["generation_physical_padded_rows"]!=32*len(promoted)):
        raise ValueError("Claimed equal path disagrees with physical work")
    return dict(exact_path=exact_path,exact_output=all(ref[k]==row[k] for k in
                ("output_ids","output_tokens","output_text","eos","truncated","t2t_edits")),
                promoted=len(promoted),recomputed=len(commits)-len(promoted))


def run(args,prepare=None):
    from bdlm_serving.evaluation.frozen import sha
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.methods.t2t.online import llada_backend
    from bdlm_serving.methods.t2t.protocol import Arm
    from experiments.t2t.prefix_window.score_run import link_requests
    from tools.prefix_cohorts import score_files
    from tools.run_t2t import durable
    data=args.samples.parent
    tasks=load_tasks(args.samples)
    if sha(args.samples)!=SAMPLES or len(tasks)!=90 or args.output.exists():
        raise ValueError("Expected registered fresh frozen90 output")
    cfg=argparse.Namespace(checkpoint=args.checkpoint,reference=data.parent/"frozen90-native"/"llada21")
    tiers=(8,16,4,32)
    for q in tiers:
        folder,policy=peers(data,q)
        if (read(folder/"qualified.json")!=dict(model="llada21",full_width=32,matched=90)
                or read(folder/"complete-{}.json".format(policy))!=dict(policy=policy,requests=90)
                or read(folder/"run.json")["samples_sha256"]!=SAMPLES):
            raise ValueError("No qualified fixed endpoint")
    args.output.mkdir()
    run_one=llada_backend(cfg,prepare,promote_verified=True)
    durable(args.output/"run.json",dict(model="llada21",suite="frozen90",samples_sha256=SAMPLES,tiers=tiers,
            prefix_policy="zero",gen_length=2048,original30=True,exact_output_required_for_parity=True))
    for q in tiers:
        folder,orig=peers(data,q)
        label=f"promote_q{q}_zero"
        # Keep the registered selection path (including offer metadata).
        arm=Arm(q,0,q,orig if orig.startswith("slottrace_") else "hard_prefix_zero")
        audits=[]
        for task in tasks:
            i=task["index"]
            ref=read(folder/f"{i:02d}-{orig}.json")
            if ref["policy"]!=orig or any(ref[k]!=task[k] for k in ("index","seed","dataset","task_id")):
                raise ValueError("Different baseline task")
            row=run_one(task,arm)
            row.update({k:task[k] for k in ("index","seed","dataset","task_id")},policy=label,model="llada21")
            audit=check_pair(ref,row)
            audits.append(dict(index=i,**audit))
            durable(args.output/f"{i:02d}-{label}.json",row)
        durable(args.output/f"complete-{label}.json",dict(policy=label,requests=90))
        durable(args.output/f"parity-{label}.json",dict(cases=audits,exact_all=all(
            a["exact_output"] and a["exact_path"] for a in audits)))
        cache=args.output/".score-inputs"
        score_files(link_requests(folder,orig,cache),link_requests(args.output,label,cache),args.samples,
                    args.output/f"{label}.csv",baseline_policy=orig,arm_policy=label,nominal_q=q)


def queue(root):
    data=root/"prefix90"
    while True:
        check=subprocess.run(["tmux","has-session","-t","=prefix90-mask-pair-llada"],capture_output=True)
        if check.returncode==1:
            break
        check.check_returncode()
        time.sleep(8)
    if not (data/"mask-pair-llada21-v1"/"mask-on-vs-off.csv").is_file():
        raise ValueError("GPU4 mask-pair predecessor did not finish scoring")
    out=data/"llada21-verified-commit90"
    env=dict(os.environ,TRACE_OUT=str(out),FROZEN90_SAMPLES=str(data/"samples.json"),
             PREFIX_MODE="llrefs",LL_BASELINE_POLICIES="promote")
    with out.with_suffix(".log").open("x") as log:
        subprocess.run([sys.executable,"-u",__file__,"llada21","4"],env=env,cwd=root,
                       stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=="__main__":
    if len(sys.argv)==3 and sys.argv[1]=="queue":
        queue(Path(sys.argv[2]).resolve())
    elif sys.argv[1:] in (["llada21","4"],["llada21","4","worker"]):
        launcher=Path(os.environ["FROZEN90_SAMPLES"]).resolve().parents[1]/"launch_prefix90.py"
        src=launcher.read_bytes()
        key=b"from experiments.t2t.serving_slots.llada_reference import run"
        if hashlib.sha256(src).hexdigest()!=SHA or src.count(key)!=1 or os.environ["PREFIX_MODE"]!="llrefs":
            raise RuntimeError("Unreviewed GPU4 launcher or mode")
        exec(compile(src.replace(key,b"from experiments.t2t.serving_slots.promotion90 import run"),
                     str(launcher),"exec"),globals())
    else:
        raise SystemExit("Use queue <runtime> or the protected GPU4 launcher")
