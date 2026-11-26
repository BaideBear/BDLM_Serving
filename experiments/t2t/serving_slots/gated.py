"""Spend externally offered Gemma native slots based only on pre-step mask count."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

from bdlm_serving.evaluation.frozen import sha
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.methods.t2t.offer_gate import choose_native
from bdlm_serving.methods.t2t.protocol import Arm, verify_work
from experiments.t2t.prefix_window.score_run import snapshot
from experiments.t2t.serving_slots.switch_report import audit, first_changed_offer, gemma_stages
from tools.run_t2t import durable


HIGH, LOW, THRESHOLD = 256, 128, 64
PLANS = {"high":(HIGH,HIGH,LOW,LOW), "low":(LOW,LOW,HIGH,HIGH)}


def original(phase):
    return "slottrace_" + "_".join("256n" if q==HIGH else str(q) for q in PLANS[phase])


def validate_gate(row, decisions, phase, mode):
    """Reconstruct pre-step inputs from the *previous* event, not the resulting canvas."""
    previous, tick, pattern = None, 0, []
    for event in row["events"]:
        if "slot_offer" in event:
            if previous is None or previous["block"]!=event["block"] or tick>=len(decisions):
                raise ValueError("Missing preceding state or gate decision")
            offered = PLANS[phase][tick%4]
            masks = HIGH-previous["resolved"]
            native = offered==HIGH and (mode=="all" or (masks<=THRESHOLD)==(mode=="tail"))
            effective = HIGH if native else LOW
            record = [tick,event["block"],masks,event["available_fresh"],offered,effective]
            if (record!=decisions[tick] or event["slot_offer"]!=effective
                    or (event["stage"]=="native_offer")!=native):
                raise ValueError("Gate depends on unobserved state or selected the wrong path")
            pattern.append("256n" if native else LOW)
            tick+=1
        previous=event
    if tick!=len(decisions) or mode not in ("tail","early","all"):
        raise ValueError("Spurious gate decision")
    return audit(row,pattern or [LOW]),gemma_stages(row)


def run(args, prepare=None):
    import torch
    from bdlm_serving.methods.t2t.online import load_backend
    if args.model != "gemma":
        raise ValueError("This gate qualification is for Gemma B256 only")
    tasks, digest = load_tasks(args.samples), sha(args.samples)
    args.output.mkdir(parents=True,exist_ok=False)
    paths = {p:snapshot(args.controls,original(p),digest,args.output/".fixed") for p in PLANS}
    matrix=[(mode,phase) for mode in ("all","tail","early") for phase in PLANS]
    names=[f"gate_{mode}_{phase}" for mode,phase in matrix]
    durable(args.output/"run.json",dict(model="gemma",block=HIGH,batch_size=1,suite="frozen90",
        samples_sha256=digest,threshold=THRESHOLD,offers=PLANS,policies=names,
        decision_columns=["tick","block","masks","mandatory","external","effective"],
        timing="synchronized warm backend call including decision log; excludes file I/O, scoring and audit"))
    backend=load_backend(args,prepare)
    for (mode,phase),policy in zip(matrix,names,strict=True):
        plan=PLANS[phase]
        arm=Arm(plan[0],0,plan[0],"hard_prefix_zero")
        clock=[]
        def execute(task):
            log=[]
            def source(info):
                offered=plan[info.tick%len(plan)]
                effective=choose_native(info,offered,LOW,THRESHOLD,mode)
                log.append([info.tick,info.block_start,info.masks,info.mandatory,offered,effective.q])
                return effective
            return backend(task,arm,next_budget=source),log
        warm,wlog=execute(tasks[0])
        for task in tasks:
            torch.cuda.synchronize()
            begin=perf_counter_ns()
            row,decisions=execute(task)
            torch.cuda.synchronize()
            elapsed=perf_counter_ns()-begin
            i=task["index"]
            if i==0 and (row!=warm or decisions!=wlog):
                raise ValueError("Warmup differs from first measured request")
            row.update({k:task[k] for k in ("index","task_id","dataset","seed")},model="gemma",policy=policy)
            verify_work(row)
            validate_gate(row,decisions,phase,mode)
            reference=json.loads((paths[phase]/f"{i:02d}-{original(phase)}.json").read_text())
            normalized=json.loads(json.dumps(row))
            first=first_changed_offer(reference,normalized)
            if mode=="all" and (first is not None or dict(normalized,policy=reference["policy"])!=reference):
                raise ValueError("Unconditional gate did not reproduce the original mixed execution")
            row.update(gate_decisions=decisions,first_changed_event=first)
            clock.append(dict(index=i,wall_ns=elapsed))
            durable(args.output/f"{i:02d}-{policy}.json",row)
            print(f"{policy} {i:02d} F={row['generation_forwards']} Q={row['generation_query_rows']}",flush=True)
        durable(args.output/f"timing-{policy}.json",dict(policy=policy,requests=clock))
        durable(args.output/f"complete-{policy}.json",dict(policy=policy,requests=len(tasks)))
    durable(args.output/"replay-qualified.json",dict(samples=digest,matched_per_phase=len(tasks),phases=list(PLANS)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices=("gemma",),default="gemma")
    for field in ("checkpoint","samples","reference","controls","output"):
        p.add_argument("--"+field,type=Path,required=True)
    run(p.parse_args())
