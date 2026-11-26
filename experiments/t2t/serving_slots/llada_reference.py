"""Reproduce the archived LLaDA IBS/OBS and W1/W2 before extending each to 90."""
import argparse
import csv
import json
from pathlib import Path

from bdlm_serving.evaluation.frozen import ROOT, sha
from bdlm_serving.evaluation.frozen90 import load_tasks
from tools.run_t2t import durable

POLICIES = tuple(f"optimus_{m}_c{q}" for q in (8,16,32,4) for m in ("ibs","obs")) + ("flow_w1","flow_w2")
IDENTITY = ("model","index","dataset","task_id","seed","policy","status","phase","generation_cap")
WORK = ("output_tokens","generation_forwards","generation_query_rows","generation_physical_padded_rows","max_q")
PARITY = IDENTITY + WORK + ("prompt_ids","output_ids","output_text","eos","truncated","prefill_rows",
                            "q_histogram","t2t_edits","request_failure")


def read(path):
    return json.loads(path.read_text())


def verify_row(old,new):
    for key in PARITY:
        if old[key] != new[key]:
            raise ValueError(f"Original30 mismatch: {old['index']} {old['policy']} {key}")
    for log,fields in (("calls",("rows","physical_rows","kv_rows","batch")),
                       ("events",("phase","q_tokens","physical_forwards","absolute_positions"))):
        for left,right in zip(old[log],new[log],strict=True):
            if any(left.get(k) != right.get(k) for k in fields):
                raise ValueError(f"Original30 physical trajectory mismatch: {old['index']} {old['policy']} {log}")


def archive(root,policies,tasks):
    meta = read(root/"metadata.json")
    meta90 = read(ROOT/"benchmarks/frozen90/manifest.json")
    comp = read(root/"complete.json")
    if (meta["model"] != "llada21" or meta["sample_sha256"] != meta90["original_samples_sha256"]
            or comp["effective_batch"] != 1 or comp["status"] != "success"
            or meta["generation_cap"] != 2048 or not set(policies) <= set(meta["policies"])
            or not set(policies) <= set(POLICIES) or len(policies) != len(set(policies))):
        raise ValueError("Unexpected LLaDA reference metadata or policies")
    scores = read(root/"frozen_scores.json")
    with (ROOT/"archive/baselines/requests.csv").open() as f:
        public = {(r["policy"],int(r["index"])):r for r in csv.DictReader(f) if r["model"]=="llada21"}
    summary = []
    for p in policies:
        fails = correct = fwd = phys = 0
        for t in tasks[:30]:
            i = t["index"]
            row = read(root/"requests"/f"{i:02d}-{p}.json")
            ref = public[p,i]
            if (any(row[k]!=t[k] for k in ("index","dataset","task_id","seed"))
                    or (row["model"],row["policy"],row["phase"]) != ("llada21",p,"formal")
                    or any(row[k]!=ref[k] for k in ("status","task_id","dataset"))
                    or any(row[k]!=int(ref[k]) for k in WORK)
                    or scores[f"{p}/{i}"]["correct"] != (ref["correct"]=="True")):
                raise ValueError(f"Archive does not reproduce published row: {p}/{i}")
            calls = [c for c,e in zip(row["calls"],[e for e in row["events"] if e.get("physical_forwards",1)],
                                      strict=True) if e["phase"]!="prefill"]
            if (len(calls) != row["generation_forwards"] or sum(c["physical_rows"] for c in calls)
                    != row["generation_physical_padded_rows"] or sum(c["rows"] for c in calls)
                    != row["generation_query_rows"]):
                raise ValueError(f"Archived work differs from physical trace: {p}/{i}")
            fails += row["status"] != "success"
            correct += scores[f"{p}/{i}"]["correct"]
            fwd += len(calls)
            phys += row["generation_physical_padded_rows"]
        summary.append(dict(policy=p,requests=30,failed=fails,correct=correct,F=fwd,physical_Q=phys))
    return summary


def run(args,prepare=None):
    tasks = load_tasks(args.samples)
    policies = args.policy or list(POLICIES)
    summary = archive(args.reference,policies,tasks)
    for s in summary:
        print(s,flush=True)
    if args.check_only:
        return
    from bdlm_serving.generation import run as generate
    for e in read(ROOT/"benchmarks/checkpoints/llada21.json"):
        path = args.checkpoint / e["name"]
        if path.stat().st_size != e["bytes"] or ("sha256" in e and e["sha256"]!=sha(path)):
            raise ValueError(f"Wrong checkpoint: {e['name']}")
    plan = dict(read(ROOT/"archive/baselines/config.json"),policies=policies)
    args.output.mkdir(exist_ok=False,parents=True)
    args.experiment, args.model, args.batch_size = "baselines","llada21",1
    durable(args.output/"run.json",dict(model=args.model,suite="frozen90",samples_sha256=sha(args.samples),
            original30=summary,batch_size=1,hard_q_cap=False,**{k:v for k,v in plan.items() if k!="batch_size"}))

    def check(p,subset):
        if subset[0]["index"] == 0:
            digests = []
            for t in subset:
                key = f"{t['index']:02d}-{p}.json"
                ref = args.reference/"requests"/key
                verify_row(read(ref),read(args.output/key))
                digests.append(sha(ref))
            durable(args.output/f"reproduced30-{p}.json",dict(policy=p,matched=30,reference_sha256=digests))
        else:
            if [t["index"] for t in subset] != list(range(30,90)):
                raise ValueError("Expected remaining original-order 60")
            durable(args.output/f"complete-{p}.json",dict(policy=p,requests=90))

    def prepare_llada(model):
        # Use the same CPU-parameter staging path as the qualified native runner.
        for p in model.parameters():
            p.data = p.data.clone()
        prepare(model)
    generate(args,tasks,plan,prepare_llada if prepare else None,cohort_done=check)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ("reference","samples","output","checkpoint"):
        parser.add_argument("--"+k,type=Path,required=k in ("reference","samples"))
    parser.add_argument("--policy",choices=POLICIES,action="append")
    parser.add_argument("--check-only",action="store_true")
    args = parser.parse_args()
    if not args.check_only and (not args.output or not args.checkpoint):
        parser.error("Generation requires --output and --checkpoint")
    run(args)
