"""Original IBS32 replay and pre-projection capped Optimus comparisons on frozen90."""
from pathlib import Path

from bdlm_serving.evaluation.frozen import ROOT, sha
from bdlm_serving.evaluation.frozen90 import load_tasks
from experiments.t2t.serving_slots.llada_reference import read, verify_row
from tools.run_t2t import durable


def policies(group):
    groups = {"anchor":((32,),), "switch":((8,8,16,16),(16,16,8,8)),
              "endpoints":((8,), (16,)), "mid":((24,), (12,)), "low":((6,), (4,))}
    return ["optcap_"+m+"_"+"_".join(map(str,qs)) for qs in groups[group] for m in ("ibs","obs")]


def audit(row, caps):
    events = [e for e in row["events"] if e.get("physical_forwards",1)]
    dirty = set()
    tick = rows = 0
    for e,c in zip(events,row["calls"],strict=True):
        if e["phase"]=="prefill":
            continue
        offer = caps[tick%len(caps)]
        picked = set(e["absolute_positions"])
        candidate = [e["block"]*32+i for i in e["candidate_local"]]
        writes = set(e["written_absolute"])
        if (e["offer_tick"]!=tick or e["physical_cap"]!=offer or
                not len(picked)==len(e["absolute_positions"])==c["physical_rows"]==c["rows"]==e["q_tokens"]
                or not 0<len(picked)<=offer or not writes<=picked or
                not picked<=set(candidate) or set(e["dirty_before"])!=dirty or
                (len(candidate)<=offer and set(candidate)!=picked) or
                (dirty-picked and not picked<=dirty) or e["accepted"]+e["edited"]>len(picked)):
            raise ValueError("Physical cap, selected positions, or pending replay disagrees")
        dirty = (dirty-picked)|writes
        if any(a<e["block"]*32+e["cached_prefix_after"] for a in dirty):
            raise ValueError("Cached an outstanding write")
        tick += 1
        rows += len(picked)
    if (tick!=row["generation_forwards"] or rows!=row["generation_query_rows"]
            or rows!=row["generation_physical_padded_rows"]):
        raise ValueError("Q/F ledger mismatch")


def run(args,prepare=None):
    from bdlm_serving.generation import run as generate
    tasks = load_tasks(args.samples)
    arms = args.policy
    if arms not in (policies(g) for g in ("anchor","switch","endpoints","mid","low")):
        raise ValueError("Unregistered capped sequence")
    for e in read(ROOT/"benchmarks/checkpoints/llada21.json"):
        f = args.checkpoint/e["name"]
        if f.stat().st_size!=e["bytes"] or ("sha256" in e and sha(f)!=e["sha256"]):
            raise ValueError("Wrong checkpoint")
    data = args.samples.parent
    peer = data/"llada21-reference90-q32"
    digest = sha(args.samples)
    if (read(peer/"run.json")["samples_sha256"]!=digest or
        read(peer/"complete-optimus_ibs_c32.json")!=dict(policy="optimus_ibs_c32",requests=90)):
        raise ValueError("Missing frozen90 IBS32 reference")
    if arms!=policies("anchor"):
        qualified=read(data/"llada21-optimus-hardcap-anchor"/"replay-qualified.json")
        if qualified!=dict(model="llada21",control="optimus_ibs_c32",samples=digest,matched=90):
            raise ValueError("Missing no-clipping physical replay qualification")
    out = args.output
    out.mkdir(exist_ok=False)
    durable(out/"run.json",dict(model="llada21",suite="frozen90",samples_sha256=digest,
        block=32,batch_size=1,hard_q_cap=True,cap_clock="all_decoder_forwards_request_local",
        cap_excludes="prompt_prefill_only",policies=arms))
    args.experiment,args.model,args.batch_size="baselines","llada21",1
    plan = dict(read(ROOT/"archive/baselines/config.json"),policies=arms,mix_policies=False)

    def check(p,subset):
        caps=tuple(map(int,p.split("_")[2:]))
        for task in subset:
            i=task["index"]
            r=read(out/f"{i:02d}-{p}.json")
            if any(r[k]!=task[k] for k in ("index","task_id","seed","dataset")):
                raise ValueError("Different request")
            audit(r,caps)
            if p=="optcap_ibs_32":
                q="optimus_ibs_c32"
                verify_row(read(peer/f"{i:02d}-{q}.json"),dict(r,policy=q))
        if subset[-1]["index"]==89:
            durable(out/f"complete-{p}.json",dict(policy=p,requests=90))

    def stage(model):
        for par in model.parameters():
            par.data=par.data.clone()
        prepare(model)
    generate(args,tasks,plan,stage if prepare else None,cohort_done=check)
    if arms==policies("anchor"):
        durable(out/"replay-qualified.json",dict(model="llada21",control="optimus_ibs_c32",samples=digest,matched=90))
