"""Strict physical-cap IBS: a spare-seat current-block KV refresh contrast."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from experiments.t2t.serving_slots import llada_all_slots90 as reg
from experiments.t2t.serving_slots import spawn_mask_gpu as lease


HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
DATA = reg.DATA
Q = 8
K = 2
LABELS = {0: "optcap_ibs_8", K: "ibs_q8_spare2_right_kvonly"}
PILOT = DATA / "llada21-q8-ibs-prefix-cpu3-v1"
OUT = DATA / "llada21-q8-ibs-prefix-gpu90-v1"
PEER = DATA / "llada21-optimus-hardcap-endpoints"
KEYS = ("index", "task_id", "dataset", "seed")


def sources():
    files = ("bdlm_serving/methods/optimus/fixed.py", "bdlm_serving/methods/optimus/upstream.py",
             "bdlm_serving/methods/optimus/prefix_refresh.py",
             "bdlm_serving/runtime/batch.py", "bdlm_serving/runtime/serial.py",
             "bdlm_serving/evaluation/scoring.py", str(HERE.relative_to(REPO)))
    return {f: reg.sha(REPO/f) for f in files}


def audit(row, quota):
    """Check the cap, eligibility, preserved original seats and physical writes."""
    events, calls = row["events"], row["calls"]
    prompt = len(row["prompt_ids"])
    initialized, dirty = set(), set()
    first, added, gen = None, [], []
    for event, call, inputs in zip(events, calls, row["forward_inputs"], strict=True):
        actual = event["absolute_positions"]
        reg.require(call["rows"] == call["physical_rows"] == event["q_tokens"] == len(actual)
                    and call["batch"] == 1 and len(actual) == len(inputs)
                    and actual == sorted(set(actual)), "Invalid selected projections")
        if event["phase"] == "prefill":
            reg.require(not initialized and actual == list(range(prompt // 32 * 32)),
                        "Unexpected prefill")
            initialized.update(actual)
            continue
        tick = len(gen)
        start = event["block"] * 32
        prior = event["cached_prefix_before"]
        candidates = [start+i for i in event["candidate_local"]]
        required = set(event["dirty_before"])
        reg.require(required == dirty and required.issubset(candidates)
                    and event["physical_cap"] == Q and event["offer_tick"] == tick,
                    "Wrong mandatory state or offer")
        planned = (sorted(dirty) + [i for i in candidates if i not in dirty]
                   if len(candidates) > Q else candidates)
        orig = sorted(planned[:Q])
        sample = [start+i for i in event["selected_local"]]
        reg.require(orig == sample and 0 < len(orig) <= len(actual) <= Q
                    and max(actual) == max(orig) < start+32
                    and event["visible_kv"] == call["kv_rows"] == max(actual)+1,
                    "The ordinary IBS plan or visible key extent changed")
        extra = sorted(set(actual)-set(orig))
        pool = sorted((set(range(max(prompt,start),start+prior)) & initialized)
                      - dirty - set(orig))
        n = min(quota, Q-len(orig), len(pool))
        wanted = pool[-n:] if n else []
        reg.require(extra == wanted and set(actual) == set(orig)|set(extra),
                    "Wrong spare-seat prefix refresh")
        writes = set(event["written_absolute"])
        edits = set(event["edited_positions"])
        reg.require(writes.issubset(orig) and edits.issubset(orig)
                    and all(p >= prompt for p in writes|edits)
                    and len(inputs) == len(actual), "KV-only positions or prompt were sampled")
        dirty = (dirty-set(orig)) | writes
        reg.require(not any(p < start+event["cached_prefix_after"] for p in dirty),
                    "Advanced past an unseen write")
        if extra and first is None:
            first = tick
        initialized.update(actual)
        added.append(len(extra))
        gen.append(event)
    n, f, q = row["output_tokens"], len(gen), sum(e["q_tokens"] for e in gen)
    reg.require(row["generation_forwards"] == f and row["generation_physical_padded_rows"] ==
                row["generation_query_rows"] == q and row["tpf"] == n/f
                and row["token_efficiency"] == n/q and row["prefill_rows"]+q ==
                sum(c["rows"] for c in calls) and row["max_q"] <= Q,
                "Incorrect physical work")
    return dict(first=first, rounds=sum(v>0 for v in added), prefix_Q=sum(added), N=n, F=f, Q=q)


def experiment_pair(a, b):
    from experiments.t2t.serving_slots.llada_reference import PARITY
    wa, wb = audit(a, 0), audit(b, K)
    reg.require(all(a[k] == b[k] for k in (*KEYS, "prompt_ids", "generation_cap"))
                and a["policy"] == LABELS[0] and b["policy"] == LABELS[K], "Different pair")
    first = wb["first"]
    pref = int(a["events"][0]["phase"] == "prefill")
    stop = len(a["events"]) if first is None else first + pref
    if first is None:
        reg.require(all(a[k] == b[k] for k in PARITY if k != "policy")
                    and len(a["events"]) == len(b["events"]), "Unexposed pair diverged")
    for t in range(stop):
        old, new = a["events"][t], b["events"][t]
        reg.require(all(v == new[k] for k,v in old.items())
                    and a["forward_inputs"][t] == b["forward_inputs"][t],
                    "Different history before injection")
    if first is not None:
        old, new = a["events"][stop], b["events"][stop]
        reg.require(all(old[k] == new[k] for k in ("phase","offer_tick","block","dirty_before",
                    "candidate_local", "cached_prefix_before", "selected_local", "selected_masked", "visible_kv")),
                    "The first intervention changed ordinary geometry")
        live = dict(zip(new["absolute_positions"], b["forward_inputs"][stop], strict=True))
        reg.require([live[p] for p in old["absolute_positions"]] == a["forward_inputs"][stop],
                    "Different inputs at first injection")
    return dict(control=wa, refresh=wb, same_ids=a["output_ids"] == b["output_ids"])


def generate(model, tok, task, prompt, cap, quota):
    import torch
    from bdlm_serving.evaluation.outcomes import capture_optimus_progress_failure
    from bdlm_serving.methods.optimus.fixed import FixedOptimus
    from bdlm_serving.methods.t2t.online import package
    from bdlm_serving.runtime.batch import ForwardBatcher, ModelFacade
    feed = []

    class Observed(ForwardBatcher):
        def execute(self, jobs):
            job, = jobs
            feed.append(job["ids"][0].tolist())
            return super().execute(jobs)

    batcher = Observed(model, "llada21", 1)
    facade = ModelFacade(batcher, str(task["index"]), task["seed"])
    engine = FixedOptimus(facade, "llada21", Q, "ibs", physical_caps=(Q,),
                          prefix_refresh=quota, prefix_order="right")
    mask, eos = tok.convert_tokens_to_ids("<|mask|>"), tok.eos_token_id
    ids = torch.tensor([prompt], device=model.device)

    def one():
        try:
            return (*engine.run(ids, cap, task["seed"], mask, [eos]), None)
        except RuntimeError as error:
            return capture_optimus_progress_failure(error, engine)

    measurements = [[] for _ in model.model.layers]
    with ExitStack() as hooks:
        for layer, widths in zip(model.model.layers, measurements, strict=True):
            hooks.enter_context(layer.attention.query_key_value.register_forward_pre_hook(
                lambda _, args, w=widths: w.append(args[0].shape[1])))
        (raw, events, extra, failure), = batcher.run([one])
    widths = [c["rows"] for c in facade.calls]
    reg.require(measurements and all(w == widths for w in measurements)
                and torch.equal(raw[:, :len(prompt)], ids), "All-layer Q or prompt mismatch")
    work = [(e["phase"], e["q_tokens"]) for e in events if e["phase"] != "prefill"]
    out_ids = [] if failure else raw[0, len(prompt):len(prompt)+cap].tolist()
    row = package(tok, out_ids, work, [eos], set(tok.all_special_ids)|{mask,eos},
                  sum(e["q_tokens"] for e in events if e["phase"]=="prefill"), **{k:task[k] for k in KEYS},
                  events=events, calls=facade.calls, extra=extra, prompt_ids=prompt,
                  phase="formal" if cap==2048 else "cpu-pilot", model="llada21", policy=LABELS[quota],
                  generation_cap=cap, request_failure=failure,
                  t2t_edits=sum(e.get("edited",0) for e in events),
                  forward_inputs=feed, verified_layers=len(measurements))
    if failure:
        row.update(status="generation_failed", truncated=False)
    row["remaining_masks"] = row["output_ids"].count(mask)
    return row


def execute(model, tok, tasks, ref, out, *, formal):
    from experiments.t2t.serving_slots.llada_reference import verify_row
    out.mkdir(mode=0o700, exist_ok=False)
    indices = list(range(90)) if formal else [0,10,20]
    cap = 2048 if formal else 96
    reg.save(out/"run.json", dict(model="llada21", arms=LABELS, q=Q, indices=indices, cap=cap,
             samples_sha256=reg.SAMPLES_SHA, sources=sources(), formal=formal, cohort="frozen90"))
    pairs = []
    for i in indices:
        task = tasks[i]
        prompt = reg.read(ref/reg.filename(i,"native32"))["prompt_ids"]
        a, b = [generate(model,tok,task,prompt,cap,k) for k in LABELS]
        if formal:
            old = reg.read(PEER/reg.filename(i,LABELS[0]))
            verify_row(old, a)
            reg.require(old["events"] == a["events"], "Independent IBS8 control differs")
        paired = experiment_pair(a,b)
        for r in (a,b):
            reg.save(out/reg.filename(i,r["policy"]), r)
        pairs.append(dict(index=i,**paired))
        print(f"index={i} first={paired['refresh']['first']} refresh_Q={paired['refresh']['prefix_Q']}",flush=True)
        if i == 29 and formal:
            reg.save(out/"original30.json", pairs)
    reg.save(out/"complete.json",dict(indices=indices, study=reg.sha(out/"run.json"), pairs=pairs))


def cpu():
    import torch
    import transformers.utils as u
    from typing import TypedDict
    if not hasattr(u,"LossKwargs"):
        u.LossKwargs=TypedDict("LossKwargs",{"num_items_in_batch":int},total=False)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.runtime.serial import install_llada_rectangular_mask
    from experiments.t2t.serving_slots.llada_mask_pilot import CKPT
    from experiments.t2t.serving_slots.mask_probe_llada import verify_checkpoint
    from experiments.t2t.serving_slots.top_transition90 import resolve_reference, validate_prompts
    reg.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available()
                and os.sched_getaffinity(0)==set(range(32,36)) and not PILOT.exists(), "CPU preflight only")
    samples=DATA/"samples.json"
    reg.require(reg.sha(samples)==reg.SAMPLES_SHA,"Wrong frozen data")
    tasks=load_tasks(samples)
    ref=resolve_reference(SimpleNamespace(samples=samples))
    verify_checkpoint(CKPT)
    opts=dict(trust_remote_code=True,local_files_only=True)
    tok=AutoTokenizer.from_pretrained(CKPT,**opts)
    validate_prompts(tok,[tasks[i] for i in [0,10,20]],ref)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model=AutoModelForCausalLM.from_pretrained(CKPT,**opts,dtype=torch.bfloat16,
              attn_implementation="sdpa",device_map={"":"cpu"}).eval()
    install_llada_rectangular_mask(model,allow_selected_cache=True)
    execute(model,tok,tasks,ref,PILOT,formal=False)


def preflight():
    receipt=reg.read(PILOT/"complete.json")
    reg.require(receipt["study"]==reg.sha(PILOT/"run.json")
                and reg.read(PILOT/"run.json")["sources"]==sources()
                and receipt["indices"]==[0,10,20]
                and any(p["refresh"]["rounds"] for p in receipt["pairs"]), "Missing exposed CPU pilot")
    for i,p in zip(receipt["indices"],receipt["pairs"],strict=True):
        a,b=[reg.read(PILOT/reg.filename(i,n)) for n in LABELS.values()]
        reg.require(dict(index=i,**experiment_pair(a,b))==p,"Damaged pilot pair")


def run(args,prepare):
    from experiments.t2t.serving_slots.spawn_frozen import load_gpu
    reg.require(args.output==OUT and args.samples==DATA/"samples.json" and prepare and
                os.environ["CUDA_VISIBLE_DEVICES"]==lease.GPU and reg.sha(args.samples)==reg.SAMPLES_SHA
                and reg.read(PEER/f"complete-{LABELS[0]}.json")==dict(policy=LABELS[0],requests=90),
                "Only the registered GPU0 worker may run frozen90")
    preflight()
    model,tok,tasks,ref=load_gpu(args,prepare)
    execute(model,tok,tasks,ref,OUT,formal=True)


def protected():
    preflight()
    launcher=reg.ROOT/"launch_prefix90.py"
    src=launcher.read_bytes()
    route=b'REPO = RUNTIME / "t2t-experiments"'
    call=b"from experiments.t2t.serving_slots.llada_reference import run"
    reg.require(not sys.flags.optimize and reg.sha(launcher)==reg.LAUNCHER_SHA
                and src.count(route)==src.count(call)==1 and REPO.parent==reg.ROOT.parent,
                "Different protected n164 launcher")
    src=src.replace(route,("REPO = Path("+repr(str(REPO))+")").encode()).replace(
        call,b"from experiments.t2t.serving_slots.ibs_prefix_pair import run")
    os.environ.update(PREFIX_MODE="llrefs",LL_BASELINE_POLICIES="ibs-prefix-refresh",
                      TRACE_OUT=str(OUT),FROZEN90_SAMPLES=str(DATA/"samples.json"))
    exec(compile(src,str(launcher),"exec"),{"__name__":"__main__","__file__":str(HERE)})
    holder,=json.loads(subprocess.check_output([reg.HOLDER,"status","0"]))
    pids=subprocess.check_output(["nvidia-smi","-i",lease.GPU,"--query-compute-apps=pid",
                                  "--format=csv,noheader,nounits"]).split()
    reg.require(holder["running"] and holder["gpu_status"]["uuid"]==lease.GPU
                and list(map(int,pids))==[holder["pid"]],"GPU0 did not return to holder")
    done=reg.read(OUT/"complete.json")
    reg.require(done["indices"]==list(range(90)) and len(done["pairs"])==90,"Incomplete study")
    reg.save(OUT/"returned.json", dict(gpu=0,uuid=lease.GPU,pid=holder["pid"],restored=True,
                                      complete=reg.sha(OUT/"complete.json")))


def score():
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.evaluation.scoring import score as judge
    from experiments.t2t.serving_slots.offer_mask_score import summarize
    from experiments.t2t.serving_slots.llada_reference import verify_row
    samples=DATA/"samples.json"
    run_meta,done=reg.read(OUT/"run.json"),reg.read(OUT/"complete.json")
    returned=reg.read(OUT/"returned.json")
    reg.require(run_meta["samples_sha256"]==reg.sha(samples)==reg.SAMPLES_SHA
                and run_meta["sources"]["bdlm_serving/evaluation/scoring.py"]==
                reg.sha(REPO/"bdlm_serving/evaluation/scoring.py")
                and returned["restored"] and returned["uuid"]==lease.GPU
                and returned["complete"]==reg.sha(OUT/"complete.json")
                and done["study"]==reg.sha(OUT/"run.json") and done["indices"]==list(range(90)),
                "Unverified study")
    data=[]
    for task,stamp in zip(load_tasks(samples),done["pairs"],strict=True):
        i=task["index"]
        a,b=[reg.read(OUT/reg.filename(i,n)) for n in LABELS.values()]
        reg.require(all(a[k]==task[k] for k in KEYS),"Wrong task")
        verify_row(reg.read(PEER/reg.filename(i,LABELS[0])),a)
        p=experiment_pair(a,b)
        reg.require(dict(index=i,**p)==stamp,"Changed paired receipt")
        def record(row,w):
            correct=(row["status"]=="success" and bool(judge(row["output_text"],task)["correct"]))
            return (correct,w["N"],w["F"],w["Q"],w["rounds"],w["prefix_Q"])
        data.append((i,task["dataset"],record(a,p["control"]),record(b,p["refresh"])))
    stats=summarize(data)
    for group in stats:
        for key in ("control","trial","delta"):
            if group[key] is not None:
                group[key]["injected_prefix_Q"]=group[key].pop("edge_pairs")
    reg.save(OUT/"aggregate.json",dict(samples_sha256=reg.SAMPLES_SHA,model="llada21",
             q=Q,quota=K,order="right",refresh_only=True,statistics=stats))


def queue():
    """One protected successor; do not seize GPU0 between earlier studies."""
    reg.require(not OUT.exists() and not OUT.with_suffix(".log").exists(),"Study already exists")
    while True:
        status=subprocess.run(["tmux","has-session","-t","=prefix90-attributed-low-offers"],
                              capture_output=True)
        if status.returncode==1:
            break
        status.check_returncode()
        time.sleep(10)
    last=DATA/"gemma-unified90-alternate256-alternate256_unmask256"
    reg.require(reg.read(last/"returned.json")==dict(gpu=0,restored=True)
                and all(reg.read(last/f"complete-{p}.json")==dict(policy=p,requests=90)
                        for p in ("alternate256","alternate256_unmask256")),
                "Predecessor did not complete and restore GPU0")
    preflight()
    with OUT.with_suffix(".log").open("x") as log:
        subprocess.run([sys.executable,"-u",str(HERE),"llada21","0"],cwd=REPO,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    score()


if __name__=="__main__":
    if sys.argv[1:] in (["llada21","0"],["llada21","0","worker"]):
        protected()
    elif sys.argv[1:]==["cpu"]:
        cpu()
    elif sys.argv[1:]==["check"]:
        preflight()
    elif sys.argv[1:]==["score"]:
        score()
    elif sys.argv[1:]==["queue"]:
        queue()
    else:
        raise SystemExit("Use cpu, check, queue, score or the protected llada21 0 launcher")
