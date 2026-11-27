"""Frozen90 Q8 factorial. Reuse the two 0.9 arms; generate only the two 0.5 arms."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.t2t.serving_slots import ibs_prefix_pair as pair
from experiments.t2t.serving_slots import llada_all_slots90 as reg
from experiments.t2t.serving_slots import llada_strict_factorial as pilot

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
OUT = reg.DATA / "llada21-strict-q8-factorial-gpu90-v1"
GPU = pair.lease.GPU
PARENT = "prefix90-ibs8-spare-pair"
NEW = {k: f"strict_ibs_q8_m50_prefix{k}" for k in pair.LABELS}
INDICES = list(range(90))


def checked(row, threshold, k, task, prompt):
    label = (NEW if threshold == .5 else pair.LABELS)[k]
    cfg = (row["config"] if threshold == .5 else row["extra"])
    reg.require(row["policy"] == label and row["generation_cap"] == 2048
                and all(row[key] == task[key] for key in reg.KEYS)
                and row["prompt_ids"] == prompt and cfg["m2t_threshold"] == threshold
                and cfg["t2t_threshold"] == .5 and cfg["mode"] == "ibs"
                and cfg["physical_caps"] == [8] and cfg.get("prefix_refresh", 0) == k,
                "Different frozen task, threshold or budget")
    return pair.audit(row, k)


def witness(a, b):
    # Reuse the independent unchanged-history/first-extra-row contrast.
    return pair.experiment_pair(dict(a, policy=pair.LABELS[0], phase="formal"),
                                dict(b, policy=pair.LABELS[2], phase="formal"))


def preflight():
    out = pilot.CPU_OUT
    m, done = reg.read(out/"run.json"), reg.read(out/"complete.json")
    reg.require(m["kernel_sha256"] == reg.sha(Path(pilot.__file__))
                and m["samples_sha256"] == reg.SAMPLES_SHA and m["indices"] == [0,10,20]
                and done["run_sha256"] == reg.sha(out/"run.json") and len(done["rows"]) == 12,
                "Missing qualified 2048-token CPU factorial")
    for i in m["indices"]:
        for threshold in (.9, .5):
            rows = [reg.read(out/reg.filename(i, f"strict_ibs_q8_m{int(100*threshold)}_prefix{k}"))
                    for k in pair.LABELS]
            w = witness(*rows)
            for k, arm, key in zip(pair.LABELS, rows, ("control","refresh"), strict=True):
                stamp, = (r for r in done["rows"] if (r["index"],r["threshold"],r["k"]) == (i,threshold,k))
                reg.require(all(stamp[x] == y for x,y in w[key].items())
                            and arm["score"]["correct"] == stamp["correct"], "Damaged pilot")


def predecessor():
    group = pair.OUT
    done = reg.read(group/"complete.json")
    returned = reg.read(group/"returned.json")
    reg.require(done["indices"] == INDICES and done["study"] == reg.sha(group/"run.json")
                and len(done["pairs"]) == 90 and returned["complete"] == reg.sha(group/"complete.json")
                and returned["restored"] and returned["uuid"] == GPU
                and reg.read(group/"aggregate.json")["samples_sha256"] == reg.SAMPLES_SHA
                and reg.read(group/"run.json")["sources"] == pair.sources(), "Unverified reused arms")
    return returned["complete"]


def receipt():
    return dict(q=8, cap=2048, indices=INDICES, uuid=GPU, generation_arms=list(NEW.values()),
                reused_arms=list(pair.LABELS.values()), reference_digest=predecessor(),
                samples_sha256=reg.SAMPLES_SHA, pilot=reg.sha(pilot.CPU_OUT/"complete.json"),
                sources={**pair.sources(), str(HERE.relative_to(REPO)):reg.sha(HERE),
                    str(Path(pilot.__file__).resolve().relative_to(REPO)):reg.sha(Path(pilot.__file__))})


def run(args, prepare):
    import torch
    from experiments.t2t.serving_slots.spawn_frozen import load_gpu
    reg.require(prepare and os.environ["CUDA_VISIBLE_DEVICES"] == GPU and args.output == OUT
                and args.samples == reg.DATA/"samples.json" and reg.sha(args.samples) == reg.SAMPLES_SHA,
                "Unregistered worker")
    preflight()
    meta = receipt()
    model,tok,tasks,ref = load_gpu(args,prepare)
    OUT.mkdir(mode=0o700,exist_ok=False)
    reg.save(OUT/"run.json", meta)
    files = {}
    with torch.inference_mode():
        for i, task in zip(INDICES,tasks,strict=True):
            reg.require(task["index"] == i,"Different task order")
            prompt = reg.read(ref/reg.filename(i,"native32"))["prompt_ids"]
            rows = []
            for k, label in NEW.items():
                row, _ = pilot.generate(model,tok,task,prompt,.5,k)
                checked(row,.5,k,task,prompt)
                rows.append(row)
                name = reg.filename(i,label)
                reg.save(OUT/name,row)
                files[name] = reg.sha(OUT/name)
            witness(*rows)
            print(f"qualified strict Q8 pair {i:02d}",flush=True)
            if i in (29,89):
                reg.save(OUT/f"sealed-{i+1}.json",dict(run=reg.sha(OUT/"run.json"),files=files))


def protected():
    predecessor()
    preflight()
    launch = reg.ROOT/"launch_prefix90.py"
    src = launch.read_bytes()
    route = b'REPO = RUNTIME / "t2t-experiments"'
    call = b"from experiments.t2t.serving_slots.llada_reference import run"
    reg.require(not sys.flags.optimize and sys.argv[1:] in (["llada21","0"],["llada21","0","worker"])
                and reg.sha(launch) == reg.LAUNCHER_SHA and src.count(route) == src.count(call) == 1
                and REPO.parent == reg.ROOT.parent,"Wrong protected launcher")
    src = src.replace(route,("REPO = Path("+repr(str(REPO))+")").encode()).replace(
        call,b"from experiments.t2t.serving_slots.llada_strict_study import run")
    os.environ.update(PREFIX_MODE="llrefs",LL_BASELINE_POLICIES="q8-factorial",
                      TRACE_OUT=str(OUT),FROZEN90_SAMPLES=str(reg.DATA/"samples.json"))
    exec(compile(src,str(launch),"exec"),{"__name__":"__main__","__file__":str(HERE)})
    holder, = json.loads(subprocess.check_output([reg.HOLDER,"status","0"]))
    pids = subprocess.check_output(["nvidia-smi","-i",GPU,"--query-compute-apps=pid",
                                    "--format=csv,noheader,nounits"]).split()
    reg.require(holder["running"] and holder["gpu_status"]["uuid"] == GPU
                and list(map(int,pids)) == [holder["pid"]],"Holder not restored")
    reg.save(OUT/"returned.json",dict(uuid=GPU,pid=holder["pid"],restored=True,
                                      sealed=reg.sha(OUT/"sealed-90.json")))


def score():
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.evaluation.scoring import score as judge
    from experiments.t2t.serving_slots.offer_mask_score import summarize
    from experiments.t2t.serving_slots.llada_reference import verify_row

    stamp, returned = reg.read(OUT/"sealed-90.json"),reg.read(OUT/"returned.json")
    reg.require(returned["restored"] and returned["uuid"] == GPU
                and returned["sealed"] == reg.sha(OUT/"sealed-90.json")
                and stamp["run"] == reg.sha(OUT/"run.json") and reg.read(OUT/"run.json")==receipt(),
                "Unverified study")
    old_stamps = reg.read(pair.OUT/"complete.json")["pairs"]
    cells = []
    for i,task in zip(INDICES,load_tasks(reg.DATA/"samples.json"),strict=True):
        points = {}
        for threshold in (.9,.5):
            group, labels = (pair.OUT,pair.LABELS) if threshold == .9 else (OUT,NEW)
            rows = [reg.read(group/reg.filename(i,lab)) for lab in labels.values()]
            if threshold == .9:
                verify_row(reg.read(pair.PEER/reg.filename(i,labels[0])),rows[0])
                reg.require(old_stamps[i] == dict(index=i,**pair.experiment_pair(*rows)),
                            "Changed original GPU pair")
                prompt = rows[0]["prompt_ids"]
            else:
                reg.require(all(stamp["files"][reg.filename(i,lab)] ==
                            reg.sha(group/reg.filename(i,lab)) for lab in labels.values()),"Changed new files")
                witness(*rows)
            for k,row in zip(labels,rows,strict=True):
                w = checked(row,threshold,k,task,prompt)
                correct = row["status"]=="success" and bool(judge(row["output_text"],task)["correct"])
                points[threshold,k] = (correct,w["N"],w["F"],w["Q"],w["rounds"],w["prefix_Q"])
        cells.append((i,task["dataset"],points))
    comparisons = {}
    for a,b in [((.9,0),(.9,2)),((.5,0),(.5,2)),((.9,0),(.5,0)),((.9,2),(.5,2))]:
        s = summarize([(i,d,v[a],v[b]) for i,d,v in cells])
        for item in s:
            for key in ("control","trial","delta"):
                if item[key] is not None:
                    item[key]["injected_prefix_Q"] = item[key].pop("edge_pairs")
        comparisons[f"{a}->{b}"] = s
    reg.save(OUT/"aggregate.json",dict(q=8,indices=INDICES,samples_sha256=reg.SAMPLES_SHA,
             contrasts=comparisons,exposed_definition="positive injected prefix in trial arm"))


def queue():
    preflight()
    log = OUT.with_suffix(".log")
    reg.require(not OUT.exists() and not log.exists(),"Study already started")
    while True:
        status = subprocess.run(["tmux","has-session","-t","="+PARENT],capture_output=True)
        if status.returncode==1:
            break
        status.check_returncode()
        time.sleep(15)
    predecessor()
    with log.open("x") as handle:
        subprocess.run([sys.executable,"-Bu",str(HERE),"llada21","0"],cwd=REPO,
                       stdout=handle,stderr=subprocess.STDOUT,check=True)
    score()


if __name__ == "__main__":
    if sys.argv[1:] == ["queue"]:
        queue()
    elif sys.argv[1:] == ["check"]:
        preflight()
    elif sys.argv[1:] == ["score"]:
        score()
    else:
        protected()
