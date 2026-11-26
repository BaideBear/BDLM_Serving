"""GPU0: fixed Q64 and progress80, causal vs open, fresh paired traces on frozen90."""
from pathlib import Path
import json
import os
import subprocess
import sys

from experiments.t2t.serving_slots import llada_all_slots90 as reg
from experiments.t2t.serving_slots import spawn_frozen as sf

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
GPU = "GPU-ceadb3cd-e63f-1446-a2d5-a291144f8efb"
OUT = reg.DATA / "llada21-q64-progress80-mask-gpu90-v1"
PREV = reg.DATA / "gemma-unified90-replay-replay_q128_prefix32"
ARMS = ("progress80-causal", "progress80-open")


def provenance():
    paths = [HERE, HERE.with_name("llada_spawn_pilot.py"), HERE.with_name("spawn_frozen.py"),
             HERE.with_name("llada_spawn.py"), HERE.with_name("mask_pair_llada.py"),
             HERE.with_name("llada_quote.py"), HERE.with_name("llada_all_slots.py"),
             HERE.with_name("llada_top_transition.py"), HERE.with_name("llada_all_slots90.py"),
             REPO/"bdlm_serving/methods/t2t/llada_dual.py", REPO/"bdlm_serving/methods/t2t/llada_sparse.py",
             REPO/"bdlm_serving/runtime/serial.py", REPO/"bdlm_serving/evaluation/scoring.py"]
    return {str(p.relative_to(REPO)): reg.sha(p) for p in paths}


def protocol():
    from experiments.t2t.serving_slots.llada_mask_pilot import CFG
    return dict(model="llada21", gpu=0, gpu_uuid=GPU, samples_sha256=reg.SAMPLES_SHA,
                launcher_sha256=reg.LAUNCHER_SHA, cap=reg.CAP, config=CFG, q=64,
                gate=.8, arms=list(ARMS), indices=list(range(90)), sources=provenance())


def run(args, prepare):
    import torch
    from experiments.t2t.serving_slots.llada_spawn_pilot import compare
    reg.require(prepare and os.environ["CUDA_VISIBLE_DEVICES"] == GPU and args.output == OUT
                and args.samples == reg.DATA/"samples.json" and reg.sha(args.samples) == reg.SAMPLES_SHA,
                "Requires protected n164 GPU0 worker and frozen sample file")
    os.umask(0o077)
    model, tok, tasks, ref = sf.load_gpu(args, prepare)
    OUT.mkdir(exist_ok=False)
    reg.save(OUT/"run.json", protocol())
    files = {}
    with torch.inference_mode():
        for i, task in enumerate(tasks):
            reg.require(task["index"] == i, "Unexpected request order")
            prompt = reg.read(ref/reg.filename(i, "native32"))["prompt_ids"]
            rows = [sf.generate(model, tok, task, prompt, arm) for arm in ARMS]
            compare(*rows)
            for arm, row in zip(ARMS, rows, strict=True):
                sf.audit(row, arm, opened_control=arm == ARMS[1])
                reg.require(all(row[k] == task[k] for k in reg.KEYS), "Unexpected task")
                name = reg.filename(i, arm)
                reg.save(OUT/name, row)
                files[name] = reg.sha(OUT/name)
            print(f"verified pair {i:02d}", flush=True)
            if i in (29, 89):
                reg.save(OUT/f"sealed-{i+1}.json", dict(run=reg.sha(OUT/"run.json"), files=files))


def sealed(n):
    reg.require(n in (30,90), "Unknown cohort")
    manifest = reg.read(OUT/f"sealed-{n}.json")
    reg.require(manifest["run"] == reg.sha(OUT/"run.json")
                and manifest["files"] == {reg.filename(i,a): reg.sha(OUT/reg.filename(i,a))
                                            for i in range(n) for a in ARMS}, "Wrong sealed cohort")
    return reg.sha(OUT/f"sealed-{n}.json")


def protected():
    reg.require(sys.argv[1:] in (["llada21","0"],["llada21","0","worker"]) and not sys.flags.optimize
                and reg.read(PREV/"returned.json") == {"gpu":0,"restored":True}
                and all(reg.read(PREV/f"complete-{a}.json") == {"requests":90,"policy":a}
                        for a in ("replay","replay_q128_prefix32")), "Not a registered successor")
    launcher = reg.ROOT/"launch_prefix90.py"
    src = launcher.read_bytes()
    route = b'REPO = RUNTIME / "t2t-experiments"'
    call = b"from experiments.t2t.serving_slots.llada_reference import run"
    reg.require(reg.sha(launcher) == reg.LAUNCHER_SHA and src.count(route) == src.count(call) == 1
                and REPO.parent == reg.ROOT.parent, "Launcher or checkout not verified")
    src = src.replace(route, ("REPO = Path("+repr(str(REPO))+")").encode()).replace(
        call, b"from experiments.t2t.serving_slots.spawn_mask_gpu import run")
    os.environ.update(PREFIX_MODE="llrefs", LL_BASELINE_POLICIES="progress80-mask",
                      TRACE_OUT=str(OUT), FROZEN90_SAMPLES=str(reg.DATA/"samples.json"))
    exec(compile(src,str(launcher),"exec"), {"__name__":"__main__","__file__":str(HERE)})
    holder, = json.loads(subprocess.check_output([reg.HOLDER,"status","0"]))
    active = subprocess.check_output(["nvidia-smi","-i",GPU,"--query-compute-apps=pid",
                                      "--format=csv,noheader,nounits"]).split()
    reg.require(holder["running"] and holder["gpu_status"]["uuid"] == GPU
                and list(map(int,active)) == [holder["pid"]], "Holder not restored")
    reg.save(OUT/"returned.json",dict(gpu=0,uuid=GPU,pid=holder["pid"],restored=True,
                                      original30=sealed(30),full90=sealed(90)))


if __name__ == "__main__":
    protected()
