"""Use the site's protected stock ready/go/return runner for the ablation.

Usage: CUDA_VISIBLE_DEVICES='' <venv>/python block_ablation_gpu.py MODEL CARD DIR
DIR must be a fresh direct child of the n164 prefix90 experiment directory.
The site supervisor binds the child to exactly one physical UUID. Do not
start if another experiment owns the requested card.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SELF = Path(__file__).resolve()
REPO = SELF.parents[3]
RT = Path("/data1/project_data/gx-erqw-20261123/runtime")
SITE = RT/"t2t-native-1125"
HOLDER = "/data1/project_data/gx-megrez-20261122/gpu-holders/gpu-hold"
UUID = {"0":"GPU-ceadb3cd-e63f-1446-a2d5-a291144f8efb","4":"GPU-d3bdc2bc-d19d-e98f-ac79-ab9c6e887e0e"}


def source():
    p=SITE/"launch_prefix90.py"
    raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!="ba80a293773326950738fd0d4ca0a8a04e6df4b86c6e9a7265bd2ba3a92e6e59":
        raise RuntimeError("Stock supervisor changed")
    for old,new in [
        ('REPO = RUNTIME / "t2t-experiments"',f"REPO = Path({str(REPO)!r})"),
        ("required_fixed = fixed_baseline_requirement(MODE)",'required_fixed = ["math_verify"]'),
        ("assert process.wait(timeout=10800) == 0","wait_child(process,pids)"),
    ]:
        if raw.count(old.encode())!=1:
            raise RuntimeError(f"Unexpected supervisor source: {old}")
        raw=raw.replace(old.encode(),new.encode())
    return compile(raw,str(p),"exec")


def wait_child(proc,pids):
    while True:
        try:
            if proc.wait(timeout=20):
                raise RuntimeError("Ablation worker failed; see log")
            return
        except subprocess.TimeoutExpired:
            if pids()-{proc.pid}:
                raise RuntimeError("Another process entered GPU; abort only this worker")


def pids(card):
    return set(map(int,subprocess.check_output(["nvidia-smi","-i",UUID[card],
        "--query-compute-apps=pid","--format=csv,noheader,nounits"]).split()))


def holder(card):
    h,=json.loads(subprocess.check_output([HOLDER,"status",card]))
    if not h["running"] or h["gpu_status"]["uuid"]!=UUID[card]:
        raise RuntimeError("Expected managed holder")
    return h["pid"]


def worker(model,card):
    from experiments.t2t.serving_slots.block_ablation import run
    import torch
    flag=Path(os.environ["DG_SIGNAL"])
    parent=int(os.environ["BLOCK_ABLATION_PARENT"])
    hh=int(os.environ["BLOCK_ABLATION_HOLDER"])
    if (os.getppid()!=parent or os.environ["CUDA_VISIBLE_DEVICES"]!=UUID[card] or
        flag.parent!=SITE or not flag.is_dir() or (flag/"go").exists() or
        holder(card)!=hh or pids(card)!={hh}):
        raise RuntimeError("Wrong protected worker or holder")
    guard=torch.empty(8<<20,dtype=torch.uint8,device="cuda").zero_()
    torch.cuda.synchronize()
    if pids(card)!={hh,os.getpid()}:
        raise RuntimeError("Missing allocation before handoff")
    (flag/"ready").touch(exist_ok=False)
    deadline=time.monotonic()+180
    while not (flag/"go").exists():
        if os.getppid()!=parent or time.monotonic()>deadline:
            raise RuntimeError("No grant")
        time.sleep(.05)
    if pids(card)!={os.getpid()}:
        raise RuntimeError("Nonexclusive grant")
    run(model,Path(os.environ["TRACE_OUT"]),Path(os.environ["FROZEN90_SAMPLES"]))
    if not guard.data_ptr() or os.getppid()!=parent or pids(card)!={os.getpid()}:
        raise RuntimeError("Lost grant at end")


def main():
    model,card,arg=sys.argv[1:]
    if model not in ("llada21","gemma") or card not in UUID or sys.flags.optimize:
        raise ValueError("Bad input")
    if arg=="worker":
        return worker(model,card)
    out=Path(arg)
    if (out.parent!=SITE/"prefix90" or out!=out.resolve() or out.exists() or
        not out.name.startswith("block-ablation-") or os.environ.get("CUDA_VISIBLE_DEVICES")!=""):
        raise ValueError("Use a fresh cluster result directory and hidden-GPU supervisor")
    hh=holder(card)
    if pids(card)!={hh}:
        raise RuntimeError("Foreign activity; leave all processes alone")
    os.umask(0o077)
    os.environ.update(TRACE_OUT=str(out),FROZEN90_SAMPLES=str(SITE/"prefix90/samples.json"),
        PREFIX_MODE="wave" if model=="gemma" else "llrefs",BLOCK_ABLATION_PARENT=str(os.getpid()),
        BLOCK_ABLATION_HOLDER=str(hh),PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1")
    sys.argv=[str(SELF),model,card]
    exec(source(),{"__name__":"__main__","__file__":str(SELF),"wait_child":wait_child})
    if pids(card)!={holder(card)}:
        raise RuntimeError("Could not verify return")


if __name__=="__main__":
    main()
