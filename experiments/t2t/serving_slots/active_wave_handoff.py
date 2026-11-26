"""GPU0 successor: qualify W2, then actually project only active B256 canvases."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.t2t.serving_slots.expanded_handoff import SHA, SAMPLES, STAGES, checks


if sys.argv[1:2] in (["chain"],["switch"]) and len(sys.argv)==3:
    root=Path(sys.argv[2]).resolve()
    data=root/"prefix90"
    switch=sys.argv[1]=="switch"
    predecessor="=prefix90-gemma-active512" if switch else "=prefix90-fixed-tiers"
    while not subprocess.run(["tmux","has-session","-t",predecessor],capture_output=True).returncode:
        time.sleep(5)
    for model,q,k,modes in STAGES:
        out=data/f"{model}-registered-q{q}-k{k}"
        checks(out,model,SAMPLES,[f"t2t_q{q}_f0_m{q-k}_hard_prefix_{m}" for m in modes])
    out=data/"gemma-fixed512-active-w2"
    policies=("flow_w2","active_w2")
    if switch:
        from bdlm_serving.methods.t2t.wave_offer import TRACES
        if any(json.loads((out/f"complete-{p}.json").read_text())!=dict(policy=p,requests=90) for p in policies):
            raise ValueError("No qualified compact W2 endpoint")
        out=data/"gemma-external-256-512"
        policies=tuple(TRACES)
    env=dict(os.environ,PREFIX_MODE="wave",FROZEN90_SAMPLES=str(data/"samples.json"),TRACE_OUT=str(out))
    env["WAVE_SWITCH"]=str(int(switch))
    if hashlib.sha256((data/"samples.json").read_bytes()).hexdigest()!=SAMPLES:
        raise ValueError("Wrong tasks")
    with out.with_suffix(".log").open("x") as log:
        subprocess.run([sys.executable,"-u",__file__,"gemma","0"],cwd=root,env=env,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    for p in policies:
        if json.loads((out/f"complete-{p}.json").read_text())!=dict(policy=p, requests=90):
            raise ValueError("Incomplete fixed 512 comparison")
elif sys.argv[1:] in (["gemma","0"],["gemma","0","worker"]):
    launcher=Path(os.environ["FROZEN90_SAMPLES"]).resolve().parents[1]/"launch_prefix90.py"
    src=launcher.read_bytes()
    key=b"run_wave(args, before_cuda=prepare)"
    if os.environ.get("PREFIX_MODE")!="wave" or hashlib.sha256(src).hexdigest()!=SHA or src.count(key)!=1:
        raise ValueError("Not the reviewed Gemma wave launcher")
    body=b'args.tier512_reference = SAMPLES.parent / "gemma-wave-reference90"\n        '
    if os.environ.get("WAVE_SWITCH","0")=="1":
        body+=b'args.wave_switch_reference = SAMPLES.parent / "gemma-fixed512-active-w2"\n        '
    body+=key
    exec(compile(src.replace(key,body),str(launcher),"exec"),globals())
else:
    raise ValueError("Use chain <runtime> or the protected GPU0 launcher")
