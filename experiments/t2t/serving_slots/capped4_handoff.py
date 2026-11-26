"""GPU4 extension, serialized after the native B32 switching qualification."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.t2t.serving_slots.expanded_handoff import SHA, SAMPLES
from experiments.t2t.serving_slots.llada_capped import policies


if sys.argv[1:2]==["chain"] and len(sys.argv)==3:
    root=Path(sys.argv[2]).resolve()
    data=root/"prefix90"
    while not subprocess.run(["tmux","has-session","-t","=prefix90-llada-native-slot"],
                             capture_output=True).returncode:
        time.sleep(5)
    proof=json.loads((root/"private-llada-native-slot90-v1"/"verified.json").read_text())
    if (proof["model"],proof["samples_sha256"],len(proof["policies"]),proof["physical_gpu"])!=(
            "llada21",SAMPLES,4,"4"):
        raise RuntimeError("Incomplete GPU4 predecessor")
    for group in ("anchor","switch","prefix","endpoints","mid","low"):
        if group=="prefix":
            out=data/"llada21-native-high-prefix16"
            patterns=("16r1","32n","32n_32n_16r1_16r1","16r1_16r1_32n_32n")
            def arm(s):
                q,m=(32,32) if s.startswith("32n") else (16,15)
                return f"{q}:0:{m}:slottrace_{s}"
            env=dict(os.environ,PREFIX_MODE="online",FROZEN90_SAMPLES=str(data/"samples.json"),TRACE_OUT=str(out),
                     T2T_ARMS=" ".join(map(arm,patterns)),
                     T2T_SLOT_CONTROLS=f"16r1={data/'llada21-16-slack'} 32n={data/'llada21-32-matched'}")
            with out.with_suffix(".log").open("x") as log:
                subprocess.run([sys.executable,"-u",root/"launch_prefix90.py","llada21","4"],cwd=root,
                               env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            for s in patterns:
                p="slottrace_"+s
                if json.loads((out/f"complete-{p}.json").read_text())!=dict(policy=p,requests=90):
                    raise ValueError("Unqualified LLaDA prefix replay")
            continue
        out=data/f"llada21-optimus-hardcap-{group}"
        env=dict(os.environ,PREFIX_MODE="llrefs",FROZEN90_SAMPLES=str(data/"samples.json"),TRACE_OUT=str(out),
                 LL_BASELINE_POLICIES=" ".join(policies(group)))
        with out.with_suffix(".log").open("x") as log:
            subprocess.run([sys.executable,"-u",__file__,"llada21","4"],cwd=root,env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        if any(json.loads((out/f"complete-{p}.json").read_text())!=dict(policy=p,requests=90)
               for p in policies(group)):
            raise RuntimeError("Incomplete capped baseline")
elif sys.argv[1:] in (["llada21","4"],["llada21","4","worker"]):
    launcher=Path(os.environ["FROZEN90_SAMPLES"]).resolve().parents[1]/"launch_prefix90.py"
    src=launcher.read_bytes()
    target=b"from experiments.t2t.serving_slots.llada_reference import run"
    if hashlib.sha256(src).hexdigest()!=SHA or src.count(target)!=1 or os.environ["PREFIX_MODE"]!="llrefs":
        raise RuntimeError("Unreviewed launcher or mode")
    exec(compile(src.replace(target,b"from experiments.t2t.serving_slots.llada_capped import run"),
                 str(launcher),"exec"),globals())
else:
    raise ValueError("Only registered GPU4 chain supported")
