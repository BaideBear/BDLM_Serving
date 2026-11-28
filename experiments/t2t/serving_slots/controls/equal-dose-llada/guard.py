"""Use the established n164 physical-4 lock and holder verification/restore trap."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

D=Path(__file__).resolve().parent
prior=D.parent/"llada-ibs-q8-prefix-change-rank-h4-six-v2"
reg=json.loads((D/"registration.json").read_bytes())
if D.name!=reg["id"] or reg["board"]!="n164:H4":
    raise RuntimeError("wrong deployment")
root=D.parent.parent
src=root/"runtime/llada-ibs-q16-paid-agreement-frozen90-h0-v1"
if sha256((src/"q16-source.sha256").read_bytes()).hexdigest()!=reg["q16_source_manifest"]:
    raise ValueError("unknown frozen runtime")
for cwd,manifest in ((prior,"report.verified"),(src,"q16-source.sha256"),(D,"source.sha256")):
    subprocess.run(["sha256sum","--quiet","-c",manifest],cwd=cwd,check=True)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",
         PYTHONPATH=f"{src}:{root}/runtime/fixed-baseline-deps")
subprocess.run([sys.executable,"-B","-m","unittest","-v","test_adapter"],cwd=D,env=env,check=True)
subprocess.run([sys.executable,"-B","study.py","check"],cwd=D,env=env,check=True)
h=json.loads((prior/"handoff.ndjson").read_text().splitlines()[-1])
if h["phase"]!="returned" or list(D.glob("case-*.json")) or (D/"start.json").exists():
    raise ValueError("not a fresh handoff")
p=prior.parent/"llada-flow-active-w2-h4-v1/llada_flow_active_w2_guard.sh"
if sha256(p.read_bytes()).hexdigest()!="32c8017c47b1e78639a2878b756b44a2435bd26ddb9327baf0b0eee95f4c4ff5":
    raise ValueError("unknown guard")
s=p.read_text()
pid,start=int(h["pid"]),int(h["start_ticks"])
for a,b,count in [
    ("D=$R/results/llada-flow-active-w2-h4-v1\n",f"D=$R/results/{D.name}\n",1),
    ('cd "$T/formal90-7a38ffc/src"',f'cd "{src}"',1),
    ("llada_flow_active_w2_study.py","study.py",1),("14400","2400",2),
    ("holder before\n",f'test "$(apps)" = "{pid}"\n'
     f'test "$(awk \'{{print $22}}\' /proc/{pid}/stat)" = "{start}"\nholder before\n',1)]:
    if s.count(a)!=count: raise ValueError("guard anchor")
    s=s.replace(a,b)
subprocess.run(["bash","-c",s],check=True)
