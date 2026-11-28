"""Reuse the sealed lock + H4 identity handoff, only after positive CPU replay."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

D=Path(__file__).resolve().parent
prior=D.parent/"llada-ibs-q8-prefix-change-rank-h4-six-v1"
base=D.parent/"llada-flow-active-w2-h4-v1"
for folder,manifest in ((D,"source.sha256"),(D,"preflight.sha256"),(prior,"source.sha256")):
    subprocess.run(["sha256sum","-c",manifest],cwd=folder,check=True,stdout=sys.stderr)
def read(p): return json.loads(p.read_bytes())
x=read(D/"exposure.json")
pre=read(D/"preflight.json")
if (not x["gate"] or x["registration"]!=sha256((D/"registration.json").read_bytes()).hexdigest()
        or not pre["passed"] or pre["same_raw_sha"]!=sha256((prior/"07-none.json").read_bytes()).hexdigest()
        or (prior/"complete.json").exists() or (prior/"compact.ndjson").stat().st_size
        or sorted(p.name for p in prior.glob("??-*.json"))!=["07-none.json"]
        or (D/"worker.pid").exists() or list(D.glob("??-*.json"))):
    raise RuntimeError("missing exposure or prior seal")
h=json.loads((prior/"handoff.ndjson").read_text().splitlines()[-1])
if h["phase"]!="returned":
    raise RuntimeError("incomplete handoff")
pid,start=h["pid"],h["start_ticks"]
path=base/"llada_flow_active_w2_guard.sh"
if sha256(path.read_bytes()).hexdigest()!="32c8017c47b1e78639a2878b756b44a2435bd26ddb9327baf0b0eee95f4c4ff5":
    raise RuntimeError("unknown handoff guard")
s=path.read_text()
for a,b,n in [
    ("D=$R/results/llada-flow-active-w2-h4-v1\n","D=$R/results/llada-ibs-q8-prefix-change-rank-h4-six-v2\n",1),
    ("llada_flow_active_w2_study.py","study.py",1),("14400","2400",2),
    ("holder before\n",f'test "$(apps)" = "{int(pid)}"\n'
     f'test "$(awk \'{{print $22}}\' /proc/{int(pid)}/stat)" = "{int(start)}"\nholder before\n',1)]:
    if s.count(a)!=n: raise RuntimeError("guard source mismatch")
    s=s.replace(a,b)
subprocess.run(["bash","-c",s],check=True)
