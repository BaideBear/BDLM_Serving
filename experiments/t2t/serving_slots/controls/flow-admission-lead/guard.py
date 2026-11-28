"""Run only with the exact returned H4 holder; reuse the sealed handoff trap."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

D=Path(__file__).resolve().parent
base=D.parent/"llada-flow-active-w2-h4-v1"
path=base/"llada_flow_active_w2_guard.sh"
if sha256(path.read_bytes()).hexdigest()!="32c8017c47b1e78639a2878b756b44a2435bd26ddb9327baf0b0eee95f4c4ff5":
    raise RuntimeError("unknown H4 handoff source")
for folder in (base,base/"first-forward"):
    subprocess.run(["sha256sum","-c","source.sha256"],cwd=folder,check=True,stdout=sys.stderr)
    if not (folder/"complete.json").is_file():
        raise RuntimeError("previous work not sealed")
returned=json.loads((base/"first-forward/handoff.ndjson").read_text().splitlines()[-1])
if returned["phase"]!="returned":
    raise RuntimeError("no valid return")
pid,start=int(returned["pid"]),int(returned["start_ticks"])
s=path.read_text()
for a,b in [
        ("D=$R/results/llada-flow-active-w2-h4-v1\n","D=$R/results/llada-flow-admission-lead-h4-v1\n"),
        ("llada_flow_active_w2_study.py","study.py"),
        ("holder before\n",f'test "$(apps)" = "{pid}"\n'
                           f'test "$(awk \'{{print $22}}\' /proc/{pid}/stat)" = "{start}"\nholder before\n')]:
    if s.count(a)!=1:
        raise RuntimeError("unexpected guard source")
    s=s.replace(a,b)
subprocess.run(["bash","-c",s],check=True)
