"""Reuse the sealed parent single-board guard, with a tighter timeout."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

D = Path(__file__).resolve().parent
parent = D.parent
path = parent/"llada_flow_active_w2_guard.sh"
if sha256(path.read_bytes()).hexdigest() != "32c8017c47b1e78639a2878b756b44a2435bd26ddb9327baf0b0eee95f4c4ff5":
    raise RuntimeError("unknown handoff implementation")
audits = json.loads((parent/"raw-audit.json").read_bytes())
if any(audits[arm]["requests"] != 90 for arm in ("flow_w2","active_w2")):
    raise RuntimeError("parent raw audit incomplete")
if not (parent/"report.verified").is_file():
    raise RuntimeError("parent not independently replay-audited")
subprocess.run(["sha256sum","-c","source.sha256"],cwd=parent,check=True)
s = path.read_text()
returned = json.loads((parent/"handoff.ndjson").read_text().splitlines()[-1])
if returned["phase"] != "returned":
    raise RuntimeError("parent still owns card")
pid = int(returned["pid"])
start = int(returned["start_ticks"])
for a,b in [
    ("D=$R/results/llada-flow-active-w2-h4-v1\n","D=$R/results/llada-flow-active-w2-h4-v1/first-forward\n"),
    ("llada_flow_active_w2_study.py","probe.py"),
    ("(( seconds <= 14400 )) || seconds=14400","(( seconds <= 600 )) || seconds=600"),
    ("holder before\n",f'test "$(apps)" = "{pid}"\n'
                       f'test "$(awk \'{{print $22}}\' /proc/{pid}/stat)" = "{start}"\nholder before\n')]:
    if s.count(a)!=1:
        raise RuntimeError("unexpected guard source")
    s=s.replace(a,b)
subprocess.run(["bash","-c",s],check=True)
