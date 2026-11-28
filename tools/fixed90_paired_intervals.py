"""Paired uncertainty for the registered LLaDA Q16 study, reading only the compact scored pairs."""
import csv
from hashlib import sha256
from io import StringIO
import json
import sys

from tools.agreement_fixed90_comparison import paired_intervals
from experiments.t2t.serving_slots.fixed90_gate import MANIFEST, ROOT


def compute(blob, candidate):
    digest=sha256(blob).hexdigest()
    if digest!="4acd1ebc4127e05fda04e19c9fc723f498ca4bfd9a955026f5652bd527fa115e":
        raise ValueError("different Q16 scored pairs")
    samples=json.loads(MANIFEST.read_bytes())["samples"]
    pairs=list(csv.DictReader(StringIO(blob.decode())))
    assert [int(p["index"]) for p in pairs]==list(range(90))
    found={}
    for name in ("strict","ibs_cap","ibs_nominal"):
        arm={}
        for p,s in zip(pairs,samples):
            i=s["index"]
            assert p["task_id"]==s["task_id"] and int(p["seed"])==s["seed"]
            assert p["case_sha256"]==candidate["seal_digests"][str(i)]
            v={key:{k:int(p[f"{tag}_{k}"]) for k in ("N","F","Q","correct")}
               for key,tag in (("stock",name),("apply","candidate"))}
            assert all(candidate["cases"][str(i)][k]==v["apply"][k] for k in v["apply"])
            arm[i]=v
        found[name]=paired_intervals(arm,samples)
    return dict(scored_pairs_sha256=digest, full90=found)


if __name__=="__main__":
    path=ROOT/"llada-full-admission-q16-frozen90.json"
    print(json.dumps(compute(sys.stdin.buffer.read(),json.loads(path.read_bytes())), indent=2))
