"""Paired diagnostics on all 90 LLaDA Q4 requests; emits aggregates, never raw answers."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean, median

from tools.frozen90_six_shards import MANIFEST
from tools.shared_anchor_report import costs, read_receipts

LEDGER_SHA = "ff9caccbf776416f0237ddcc73af774fbe567c57929b359583f70fba128d64ce"


def summarize(pairs):
    if not pairs:
        raise ValueError("empty comparison")
    out = dict(requests=len(pairs), gained=sum(b["correct"] and not a["correct"] for a,b in pairs),
               lost=sum(a["correct"] and not b["correct"] for a,b in pairs))
    for name, rows in zip(("IBS capped","shared"), zip(*pairs), strict=True):
        n, f, q = (sum(r[k] for r in rows) for k in ("N","F","Q"))
        out[name] = dict(correct=sum(r["correct"] for r in rows), N=n, F=f, Q=q,
                         macro_TPF=fmean(r["N"]/r["F"] for r in rows), pooled_TE=n/q,
                         median_N=median(r["N"] for r in rows), output_N_ge_2000=sum(r["N"]>=2000 for r in rows))
    return out


def run(root, base, ledger):
    content = ledger.read_bytes()
    if sha256(content).hexdigest() != LEDGER_SHA:
        raise ValueError("reference ledger fingerprint mismatch")
    arm = json.loads(content)["arms"]["llada21:optcap_ibs_4"]
    samples = json.loads(MANIFEST.read_bytes())["samples"]
    version, data, _ = read_receipts(root,samples,base)
    selected = [data["llada21",4,j] for j in range(90)]
    pairs = list(zip(arm["items"],selected,strict=True))
    if (arm["q"] != 4 or len(pairs) != 90 or
        any(any(a[k] != b[k] for k in ("index","dataset","task_id","seed")) for a,b in pairs)):
        raise ValueError("not a complete matched Q4 panel")
    roles = {k:sum(costs(r)[k] for r in selected) for k in costs(selected[0])}
    groups = {"all90": pairs, "original30":pairs[:30], "new60":pairs[30:]}
    for d in sorted({r["dataset"] for r in selected}):
        groups[d] = [(a,b) for a,b in pairs if b["dataset"]==d]
    return dict(source=version,reference=LEDGER_SHA,physical_Q_by_role=roles,
                groups={k:summarize(v) for k,v in groups.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root","registration","ledger"):
        p.add_argument(key,type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.root,json.loads(a.registration.read_bytes()),a.ledger),allow_nan=False,indent=2))
