"""Audit the original LLaDA Optimus chunk sizes against physical decoder Q, on frozen30."""
import argparse
from pathlib import Path

from bdlm_serving.evaluation.frozen90 import load_tasks
from experiments.t2t.serving_slots.external_report import read, write
from experiments.t2t.serving_slots.llada_reference import archive


def report(root, cases, out):
    tasks = load_tasks(cases)
    policies = [f"optimus_{m}_c{q}" for q in (4,8,16,32) for m in ("ibs","obs")]
    registered = {r["policy"]:r for r in archive(root,policies,tasks)}
    result = []
    for p in policies:
        cap = int(p.split("_c")[1])
        cuts, exceeded, counts, forwards, qsum = 0,0,0,0,0
        peak = written = correct = 0
        for t in tasks[:30]:
            r = read(root/"requests"/f"{t['index']:02d}-{p}.json")
            events = [e for e in r["events"] if e.get("physical_forwards",1)]
            steps = [(e,c) for e,c in zip(events,r["calls"],strict=True) if e["phase"]!="prefill"]
            if len(steps) != r["generation_forwards"]:
                raise ValueError("Missing physical step")
            excess = []
            for e,c in steps:
                rows = len(e["absolute_positions"])
                if not rows == c["rows"] == c["physical_rows"] == e["q_tokens"]:
                    raise ValueError("Logical plan and observed physical Q disagree")
                peak = max(peak, rows)
                written = max(written,e["accepted"]+e["edited"])
                forwards += 1
                qsum += rows
                excess.append(max(0,rows-cap))
            cuts += sum(excess)
            counts += sum(n>0 for n in excess)
            exceeded += any(excess)
        a = registered[p]
        if (forwards,qsum) != (a["F"],a["physical_Q"]):
            raise ValueError("Work totals differ")
        result.append(dict(model="llada21",cohort="original30",policy=p,requests=30,nominal=cap,
                           correct=a["correct"],physical_Q=qsum,F=forwards,peak_physical_Q=peak,
                           requests_over=exceeded,steps_over=counts,excess_rows=cuts,
                           peak_admitted_or_edited=written,hard_cap_enforced=False))
    write(out,result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("root","cases","out"):
        p.add_argument("--"+k,type=Path,required=True)
    report(**vars(p.parse_args()))
