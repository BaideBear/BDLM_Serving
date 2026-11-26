"""Describe the first physical update after an external offer changes inside a block."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from bdlm_serving.evaluation.frozen import sha
from bdlm_serving.evaluation.frozen90 import load_tasks
from experiments.t2t.prefix_window.score_run import snapshot
from experiments.t2t.serving_slots.external_report import read, write
from experiments.t2t.serving_slots.switch_report import audit


SERIES = {
    "llada21-slots-8-16": ("llada21", ("8_8_16_16","16_16_8_8")),
    "gemma-native-high-prefix128": ("gemma", ("128u32_128u32_256n_256n","256n_256n_128u32_128u32")),
}


def boundary_rows(row):
    """Exclude first selections and block transitions, which inherently include bootstrap."""
    prev=None
    for j,e in enumerate(row["events"]):
        if "slot_offer" not in e:
            continue
        q=e["slot_offer"]
        if prev and e["block"]==prev[0]:
            old, last=prev[1:]
            ll=row["model"]=="llada21"
            actual=e["physical_q" if ll else "q"]
            yield ("up" if q>old else "down" if old>q else "hold",old,q), Counter(
                offers=1, physical_Q=actual, required=e["mandatory_fresh"],
                above_nominal=max(actual-q,0), over_rounds=int(actual>q),
                intermediate_events=j-last-1,
                net_admitted=e["accepted"] if ll else e["admitted"]-e["revoked"],
                edits=e["edited"])
        prev=(e["block"],q,j)


def report(data,out):
    work=[]
    tasks=load_tasks(data/"samples.json")
    digest=sha(data/"samples.json")
    for subdir,(model,patterns) in SERIES.items():
        for pat in patterns:
            name="slottrace_"+pat
            raw=snapshot(data/subdir,name,digest,out.parent/".inputs")
            groups=defaultdict(Counter)
            for task in tasks:
                row=read(raw/f"{task['index']:02d}-{name}.json")
                if (row["model"]!=model or row["policy"]!=name or
                    any(row[k]!=task[k] for k in ("index","task_id","dataset","seed"))):
                    raise ValueError("Different request or offer sequence")
                audit(row,pat.split("_"))
                seen=set()
                for key,counts in boundary_rows(row):
                    for cohort in ("all90", "original30" if task["index"]<30 else "added60"):
                        tag=(cohort,*key)
                        groups[tag].update(counts)
                        seen.add(tag)
                for tag in seen:
                    groups[tag]["exposed_requests"]+=1
            for (cohort,direction,old,new),c in sorted(groups.items()):
                work.append(dict(model=model,pattern=pat,cohort=cohort,direction=direction,
                                 previous_q=old,next_q=new,**c))
    write(out,work)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    report(a.data,a.out)
