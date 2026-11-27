"""Report the three fixed low-budget arms only on identical, sealed CPU requests."""
import argparse
import csv
from statistics import fmean
import sys

from .llada_all_slots90 import DATA, read, sha, require
from .fixed_cold_exposure import exposure
from .llada_low_start_pilot import INDICES, ARMS


def table(case=None):
    root = DATA/"llada21-low-start-cpu3-v1"
    register = root/"run.json"
    meta = read(register)
    require(meta["samples"] == sha(DATA/"samples.json") and meta["indices"] == list(INDICES)
            and meta["budgets"] == [4,8] and meta["arms"] == list(ARMS), "Invalid protocol")
    seal = None if case is not None else read(root/"complete.json")
    if seal is not None:
        require(seal["registration"] == sha(register) and len(seal["files"]) == 18, "Incomplete run")
    records=[]
    for q in (4,8):
        idxs = INDICES if case is None else (case,)
        paths={a:[root/f"{i:02}-q{q}-{a}.json" for i in idxs] for a in ARMS}
        if case is not None and not all(p.is_file() for group in paths.values() for p in group):
            continue
        pairs={}
        for arm in ("eager", "resident", "credit"):
            group=[]
            for i, p in zip(idxs, paths[arm], strict=True):
                if seal is not None:
                    require(seal["files"].get(p.name) == sha(p), "Changed sealed row")
                row=read(p)
                require(row["registration_sha256"] == sha(register) and row["index"] == i and
                        row["arm"] == arm and row["q"] == q and
                        row["exposure"] == exposure(row), "Wrong row or exposure")
                calls=row["trace"]
                require(len(calls)==row["F"] and sum(c["q"] for c in calls)==row["Q"] and
                        all(c["layer_physical_q"]==[c["q"]]*20 and c["q"]<=q for c in calls),
                        "Invalid F/Q")
                key=tuple(row[k] for k in ("input_sha256","seed","dataset","task_id"))
                require(pairs.setdefault(i,key)==key,"Unpaired input")
                group.append(row)
            n,f,paid=(sum(r[k] for r in group) for k in ("N","F","Q"))
            visits=lambda k:sum(r["row_visits"].get(k,0) for r in group)
            records.append(dict(indices=",".join(map(str,idxs)),q=q,arm=arm,correct=sum(
                r["correct"] for r in group),requests=len(group),macro_TPF=fmean(r["N"]/r["F"] for r in group),
                efficiency=n/paid,N=n,F=f,paid_Q=paid,build_F=sum(r["phase_F"].get("build",0) for r in group),
                build_Q=sum(c["q"] for r in group for c in r["trace"] if c["phase"]=="build"),
                prefix_Q=visits("prefix"),prefix_edits=visits("prefix_edits"),
                first_before_full=sum(r["exposure"]["first_before_full"] for r in group),
                failures=sum(bool(r["failure"]) for r in group)))
    require(records,"No complete matched comparison for this item")
    return records


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case",type=int,choices=INDICES)
    rows=table(ap.parse_args().case)
    out=csv.DictWriter(sys.stdout,fieldnames=rows[0],lineterminator="\n")
    out.writeheader()
    out.writerows(rows)
