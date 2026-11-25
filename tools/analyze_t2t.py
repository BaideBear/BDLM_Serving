"""Evaluate pre-forward rankings in teacher-forced native traces. No training or GPU needed."""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

from bdlm_serving.evaluation.frozen import MANIFEST
from bdlm_serving.methods.t2t.planner import History

SIGNALS = ("uniform_expected", "uncertain", "young", "neighbor", "recent_then_uncertain")


def replay(path, model, stats):
    with gzip.open(path, "rt") as f:
        records = map(json.loads, f)
        meta = next(records)
        idx = meta["index"]
        task = json.loads(MANIFEST.read_text())["samples"][idx]
        assert all(task[k] == meta[k] for k in ("index","dataset","task_id","seed"))
        ll = model == "llada21"
        plen = next(records)["prompt_length"] if ll else 0
        fold = "development" if idx % 10 < 7 else "holdout"
        previous = history = None
        steps = edits = drops = drafts = first = 0
        for row in records:
            start, before, after, accepted = (row[k] for k in ("block","before","after","accepted"))
            b = len(before)
            assert b == (32 if ll else 256) and all(len(row[k]) == b for k in ("after","accepted","draft","certainty"))
            valid = [i for i in range(b) if i+start >= plen]
            if previous is None or previous["block"] != start:
                previous = None
                history = History([False]*b, [None]*b, [0]*b, set(), set())
            else:
                assert previous["after"] == before and row["clock"] == previous["clock"] + (1 if ll else -1)
            # Keep old rankings and eligibility distinct from outcome calculation.
            resolved = [i for i in valid if history.resolved[i]]
            targets = {i for i in resolved if accepted[i] and before[i] != after[i]}
            drops += sum(not accepted[i] for i in resolved)
            fresh = {i for i in valid if accepted[i] and (not history.resolved[i] or i in targets)}
            first += sum(not history.resolved[i] for i in fresh)
            if previous:
                drafts += sum(previous["draft"][i] != row["draft"][i] for i in valid)
            edits += len(targets)
            steps += 1
            for pool, candidates in (("all_resolved", resolved), ("outside_fresh", [i for i in resolved if i not in history.fresh])):
                if not candidates:
                    continue
                positives = len(targets.intersection(candidates))
                for signal in SIGNALS:
                    ranking = [] if signal=="uniform_expected" else history.rank(candidates,signal)
                    for k in (1,2,4,8):
                        n = min(len(candidates), k)
                        hits = n * positives / len(candidates) if signal=="uniform_expected" else len(targets.intersection(ranking[:n]))
                        for group in ("all",fold):
                            acc = stats[group,pool,signal,k]
                            for j,v in enumerate((hits,n,positives,len(candidates))):
                                acc[j] += v
            history = History(accepted, row["certainty"],
                [0 if i in fresh or not accepted[i] else history.streak[i]+1 for i in range(b)],
                fresh | {i for i in resolved if not accepted[i]}, fresh)
            previous = row
        return dict(index=idx,dataset=meta["dataset"],task_id=meta["task_id"],fold=fold,
                    denoise_steps=steps,admissions=first,t2t=edits,revoked=drops,draft_changes=drafts)


def write(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main(args):
    assert json.loads((args.raw/"verified.json").read_text()) == dict(model=args.model,matched_requests=30)
    stats = defaultdict(lambda: [0,0,0,0])
    req = [replay(args.raw/f"{i:02d}.trace.jsonl.gz",args.model,stats) for i in range(30)]
    assert [r["index"] for r in req] == list(range(30))
    metric = []
    for (fold,pool,signal,k),(hit,q,pos,eligible) in sorted(stats.items()):
        metric.append(dict(fold=fold,pool=pool,signal=signal,k=k,hits=hit,charged_slots=q,
            positives=pos,eligible=eligible,recall=hit/pos if pos else "",precision=hit/q if q else ""))
    args.out.mkdir(exist_ok=True,parents=True)
    write(args.out/f"{args.model}-paths.csv",req)
    write(args.out/f"{args.model}-rankings.csv",metric)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices=("gemma","llada21"),required=True)
    for k in ("raw","out"):
        p.add_argument("--"+k,type=Path,required=True)
    main(p.parse_args())
