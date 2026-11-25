"""Evaluate pre-forward rankings in teacher-forced native traces. No training or GPU needed."""
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from bdlm_serving.evaluation.frozen import MANIFEST
from bdlm_serving.methods.t2t.planner import History

SIGNALS = ("uniform_expected", "uncertain", "young", "neighbor", "recent_then_uncertain")


def final_fate(index, output, before, after):
    """Endpoint only: an ID can disappear and later reappear."""
    if index < 0 or before == after:
        raise ValueError("Expected a generated position and a strict token change")
    if index >= len(output):
        return "not_returned"
    token = output[index]
    return "new_id" if token == after else "old_id" if token == before else "other_id"


def retention_counts(positions):
    groups = defaultdict(Counter)
    for p in positions:
        if p["event"] == "t2t":
            key = p["native_phase"], p["previously_fresh"]
            groups[key][p["final_fate"]] += 1
    return [dict(native_phase=k[0], previously_fresh=k[1], strict_edits=sum(c.values()),
                 **{name:c[name] for name in ("new_id","old_id","other_id","not_returned")})
            for k,c in sorted(groups.items())]


def replay(path, model, stats, positions=None, native=None):
    with gzip.open(path, "rt") as f:
        records = map(json.loads, f)
        meta = next(records)
        idx = meta["index"]
        task = json.loads(MANIFEST.read_text())["samples"][idx]
        assert all(task[k] == meta[k] for k in ("index","dataset","task_id","seed"))
        ll = model == "llada21"
        plen = next(records)["prompt_length"] if ll else 0
        if native is not None:
            assert all(meta[k] == native[k] for k in ("index","dataset","task_id","seed"))
            origin = len(native["prompt_ids"]) if ll else native["prefill_rows"]
            if ll:
                assert origin == plen
            eos_at = len(native["output_ids"])-1 if native["eos"] else None
            returned = len(native["output_ids"])
        else:
            origin = eos_at = returned = None
        fold = "development" if idx % 10 < 7 else "holdout"
        previous = history = None
        previous_edits = set()
        steps = edits = drops = drafts = first = 0
        for row in records:
            start, before, after, accepted = (row[k] for k in ("block","before","after","accepted"))
            b = len(before)
            assert b == (32 if ll else 256) and all(len(row[k]) == b for k in ("after","accepted","draft","certainty"))
            valid = [i for i in range(b) if i+start >= plen]
            if previous is None or previous["block"] != start:
                previous = None
                previous_edits = set()
                seen = {}
                history = History([False]*b, [None]*b, [0]*b, set(), set())
            else:
                assert previous["after"] == before and row["clock"] == previous["clock"] + (1 if ll else -1)
            # Keep old rankings and eligibility distinct from outcome calculation.
            resolved = [i for i in valid if history.resolved[i]]
            targets = {i for i in resolved if accepted[i] and before[i] != after[i]}
            if positions is not None:
                if native is None:
                    raise ValueError("Positional export requires the verified native response")
                readmit = {i for i in valid if accepted[i] and not history.resolved[i] and i in seen}
                phase = ("canvas_denoise" if not ll else "bootstrap" if previous is None else
                         "verify" if len(resolved) == len(valid) else "selected")
                for i in sorted(targets | readmit):
                    rel = start + i - origin
                    assert rel >= 0
                    region = "after_return" if rel >= returned else "eos" if rel == eos_at else "pre_eos"
                    kind = "t2t" if i in targets else ("readmit_different" if after[i] != seen[i] else "readmit_same")
                    positions.append(dict(index=idx,dataset=meta["dataset"],task_id=meta["task_id"],
                        request_denoise_step=steps+1,block_start_abs=start,clock=row["clock"],offset_in_block=i,
                        output_offset=rel,event=kind,previously_fresh=int(i in history.fresh),region=region,
                        native_phase=phase,final_fate=final_fate(rel,native["output_ids"],before[i],after[i])
                        if i in targets else ""))
            seen.update((i,after[i]) for i in valid if accepted[i])
            drops += sum(not accepted[i] for i in resolved)
            fresh = {i for i in valid if accepted[i] and (not history.resolved[i] or i in targets)}
            first += sum(not history.resolved[i] for i in fresh)
            if previous:
                drafts += sum(previous["draft"][i] != row["draft"][i] for i in valid)
            edits += len(targets)
            steps += 1
            for pool, candidates in (("all_resolved", resolved),
                                     ("outside_fresh", [i for i in resolved if i not in history.fresh]),
                                     ("fresh_only", [i for i in resolved if i in history.fresh])):
                if not candidates:
                    continue
                positives = len(targets.intersection(candidates))
                special = ("near_miss", "near_miss_gated", "gated_uniform_expected") if ll and pool == "outside_fresh" else ()
                for signal in SIGNALS + special:
                    random = signal.endswith("uniform_expected")
                    if signal in special:
                        nm = {i for i in candidates if previous["draft"][i] != before[i]
                              and history.certainty[i] <= .5}
                        rankable = candidates if signal == "near_miss" else nm
                        ranking = sorted(rankable, key=lambda i: (i not in nm,
                            min((abs(j-i) for j in previous_edits if j != i), default=b),
                            history.certainty[i], i))
                    else:
                        ranking = candidates if random else history.rank(candidates,signal)
                    for k in (1,2,4,8):
                        n = min(len(ranking), k)
                        hits = n * positives / len(candidates) if random else len(targets.intersection(ranking[:n]))
                        for group in ("all",fold):
                            acc = stats[group,pool,signal,k]
                            for j,v in enumerate((hits,n,positives,len(candidates))):
                                acc[j] += v
            history = History(accepted, row["certainty"],
                [0 if i in fresh or not accepted[i] else history.streak[i]+1 for i in range(b)],
                fresh | {i for i in resolved if not accepted[i]}, fresh)
            previous = row
            previous_edits = targets
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
    positions = []
    req = []
    for i in range(30):
        native = (json.loads((args.native_output/f"{i:02d}-native{32 if args.model=='llada21' else 256}.json").read_text())
                  if args.native_output else None)
        req.append(replay(args.raw/f"{i:02d}.trace.jsonl.gz",args.model,stats,
                          positions if native is not None else None,native))
    assert [r["index"] for r in req] == list(range(30))
    metric = []
    for (fold,pool,signal,k),(hit,q,pos,eligible) in sorted(stats.items()):
        metric.append(dict(fold=fold,pool=pool,signal=signal,k=k,hits=hit,charged_slots=q,
            positives=pos,eligible=eligible,recall=hit/pos if pos else "",precision=hit/q if q else ""))
    args.out.mkdir(exist_ok=True,parents=True)
    write(args.out/f"{args.model}-paths.csv",req)
    write(args.out/f"{args.model}-rankings.csv",metric)
    if args.native_output:
        assert sum(p["event"] == "t2t" for p in positions) == sum(r["t2t"] for r in req)
        write(args.out/f"{args.model}-positions.csv",positions)
        write(args.out/f"{args.model}-edit-retention.csv",retention_counts(positions))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices=("gemma","llada21"),required=True)
    p.add_argument("--native-output",type=Path,help="Verified native rows, to export event coordinates without token IDs")
    for k in ("raw","out"):
        p.add_argument("--"+k,type=Path,required=True)
    main(p.parse_args())
