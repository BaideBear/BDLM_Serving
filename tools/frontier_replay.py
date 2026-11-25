"""Same-state hard-refresh coverage and oracle headroom on verified native trajectories.

Decision inputs are native committed writes and a simulated query clock, not logits.
Native writes outside the replay's simulated Q are visible: this is a teacher-forced
diagnostic, not a sparse rollout or a full-state oracle for a diverged online policy.
"""
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from bdlm_serving.evaluation.frozen import MANIFEST
from bdlm_serving.methods.t2t.frontier import select_hard


def replay(path, model, index, sample, budgets):
    width = 32 if model == "llada21" else 256
    with gzip.open(path, "rt") as stream:
        it = map(json.loads, stream)
        meta = next(it)
        assert all(meta[k] == sample[k] for k in ("index", "task_id", "dataset", "seed"))
        plen = next(it)["prompt_length"] if model == "llada21" else 0
        counts = defaultdict(Counter)
        variants = [(q, e) for q in budgets for e in sorted({0, max(1, q//8), max(1, q//4)})]
        block = old_after = None
        for row in it:
            start, before = row["block"], row["before"]
            new = block != start
            if new:
                block, step = start, 0
                history = {key: [-1]*width for key in variants}
                valid = list(range(max(0, plen-start), width))
                resolved = [False]*width
                write = [-1]*width
                fresh = set()
            else:
                assert old_after == before
            step += 1
            # Only the evaluation below is allowed to read current outcomes.
            rankings = {}
            for (q,e), last_query in history.items():
                mode = "hard_frontier" if e else "hard_noedit"
                if new:
                    chosen, reserve = valid, []
                else:
                    chosen, reserve = select_hard(valid, resolved, fresh, write, last_query,
                                                  step, q, e, mode, index)
                assert fresh.issubset(chosen) and set(reserve).issubset(chosen)
                rankings[q,e] = (chosen,reserve)
            after, accepted = row["after"], row["accepted"]
            targets = {i for i in valid if resolved[i] and accepted[i] and before[i] != after[i]}
            pool = {i for i in valid if resolved[i]} - fresh
            old = targets & pool
            for (q,e),(chosen,reserve) in rankings.items():
                c = counts[q,e]
                c.update(dict(steps=1, positives=len(targets), fresh_hits=len(fresh & targets),
                              old_positives=len(old), reserved=len(reserve), reserved_hits=len(old.intersection(reserve)),
                              expected_uniform=len(reserve)*len(old)/len(pool) if pool else 0.,
                              oracle_reserved_hits=min(len(old),len(reserve)),
                              old_hits_including_backfill=len(old.intersection(chosen)), total_hits=len(targets.intersection(chosen)),
                              selected_steps=int(not new), fresh_overflow_steps=int(not new and len(fresh)>q)))
                for i in chosen:
                    history[q,e][i] = step
            changed = {i for i in valid if resolved[i] != accepted[i] or
                       (accepted[i] and before[i] != after[i])}
            for i in changed:
                write[i] = step
            fresh = {i for i in changed if accepted[i]}
            old_after, resolved = after, accepted
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True, help="Directory containing the two model trace folders")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    tasks = json.loads(MANIFEST.read_text())["samples"]
    assert len(tasks) == 30
    results = []
    for model, budgets, expected in (("llada21",(4,8,16,32,64),754),("gemma",(16,32,64,128,256),10)):
        root = args.raw/model
        assert json.loads((root/"verified.json").read_text()) == {"model":model,"matched_requests":30}
        acc = defaultdict(Counter)
        for i, task in enumerate(tasks):
            assert i == task["index"]
            counts = replay(root/f"{i:02d}.trace.jsonl.gz",model,i,task,budgets)
            for (q,e), c in counts.items():
                for fold in ("all", "holdout" if i%10>=7 else "development"):
                    acc[fold,q,e].update(c)
        for (fold,q,e), c in sorted(acc.items()):
            if fold == "all":
                assert c["positives"] == expected and c["fresh_hits"]+c["old_positives"] == expected
                assert c["old_positives"] == (311 if model=="llada21" else 0)
            results.append(dict(model=model,fold=fold,q=q,extra_edit_cap=e,**c))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f,results[0].keys(),lineterminator="\n")
        w.writeheader()
        w.writerows(results)


if __name__ == "__main__":
    main()
