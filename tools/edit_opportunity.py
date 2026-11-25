"""Separate unavoidable full verification from addressable LLaDA edit positions.

Evaluate a logit-free positional hazard on held-out requests. The frozen native
path supplies historical writes, identical for all rankers, so results describe
selection opportunity, not sparse decoding quality. No token identities, drafts,
probabilities, texts, data-set names or per-request IDs enter the predictor.
"""
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from bdlm_serving.evaluation.frozen import MANIFEST
from bdlm_serving.methods.t2t.frontier import select_hard
from bdlm_serving.methods.t2t.hazard import features


def load(path, task, *, old_token_features=None, history_factory=None, context_features=None):
    """Append optional old-row features; only completed transitions reach history."""
    steps = []
    with gzip.open(path, "rt") as f:
        source = map(json.loads, f)
        meta, prompt = next(source), next(source)
        assert all(meta[k] == task[k] for k in ("index", "task_id", "seed", "dataset"))
        last = None
        for row in source:
            start, before = row["block"], row["before"]
            new = start != last
            if new:
                clock = 0
                valid = list(range(max(0, prompt["prompt_length"]-start), 32))
                fresh, strict, write = set(), set(), [-1]*32
                history = history_factory() if history_factory is not None else None
            else:
                assert before == prior
            clock += 1
            masks = [i for i in valid if before[i] == prompt["mask_id"]]
            old = [i for i in valid if i not in fresh and i not in masks]
            phase = "bootstrap" if new else "verify" if not masks else "selected"
            # Freeze the decision features before exposing this step's labels.
            x = np.asarray([features(i, masks, fresh, strict, write, clock) for i in old])
            if old and context_features is not None:
                extra = context_features(old, [before[i] for i in old],
                                         [(j, before[j]) for j in sorted(fresh)], x)
                x = np.column_stack((x, extra))
            if old and old_token_features is not None:
                extra = [old_token_features(before[i]) for i in old]
                x = np.column_stack((x, extra))
            if old and history is not None:
                x = np.column_stack((x, [history.features(i) for i in old]))
            after = row["after"]
            changed = {i for i in valid if before[i] != after[i]}
            positive = {i for i in changed if before[i] != prompt["mask_id"]
                        and after[i] != prompt["mask_id"]}
            steps.append((phase, valid, masks, old, fresh, write.copy(), clock, x, positive))
            fresh = {i for i in changed if after[i] != prompt["mask_id"]}
            strict = positive
            if history is not None:
                history.observe(strict, fresh)
            for i in changed:
                write[i] = clock
            prior, last = after, start
    return steps


def fit(train):
    xs, ys = [], []
    for path in train:
        for phase, _, _, old, _, _, _, x, y in path:
            if phase == "selected" and old:
                xs.extend(x)
                ys.extend(float(i in y) for i in old)
    x, y = np.asarray(xs), np.asarray(ys)
    # Predeclared L2=2 for slopes; intercept is unpenalized. No holdout tuning.
    penalty = np.r_[0., np.full(x.shape[1]-1, 2.)]

    def loss(w):
        z = x @ w
        return (sum(np.logaddexp(0., z) - y*z) + .5*np.dot(w*penalty, w),
                x.T @ (expit(z)-y) + penalty*w)
    result = minimize(loss, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 500, "ftol": 1e-12})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def evaluate(path, weights, q, e, kind):
    c = Counter()
    for phase, valid, masks, old, fresh, write, clock, x, pos in path:
        if phase == "bootstrap":
            last_q = [-1]*32
        old_hits = pos.intersection(old)
        c[phase + "_strict"] += len(pos)
        c[phase + "_old"] += len(old_hits)
        if phase != "selected" or q >= 32:
            chosen = valid
            c["all_covered"] += len(pos)
        else:
            resolved = [i not in masks for i in range(32)]
            chosen, reserve = select_hard(valid, resolved, fresh, write, last_q, clock, q, e,
                                          "hard_frontier" if e else "hard_noedit", 0)
            if kind == "hazard" and reserve:
                ranked = sorted(zip(-(x @ weights), old))
                reserve = [i for _, i in ranked[:len(reserve)]]
                # Identical physical cap, mandatory MASK, MASK fill and neutral backfill.
                cap = len(chosen)
                selected = set(fresh) | set(reserve)
                for i in [*masks, *old]:
                    if len(selected) >= cap:
                        break
                    selected.add(i)
                chosen = sorted(selected)
                assert len(chosen) == cap and set(fresh) <= selected
            c["sparse_seats"] += len(reserve)
            c["sparse_reserved_hits"] += len(old_hits.intersection(reserve))
            c["sparse_uniform_expectation"] += len(reserve)*len(old_hits)/len(old) if old else 0.
            c["sparse_slot_oracle"] += min(len(reserve), len(old_hits))
            c["sparse_covered_old"] += len(old_hits.intersection(chosen))
            c["all_covered"] += len(pos.intersection(chosen))
        for i in chosen:
            last_q[i] = clock
    return c


def shortlist(path, weights):
    """Rank the whole old pool before applying layout, fresh or MASK quotas."""
    result = defaultdict(Counter)
    for phase, _, _, old, _, _, _, x, pos in path:
        if phase != "selected" or not old:
            continue
        targets = pos.intersection(old)
        ranking = [i for _, i in sorted(zip(-(x @ weights), old))]
        for k in (1, 2, 4, 8, 16, 32):
            picked = ranking[:k]
            result[k].update(steps=1, old_candidates=len(old), seats=len(picked),
                             positives=len(targets), hits=len(targets.intersection(picked)),
                             uniform_expected=len(picked)*len(targets)/len(old))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shortlist", type=Path)
    args = p.parse_args()
    assert json.loads((args.raw/"verified.json").read_text()) == {"model":"llada21","matched_requests":30}
    tasks = json.loads(MANIFEST.read_text())["samples"]
    paths = [load(args.raw/f"{t['index']:02d}.trace.jsonl.gz", t) for t in tasks]
    # Five disjoint test folds, two cases from each dataset per fold. Each test
    # request uses coefficients fitted on the other 24; no native labels of that
    # request are used to fit or tune the policy.
    totals = defaultdict(Counter)
    curves = defaultdict(Counter)
    for fold in range(5):
        test = [i for i in range(30) if i%5 == fold]
        assert len({tasks[i]["dataset"] for i in test}) == 3
        w = fit([path for i, path in enumerate(paths) if i not in test])
        for i in test:
            for k, c in shortlist(paths[i], w).items():
                for group in ("all", f"fold{fold}"):
                    curves[group,k].update(c)
            for q in (4, 8, 16, 32, 64):
                for e in sorted({0, max(1, q//8), max(1, q//4)} if q < 32 else {0}):
                    for kind in (("frontier", "hazard") if e else ("noedit",)):
                        c = evaluate(paths[i], w, q, e, kind)
                        for group in ("all", f"fold{fold}"):
                            totals[group, q, e, kind].update(c)
    fields = ("bootstrap_strict","verify_strict","selected_strict","bootstrap_old","verify_old",
              "selected_old","sparse_seats","sparse_reserved_hits","sparse_uniform_expectation",
              "sparse_slot_oracle","sparse_covered_old","all_covered")
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("fold","q","edit_cap","ranker",*fields))
        for key, c in sorted(totals.items()):
            if key[0] == "all":
                assert sum(c[k] for k in fields[:3]) == 754
                assert sum(c[k] for k in fields[3:6]) == 311
            writer.writerow((*key, *[c[k] for k in fields]))
    if args.shortlist:
        fields = ("steps", "old_candidates", "seats", "positives", "hits", "uniform_expected")
        with args.shortlist.open("w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(("fold", "top_k", *fields, "recall", "precision"))
            for key, c in sorted(curves.items()):
                writer.writerow((*key, *[c[k] for k in fields], c["hits"]/c["positives"],
                                 c["hits"]/c["seats"]))


if __name__ == "__main__":
    main()
