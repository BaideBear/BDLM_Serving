"""Count strict edits inside a stable generated prefix on verified native inputs."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path

from bdlm_serving.evaluation.frozen import MANIFEST


def stable_prefix(valid, accepted, fresh, before, previous_input):
    rows = []
    for i in valid:
        if not accepted[i] or i in fresh or before[i] != previous_input[i]:
            break
        rows.append(i)
    return rows


def audit(root, sample, model):
    llada = model == "llada21"
    width = 32 if llada else 256
    c = Counter(states=0, nonempty=0, prefix_rows=0, strict=0, fresh_strict=0,
                nonfresh_strict=0, prefix_strict=0, states_with_prefix_strict=0)
    with gzip.open(root / f"{sample['index']:02d}.trace.jsonl.gz", "rt") as f:
        records = map(json.loads, f)
        meta = next(records)
        assert all(meta[k] == sample[k] for k in ("dataset", "task_id", "index", "seed"))
        prompt = next(records)["prompt_length"] if llada else 0
        block = None
        for r in records:
            before = r["before"]
            new = block != r["block"]
            if new:
                block = r["block"]
                valid = range(max(0, prompt-block), width)
                resolved, fresh = [False]*width, set()
            else:
                assert previous_output == before
            # Native pre-state only; verification-only clocks are not opportunities.
            eligible_state = not new and (any(not resolved[i] for i in valid)
                                          if llada else r["clock"] > 1)
            pool = stable_prefix(valid, resolved, fresh, before, previous_input) if eligible_state else []
            accepted, after = r["accepted"], r["after"]
            assert len(before) == len(after) == len(accepted) == width
            strict = {i for i in valid if resolved[i] and accepted[i] and after[i] != before[i]}
            if eligible_state:
                hits = strict.intersection(pool)
                c.update(states=1, nonempty=int(bool(pool)), prefix_rows=len(pool),
                         strict=len(strict), fresh_strict=len(strict & fresh),
                         nonfresh_strict=len(strict-fresh), prefix_strict=len(hits),
                         states_with_prefix_strict=int(bool(hits)))
            fresh = strict | {i for i in valid if accepted[i] and not resolved[i]}
            resolved, previous_input, previous_output = accepted, before, after
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    tasks = json.loads(MANIFEST.read_text())["samples"]
    assert [s["index"] for s in tasks] == list(range(30))
    records = []
    for model in ("llada21", "gemma"):
        root = args.raw / model
        assert json.loads((root/"verified.json").read_text()) == dict(model=model, matched_requests=30)
        counts = [audit(root, s, model) for s in tasks]
        total = Counter()
        for row in counts:
            total.update(row)
        assert total["strict"] == total["fresh_strict"] + total["nonfresh_strict"]
        assert total["nonfresh_strict"] == (191 if model == "llada21" else 0)
        if model == "llada21":
            assert total["states"] == 1170
        records.append(dict(model=model, requests=len(tasks), requests_with_prefix_strict=sum(
            c["prefix_strict"] > 0 for c in counts), **total))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, records[0].keys(), lineterminator="\n")
        w.writeheader()
        w.writerows(records)


if __name__ == "__main__":
    main()
