"""Score complete, nonoverlapping frozen90 LLaDA shards on the evaluation host."""
import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

from bdlm_serving.evaluation.frozen import sha
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.evaluation.scoring import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("shards", type=Path, nargs="+")
    args = p.parse_args()
    tasks = load_tasks(args.samples)
    suite_sha = sha(args.samples)
    by_id, expected = {}, None
    for root in args.shards:
        meta = json.loads((root / "run.json").read_bytes())
        done = json.loads((root / "complete.json").read_bytes())
        spec = {k: meta[k] for k in ("model","gate","nominal_C","effective_C","protocol","generation_cap")}
        if (meta["suite"] != "frozen90" or meta["samples_sha256"] != suite_sha
                or len(meta["indices"]) != done["requests"] or spec["model"] != "llada21"
                or (expected is not None and spec != expected)):
            raise ValueError(f"incompatible or unfinished shard: {root}")
        expected = spec
        for i in meta["indices"]:
            row = json.loads((root / f"{i:02d}-{done['tag']}.json").read_bytes())
            if i in by_id or any(row[k] != tasks[i][k] for k in ("index","task_id","dataset","seed")):
                raise ValueError(f"non-unique or wrong task: {root} #{i}")
            by_id[i] = row
    if set(by_id) != set(range(90)):
        raise ValueError("all 90 tasks required")
    scores = []
    for i in range(90):
        row = by_id[i]
        n,f,q = (row[k] for k in ("output_tokens","generation_forwards","generation_physical_padded_rows"))
        if n < 0 or min(f,q) <= 0:
            raise ValueError(f"invalid measured work on task {i}")
        scores.append((bool(score(row["output_text"], tasks[i])["correct"]),n,f,q))
    with args.output.open("x",newline="") as stream:
        w = csv.writer(stream)
        w.writerow(("C","cohort","requests","correct","macro_tpf","token_efficiency","N","F","physical_Q"))
        for name, subset in (("all90",scores),("original30",scores[:30]),("added60",scores[30:])):
            correct,n,f,q = map(sum,zip(*subset))
            w.writerow((expected["nominal_C"],name,len(subset),correct,
                        fmean(r[1]/r[2] for r in subset),n/q,n,f,q))


if __name__ == "__main__":
    main()
