"""Score complete T2T arms and publish sanitized paired frozen30 results."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from bdlm_serving.evaluation.frozen import ROOT, load_tasks
from bdlm_serving.evaluation.scoring import score
from bdlm_serving.methods.t2t.protocol import verify_work


def table(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def summarize(rows):
    n, f, q = (sum(r[k] for r in rows) for k in ("N", "F", "Q"))
    return dict(requests=len(rows), correct=sum(r["correct"] for r in rows),
                failed=sum(r["failed"] for r in rows), truncated=sum(r["truncated"] for r in rows),
                N=n, F=f, Q=q, physical_Q=sum(r["physical_Q"] for r in rows),
                tpf=mean(r["N"]/r["F"] for r in rows), token_efficiency=n/q, q_per_f=q/f,
                max_q=max(r["max_q"] for r in rows))


def arm_sources(folders):
    if not folders:
        raise ValueError("No input run")
    sources, names = [], set()
    ident = None
    for folder in folders:
        meta = json.loads((folder / "run.json").read_text())
        key = (meta["model"], meta["block"])
        if ident is not None and ident != key:
            raise ValueError("Do not merge different models or block widths")
        ident = key
        if json.loads((folder / "qualified.json").read_text()) != dict(model=key[0], full_width=key[1], matched=30):
            raise ValueError(f"Missing full-width gate: {folder.name}")
        for arm, policy in zip(meta["arms"], meta["policies"], strict=True):
            if policy in names:
                raise ValueError(f"Duplicate policy across runs: {policy}")
            names.add(policy)
            sources.append((folder,arm,policy))
    return *ident, sources


def main(args):
    tasks = load_tasks(args.samples)
    model, b, sources = arm_sources(args.run)
    args.out.mkdir(parents=True, exist_ok=True)
    identity = lambda t: tuple(t[k] for k in ("index", "dataset", "task_id"))
    ordered = [identity(t) for t in tasks]
    baseline = []
    with (ROOT/"experiments/baselines/requests.csv").open() as src:
        for x in csv.DictReader(src):
            if (x["model"], x["policy"], x["role"]) != (model, "native", "baseline"):
                continue
            baseline.append(dict(model=model, policy="native", nominal_q=b, fresh="", mask="", edit="",
                index=int(x["index"]), dataset=x["dataset"], task_id=x["task_id"], max_q=b,
                **{k:int(x[k]) for k in ("N","F","Q","physical_Q","correct","failed","truncated")}))
    baseline.sort(key=lambda r:r["index"])
    if [identity(r) for r in baseline] != ordered:
        raise ValueError("No matching native baseline")
    arms, phases = [baseline], []
    for folder, arm, policy in sources:
        complete = folder / f"complete-{policy}.json"
        if not complete.exists() and args.only_complete:
            continue
        if json.loads(complete.read_text()) != dict(policy=policy, requests=30):
            raise ValueError(f"Incomplete arm: {policy}")
        rows, phase_f, phase_q = [], Counter(), Counter()
        cache = folder/f"marks-{policy}.json"
        marks = json.loads(cache.read_text()) if cache.exists() else {}
        for t in tasks:
            r = json.loads((folder / f"{t['index']:02d}-{policy}.json").read_text())
            if r["seed"] != t["seed"] or identity(t) != identity(r) or r["policy"] != policy or r["model"] != model:
                raise ValueError("Wrong request or policy")
            verify_work(r)
            if sum(r["q_by_phase"].values()) != r["generation_query_rows"] or sum(r["forwards_by_phase"].values()) != r["generation_forwards"]:
                raise ValueError("Phase totals disagree")
            failed = r["status"] != "success"
            if str(t["index"]) not in marks:
                marks[str(t["index"])] = not failed and bool(score(r["output_text"], t)["correct"])
            rows.append(dict(model=model,policy=policy,nominal_q=arm["budget"],fresh=arm["fresh"],
                mask=arm["mask"],edit=arm["budget"]-arm["fresh"]-arm["mask"],index=t["index"],
                dataset=t["dataset"],task_id=t["task_id"],max_q=r["max_q"],N=r["output_tokens"],
                F=r["generation_forwards"],Q=r["generation_query_rows"],physical_Q=r["generation_physical_padded_rows"],
                correct=int(marks[str(t["index"])]),failed=int(failed),truncated=int(r["truncated"])))
            phase_f.update(r["forwards_by_phase"])
            phase_q.update(r["q_by_phase"])
        if not cache.exists():
            with cache.open("x") as f:
                json.dump(marks, f)
        phases.extend(dict(policy=policy, phase=p, F=n, Q=phase_q[p]) for p,n in sorted(phase_f.items()))
        arms.append(rows)
    if not phases:
        raise ValueError("No fully measured sparse arm yet")
    all_rows = [r for a in arms for r in a]
    summaries = []
    for a in arms:
        for group in ("all", "gsm8k", "math500", "humaneval"):
            group_rows = [r for r in a if group == "all" or r["dataset"] == group]
            summaries.append(dict(model=model,policy=a[0]["policy"],nominal_q=a[0]["nominal_q"],
                fresh=a[0]["fresh"],mask=a[0]["mask"],edit=a[0]["edit"],dataset=group,**summarize(group_rows)))
    for name, rows in (("requests", all_rows), ("summary", summaries), ("phases", phases)):
        table(args.out/f"{model}-online-{name}.csv", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("samples", "out"):
        p.add_argument("--"+k, type=Path, required=True)
    p.add_argument("--run", type=Path, required=True, action="append", help="Repeat for disjoint policies of one model")
    p.add_argument("--only-complete", action="store_true", help="Allow interim reports for complete 30-request arms")
    main(p.parse_args())
