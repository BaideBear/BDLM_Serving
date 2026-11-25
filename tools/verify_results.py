"""Validate shipped tables without torch, network access, or a model checkpoint."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ("baselines", "flowblock-prefix", "optimus-masked-obs")
DATASETS = ("gsm8k", "math500", "humaneval")
NAMES = ("n", "correct", "TPF", "efficiency", "N", "F", "Q", "physical_Q",
         "physical_efficiency", "failed", "truncated", *DATASETS)
SUMMARY = (
    "count correct tpf_mean token_efficiency_pooled output_tokens generation_forwards generation_query_rows generation_physical_padded_rows physical_efficiency_pooled generation_failures truncated gsm8k_correct math500_correct humaneval_correct",
    "requests correct tpf_mean token_efficiency N F useful_Q physical_Q physical_token_efficiency failures truncated gsm8k math500 humaneval",
    "n correct TPF N_Q N F Q physical_Q N_physicalQ failed truncated gsm8k math500 humaneval",
)


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def policy(row):
    if "policy" in row:
        return row["policy"]
    if "arm" in row:
        return f"optimus_{row['arm']}_c{row['cap']}"
    return f"{row['method']}_w{row['W']}"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def checked_table(study, tasks, controls):
    directory = ROOT / "archive" / study
    groups = defaultdict(dict)
    own = json.loads((directory / "config.json").read_text())["policies"]
    shown = set(own)
    if study == STUDIES[2]:
        shown.update(f"optimus_{m}_c{c}" for m in ("ibs","obs") for c in (4,8,16,32))
    for r in read(directory / "requests.csv"):
        model, arm, i = r["model"], policy(r), int(r["index"])
        ctx = f"{study}/{model}/{arm}/{i}"
        check(model in ("llada21","sdar") and arm in shown and i in tasks, ctx)
        row = groups[model, arm]
        task = tasks[i]
        check(i not in row and task["task_id"] == r.get("task_id",r.get("task"))
              and task["dataset"] == r["dataset"], ctx+" sample identity")
        keys = ("N F Q physical_Q maxQ" if "arm" in r else
                "output_tokens generation_forwards generation_query_rows generation_physical_padded_rows max_q")
        n, f, q, qp, peak = [int(r[k]) for k in keys.split()]
        check(f > 0 and n >= 0 and qp >= q >= f and 0 < peak <= q <= peak*f, ctx+" counts")
        check(math.isclose(float(r.get("TPF", r.get("tpf"))), n/f), ctx+" TPF")
        if "token_efficiency" in r:
            check(math.isclose(float(r["token_efficiency"]), n/q), ctx+" N/Q")
        failed = r.get("status") == "generation_failed" or r.get("failed") == "True"
        correct = r["correct"] == "True"
        check(not failed or (not correct and n == 0), ctx+" failed outcome")
        v = (n, f, q, qp, peak, correct, failed, r["truncated"]=="True", r["dataset"])
        row[i] = v
        if study == STUDIES[0]:
            controls[model, arm, i] = v
        elif arm not in own:
            check(v == controls[model, arm, i], ctx+" differs from reused baseline")
    expected = {(m,p) for m in ("llada21","sdar") for p in shown}
    check(groups.keys() == expected, study+" missing request groups")
    results = []
    seen = set()
    summary_keys = SUMMARY[STUDIES.index(study)].split()
    for r in read(directory / "summary.csv"):
        key = r["model"], policy(r)
        check(key not in seen, str(key)+" duplicate summary")
        seen.add(key)
        rows = groups[key]
        check(rows.keys() == tasks.keys(), str(key)+" incomplete sample set")
        vs = list(rows.values())
        n, f, q, qp = [sum(v[j] for v in vs) for j in range(4)]
        values = [len(vs), sum(v[5] for v in vs), mean(v[0]/v[1] for v in vs),
                  n/q, n, f, q, qp, n/qp, sum(v[6] for v in vs), sum(v[7] for v in vs),
                  *[sum(v[5] for v in vs if v[8] == d) for d in DATASETS]]
        for k, x in zip(summary_keys, values):
            check(math.isclose(float(r[k]), x, rel_tol=1e-11), str(key)+" summary "+k)
        peak = max(v[4] for v in vs)
        if "maxQ" in r or "max_q" in r:
            check(int(r.get("maxQ",r.get("max_q"))) == peak, str(key)+" maxQ")
        results.append(dict(study=study, model=key[0], policy=key[1], max_Q=peak,
                            avg_Q=q/f, reused=key[1] not in own, **dict(zip(NAMES, values))))
    check(seen == expected, study+" incomplete summary")
    return results


def audit():
    manifest = json.loads((ROOT / "benchmarks/frozen30/manifest.json").read_text())
    tasks = {s["index"]: s for s in manifest["samples"]}
    check(set(tasks)==set(range(30)) and len({v["task_id"] for v in tasks.values()})==30,
          "Expected 30 distinct tasks")
    check(all(sum(t["dataset"]==d for t in tasks.values())==10 for d in DATASETS), "dataset mix")
    controls = {}
    result = [r for s in STUDIES for r in checked_table(s, tasks, controls)]
    check(len(result)==62 and sum(r["n"] for r in result if not r["reused"])==1380, "matrix size")
    for name in STUDIES:
        figures = ROOT / "archive" / name / "figures"
        original = json.loads((figures.parent / "provenance.json").read_text())["source_sha256"]
        expected = sorted(v for k,v in original.items() if k.endswith(".png"))
        check(sorted(hashlib.sha256(p.read_bytes()).hexdigest() for p in figures.glob("*.png")) == expected,
              name+" archived figure digests")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", action="store_true", help="show all reported rows across batch sizes")
    args = p.parse_args()
    rows = audit()
    if args.table:
        print("|Study|Model|Policy|Mean Q/F|Max Q|TPF|Quality|Token efficiency|")
        print("|---|---|---|---:|---:|---:|---:|---:|")
        for r in rows:
            print("|{study}|{model}|{policy}|{avg_Q:.2f}|{max_Q}|{TPF:.3f}|{correct}/30|{efficiency:.2%}|".format(**r))
    print("OK: 46 measured groups / 1,380 requests; 16 reused groups / 480 requests; 6 original figures.")
