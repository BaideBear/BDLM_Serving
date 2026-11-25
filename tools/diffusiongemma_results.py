"""Export allowlisted native metrics and reconstruct all four summary rows."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/diffusiongemma-native"
DATASETS = ("gsm8k", "math500", "humaneval")
COUNTS = ("prefill_rows", "output_tokens", "generation_forwards", "generation_query_rows",
          "generation_physical_padded_rows", "decoder_forwards", "decoder_query_rows",
          "reencode_forwards", "reencode_query_rows", "native_nonpad")
FIELDS = ("index", "dataset", "task_id", "seed", "correct", "truncated", *COUNTS,
          "tpf", "native_tpf", "token_efficiency")


def check(ok, msg):
    if not ok:
        raise ValueError(msg)


def expected():
    return json.loads((ROOT / "benchmarks/frozen30/manifest.json").read_text())["samples"]


def boolean(x):
    check(str(x) in ("True", "False"), "Invalid boolean")
    return str(x) == "True"


def export(run):
    with (run / "scores.csv").open() as f:
        scores = list(csv.DictReader(f))
    plan = json.loads((STUDY / "config.json").read_text())
    metadata = json.loads((run / "run.json").read_text())
    check(all(metadata[k] == v for k, v in plan.items()), "Run protocol mismatch")
    rows = []
    for t, s in zip(expected(), scores, strict=True):
        r = json.loads((run / f"{t['index']:02d}-native256.json").read_text())
        check(all(r[k] == t[k] for k in ("index", "dataset", "task_id", "seed")), "Wrong request")
        check(all(s[k] == str(r[k]) for k in ("index", "dataset", "task_id", "status", "policy",
                  "model", "output_tokens", "generation_forwards", "generation_query_rows",
                  "generation_physical_padded_rows")), "Scores and generation do not match")
        check(r["status"] == "success" and r["policy"] == "native256" and r["model"] == plan["model"],
              "Unexpected request state")
        r["correct"] = boolean(s["correct"])
        rows.append({k: r[k] for k in FIELDS})
    return rows


def summarize(rows):
    grouped = {k: [] for k in ("all", *DATASETS)}
    for t, r in zip(expected(), rows, strict=True):
        check(all(str(r[k]) == str(t[k]) for k in ("index", "dataset", "task_id", "seed")),
              "Identity/order mismatch")
        c = {k: int(r[k]) for k in COUNTS}
        n, f, q = (c[k] for k in ("output_tokens", "generation_forwards", "generation_query_rows"))
        d, e = c["decoder_forwards"], c["reencode_forwards"]
        check(0 < c["prefill_rows"] and 0 <= n <= c["native_nonpad"] <= 256 * (1 + e) <= 2048,
              "Invalid delivered count")
        check(0 <= e and 1 + e <= d <= 48 * (1 + e) and f == d + e, "Bad forward accounting")
        check(c["decoder_query_rows"] == 256*d and c["reencode_query_rows"] == 256*e
              and q == 256*f == c["generation_physical_padded_rows"], "Bad Q accounting")
        for k, ref in (("tpf", n/f), ("token_efficiency", n/q), ("native_tpf", c["native_nonpad"]/d)):
            check(math.isclose(float(r[k]), ref, rel_tol=1e-6), f"Inconsistent {k}")
        metric = dict(c, correct=boolean(r["correct"]), truncated=boolean(r["truncated"]),
                      tpf=n/f, native_tpf=float(r["native_tpf"]))
        for key in ("all", t["dataset"]):
            grouped[key].append(metric)
    output = []
    for dataset, group in grouped.items():
        totals = {k: sum(r[k] for r in group) for k in (*COUNTS, "correct", "truncated")}
        check(len(group) == (30 if dataset == "all" else 10), "Incomplete dataset")
        output.append(dict(dataset=dataset, requests=len(group), **totals,
            mean_tpf=mean(r["tpf"] for r in group), mean_native_tpf=mean(r["native_tpf"] for r in group),
            token_efficiency=totals["output_tokens"]/totals["generation_query_rows"]))
    return output


def write(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys(), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", type=Path, metavar="PRIVATE_SCORED_RUN")
    p.add_argument("--folder", type=Path, default=STUDY)
    args = p.parse_args()
    if args.export:
        rows = export(args.export)
        summary = summarize(rows)
        args.folder.mkdir(exist_ok=True, parents=True)
        write(args.folder / "requests.csv", rows)
        write(args.folder / "summary.csv", summary)
    else:
        with (args.folder / "requests.csv").open() as f:
            reader = csv.DictReader(f)
            check(tuple(reader.fieldnames) == FIELDS, "Unexpected publication fields")
            summary = summarize(list(reader))
        with (args.folder / "summary.csv").open() as f:
            check(list(csv.DictReader(f)) == [{k: str(v) for k, v in r.items()} for r in summary],
                  "Published summary differs from per-request reconstruction")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
