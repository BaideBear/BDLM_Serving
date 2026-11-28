"""Audit live formal90 shards; publish scores only for individually sealed points."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from experiments.t2t.serving_slots.shared_fixed_shard import identity
from experiments.t2t.serving_slots.shared_point_metrics import _receipt, summarize
from tools.frozen90_six_shards import BOARDS, MANIFEST, OFFERS, split


def source_paths(root, board, model):
    folder = root / board.replace("n164:", "").replace("A800:", "A")
    return [p for p in (folder / (model + ".ndjson"), folder / (model + ".run.jsonl"),
                        folder / model / "receipts.ndjson") if p.is_file()]


def costs(row):
    roles = row["roles"]
    if any(type(v) is not int or v < 0 for v in roles.values()):
        raise ValueError("non-integer or negative Q/F role counter")
    if row["model"] == "llada21":
        dirty, cold, warm, prefix = (roles[k] for k in
                                     ("dirty_write", "uninitialized_mask", "resident_mask", "clean_resolved"))
        verify = rebuild = 0
    else:
        dirty, mask, cold, prefix, verify, rebuild = (roles.get(k + "_Q", 0) for k in
                                                       ("dirty", "mask", "cold", "prefix", "verify", "rebuild"))
        warm = mask - cold
        routes = ("sample", "drain", "verify", "rebuild")
        if (warm < 0 or sum(roles.get(k + "_F", 0) for k in routes) != row["decoder_F"]
                or roles["peak_Q"] != row["max_decoder_Q"]
                or roles.get("sample_Q", 0) + roles.get("drain_Q", 0) != dirty + mask + prefix):
            raise ValueError("Gemma decoder charge or row split disagrees with the meter")
    result = dict(dirty=dirty, cold_mask=cold, warm_mask=warm, prefix=prefix,
                  terminal_verification=verify, rebuild=rebuild)
    if sum(result.values()) != row["decoder_Q"]:
        raise ValueError("Q role partition does not sum to the physical decoder Q")
    return result


def audit(root, samples, registration):
    if registration["stage"] != "formal90":
        raise ValueError("not a formal90 registration")
    version = identity(registration)
    seen = {}
    deferred = []
    for board, shard in split(samples).items():
        want = {r["index"]: r for r in shard}
        for model, offers in OFFERS.items():
            paths = source_paths(root, board, model)
            if len(paths) > 1:
                raise ValueError(f"ambiguous ledger for {board}/{model}: {paths}")
            for path in paths:
                with path.open("rb") as stream:
                    for n, line in enumerate(stream, 1):
                        if not line.endswith(b"\n"):
                            deferred.append(f"{board}/{model}:line{n}")
                            break
                        try:
                            row = json.loads(line)
                            idx, q = row["index"], row["offer"]
                            if (type(idx) is not int or idx not in want or type(q) is not int or q not in offers
                                    or (row["board"], row["model"], row["version"], row["stage"])
                                    != (board, model, version, "formal90")
                                    or any(row[k] != want[idx][k] for k in ("dataset", "task_id", "seed", "prompt_sha256"))
                                    or (model, q, idx) in seen):
                                raise ValueError("unexpected, duplicate or mismatched request")
                            if not {"decoder_Q", "max_decoder_Q"} <= row.keys():
                                raise ValueError("missing physical bill")
                            _receipt(row, model, q)
                            costs(row)
                            seen[model, q, idx] = row
                        except (KeyError, TypeError, ValueError) as exc:
                            raise ValueError(f"{path}:{n}: {exc}") from exc
    points = []
    boards = Counter(r["board"] for r in seen.values())
    for model, offers in OFFERS.items():
        for q in offers:
            rows = [r for (m, offer, _), r in seen.items() if (m, offer) == (model, q)]
            p = dict(model=model, offer=q, verified=len(rows), target=90)
            if len(rows) == 90:
                summary = summarize(samples, rows, model, q, version)
                p["cohorts"] = summary["cohorts"]
                charge = Counter()
                for r in rows:
                    charge.update(costs(r))
                p["physical_Q_by_role"] = dict(charge)
            points.append(p)
    return dict(version=version, seen=len(seen), total=900, deferred_incomplete_lines=deferred,
                boards={b: boards[b] for b in BOARDS}, points=points)


def comparisons(report, baseline_csv):
    by_key = defaultdict(list)
    with baseline_csv.open(newline="") as source:
        for r in csv.DictReader(source):
            if r["method"] in ("Live profile", "IBS capped", "IBS nominal", "Native", "Flow W1", "Flow W2"):
                by_key[r["model"], int(r["budget"]), r["cohort"]].append(r)
    for p in report["points"]:
        if "cohorts" not in p:
            continue
        p["comparisons"] = {}
        for c, result in p["cohorts"].items():
            key = p["model"], p["offer"], "frozen90" if c == "all90" else c
            matched = by_key[key]
            if not matched or len({r["method"] for r in matched}) != len(matched):
                raise ValueError(f"missing or duplicate baseline group: {key}")
            tier = OFFERS[p["model"]]
            required = {"Live profile", "IBS nominal", "Flow W2" if p["offer"] == tier[-1] else "IBS capped"}
            if p["offer"] == tier[-2]:
                required |= {"Native", "Flow W1"}
            if not required <= {r["method"] for r in matched}:
                raise ValueError(f"missing reference for {key}")
            diffs = {}
            for r in matched:
                if int(r["n"]) != result["requests"]:
                    raise ValueError(f"baseline cohort mismatch: {key}, {r['method']}")
                diffs[r["method"]] = dict(correct=result["correct"] - int(r["correct"]),
                                           macro_TPF=result["macro_TPF"] - float(r["macro_TPF"]),
                                           TE_percentage_points=(100 * (result["pooled_TE"] - float(r["token_efficiency"]))
                                                                 if r["token_efficiency"] else None))
            p["comparisons"][c] = diffs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("registration", type=Path)
    parser.add_argument("--baselines", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = audit(args.root, json.loads(MANIFEST.read_bytes())["samples"],
                   json.loads(args.registration.read_bytes()))
    if args.require_complete and (report["seen"] != 900 or report["deferred_incomplete_lines"]):
        parser.error("refusing to publish incomplete ledger")
    if args.baselines:
        comparisons(report, args.baselines)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
