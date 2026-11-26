"""Join published prefix arms to historical frozen30 policies, without interpolation.

Run as a module with --private pointing at score_run's private sidecar directory.
Only aggregate pair comparisons leave the cluster. All Q costs refer to generation.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from tools.work_envelope import compare

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def identity(r):
    return int(r["index"]), r["dataset"], r["task_id"]


def pack30(rows, ids, prefix="", historical=False):
    packed, logical = {}, 0
    for r in rows:
        key = identity(r)
        if key not in ids or key in packed:
            raise ValueError(f"Missing identity or duplicate: {key}")
        if not historical and int(r["seed"]) != ids[key]:
            raise ValueError(f"Seed mismatch: {key}")
        n, f, q, c = (int(r[prefix + field]) for field in ("N", "F", "Q", "correct"))
        physical = int(r["physical_Q"]) if historical else q
        if not (physical >= q >= f > 0 and n >= 0 and c in (0, 1)):
            raise ValueError(f"Invalid work or score: {key}")
        packed[key] = n, f, physical, c
        logical += q
    if len(ids) != 30 or packed.keys() != ids.keys():
        raise ValueError("Comparison is not the frozen 30 requests")
    return packed, logical


def summary_sources(folder, private):
    sources = {p.name: p for p in folder.glob("*-q*.csv")}
    # Large Gemma sweeps are published as one aggregate table. Resolve only
    # its published arms, not arbitrary in-progress files from the private run.
    listing = folder / "gemma-ratio-online.csv"
    if listing.exists():
        for row in read(listing):
            if row["cohort"] != "original30":
                continue
            q, k, selector = int(row["q"]), int(row["k"]), row["selector"]
            if selector not in ("right", "random") or not 0 < k < q:
                raise ValueError("Unknown published ratio arm")
            name = f"gemma-q{q}-slack-k{k}-{selector}.csv"
            path = sources.setdefault(name, private / name)
            original, = [s for s in read(path) if s["scope"] == "all" and s["window"] == "30"]
            if any(original[a] != row[b] for a, b in
                   (("arm_correct", "correct"), ("arm_physical_Q", "full_generation_q"),
                    ("arm_macro_tpf", "tpf"), ("baseline_correct", "zero_correct"))):
                raise ValueError(f"Ratio listing disagrees with paired score: {name}")
    return sorted(sources.values(), key=lambda p: p.name)


def pairs(private):
    old, new = (json.loads((ROOT / f"benchmarks/frozen{n}/manifest.json").read_text()) for n in (30, 90))
    if (old["samples"] != new["samples"][:30] or old["samples_sha256"] != new["original_samples_sha256"]
            or old["scorer_version"] != new["scorer_version"]):
        raise ValueError("Historical cohort or scorer mismatch")
    ids = {identity(r): r["seed"] for r in old["samples"]}
    archive = defaultdict(list)
    for r in read(ROOT / "experiments/baselines/requests.csv"):
        if r["model"] in ("gemma", "llada21"):
            archive[r["model"], r["policy"], r["role"], r["run"]].append(r)
    refs = {k: pack30(rs, ids, historical=True) for k, rs in archive.items()}
    arms = {}
    for path in summary_sources(HERE, private):
        rows = [r for r in read(path) if r.get("scope") == "all" and r["window"] == "30"]
        if not rows:
            continue
        if len(rows) != 1:
            raise ValueError(f"Ambiguous original30: {path}")
        s = rows[0]
        detail = [r for r in read(private / (path.stem + "-paired.csv"))
                  if r["cohort"] == "original-0-29" and r["window"] == "30"]
        for side in ("arm", "baseline"):
            pol = s[side + "_policy"]
            if any((r["model"], r[side + "_policy"], r["nominal_q"]) !=
                   (s["model"], pol, s["nominal_q"]) for r in detail):
                raise ValueError(f"Sidecar provenance mismatch: {path}")
            data, q = pack30(detail, ids, side + "_")
            nsum, fsum, physical, correct = map(sum, zip(*data.values()))
            if (nsum, fsum, q, physical, correct) != tuple(
                    int(s[side + "_" + k]) for k in ("N", "F", "Q", "physical_Q", "correct")):
                raise ValueError(f"Published aggregate disagrees with sidecar: {path}")
            key = s["model"], int(s["nominal_q"]), pol
            if arms.setdefault(key, (data, q)) != (data, q):
                raise ValueError(f"Inconsistent zero or duplicate arm: {key}")
    result = []
    for (model, q, policy), (data, logical) in sorted(arms.items()):
        for (ref_model, ref_policy, role, run), (ref, ref_logical) in sorted(refs.items()):
            if ref_model != model:
                continue
            opt = re.fullmatch(r"(ibs|obs)_(\d+)", ref_policy)
            if opt and int(opt[2]) != q:
                continue
            if not opt and not (ref_policy in ("b3", "native") or re.fullmatch(r"flow_w[1-4]", ref_policy)):
                continue
            m = compare(data, ref)
            result.append(dict(model=model, nominal_q=q, prefix_policy=policy, ref_policy=ref_policy,
                ref_role=role, ref_run=run, comparison="same_nominal_window" if opt else "work_reference",
                cohort="original30", requests=len(data), **m, logical_Q=logical,
                ref_logical_Q=ref_logical, ref_logical_efficiency=m["ref_N"]/ref_logical,
                ref_logical_q_per_f=ref_logical/m["ref_F"]))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--private", required=True, type=Path)
    p.add_argument("--out", type=Path, default=HERE / "historical30.csv")
    args = p.parse_args()
    rows = pairs(args.private)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
