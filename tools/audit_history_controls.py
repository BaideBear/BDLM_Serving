"""Reconcile frozen90 native, uncapped IBS and W1/W2 control archives.

Only request metadata and aggregate scores leave the cluster. Archived accuracy
is read from the registered scorer table; N/F/Q and request identities are
recomputed from individual GPU records, including failed/truncated requests.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
SLOTS = ROOT / "experiments/t2t/serving_slots"
COHORTS = [("original-0-29", 0, 30), ("added-30-89", 30, 90), ("all-0-89", 0, 90)]


def read(path):
    return json.loads(path.read_bytes())


def table(path, **match):
    with path.open() as src:
        rows = [r for r in csv.DictReader(src) if all(r[k] == v for k, v in match.items())]
    if len(rows) != 1:
        raise ValueError(f"{path.name}: expected exactly one row for {match}, got {len(rows)}")
    return rows[0]


def controls(base):
    data = base / "prefix90"
    for model, width in (("llada21", 32), ("gemma", 256)):
        yield model, "native", width, f"native{width}", base / "frozen90-native" / model
        for w in (1, 2):
            name = "llada21-reference90-wave" if model == "llada21" else "gemma-wave-reference90"
            yield model, "flow", w, f"flow_w{w}", data / name
        for q in (width//8, width//4, width//2, width):
            if model == "llada21":
                name, policy = f"llada21-reference90-q{q}", f"optimus_ibs_c{q}"
            else:
                name = "gemma-optimus-reference90" if q in (32, 128) else "gemma-optimus-primary90"
                policy = f"ibs_{q}"
            yield model, "ibs", q, policy, data / name


def published(model, method, setting, policy, cohort):
    if method == "native":
        r = table(SLOTS.parent / "prefix_window/native90.csv", model=model, policy=policy,
                  cohort=cohort, dataset="all")
        keys = ["correct","N","F","Q","macro_tpf","token_efficiency"]
    elif method == "flow":
        r = table(SLOTS / "controls/fixed-family-references.csv", model=model, method=policy, cohort=cohort)
        keys = ["correct","N","F","Q","macro_TPF","efficiency"]
    elif model == "gemma":
        r = table(SLOTS / "gemma-optimus.csv", q=str(setting), method="IBS", cohort=cohort)
        keys = ["correct","N","F","physical_Q","macro_tpf","pooled_token_efficiency"]
    else:
        r = table(SLOTS / f"controls/llada21-q{setting}-original-ibs.csv", baseline_policy=policy,
                  cohort=cohort, scope="all")
        keys = ["baseline_"+k for k in ("correct","N","F","physical_Q","macro_tpf","pooled_token_efficiency")]
    return [float(r[k]) for k in keys]


def audit(base):
    manifest = read(ROOT / "benchmarks/frozen90/manifest.json")
    samples = base / "prefix90/samples.json"
    if hashlib.sha256(samples.read_bytes()).hexdigest() != manifest["samples_sha256"]:
        raise ValueError("different frozen90 input")
    results = []
    for model, method, setting, policy, group in controls(base):
        if method != "native":
            meta = read(group / "run.json")
            receipt = read(group / f"complete-{policy}.json")
            if (meta["samples_sha256"] != manifest["samples_sha256"] or policy not in meta["policies"]
                    or receipt["requests"] != 90 or receipt["policy"] != policy
                    or (method == "ibs" and meta["hard_q_cap"] is not False)):
                raise ValueError(f"unexpected runner identity {group} {policy}")
        tape = []
        for i, task in enumerate(manifest["samples"]):
            row = read(group / f"{i:02d}-{policy}.json")
            if row["policy"] != policy or any(row[k] != task[k] for k in ("index","dataset","task_id","seed")):
                raise ValueError(f"mispaired record {group} {policy} {i}")
            n, f, q = (row[k] for k in ("output_tokens","generation_forwards","generation_physical_padded_rows"))
            if n < 0 or f <= 0 or q <= 0:
                raise ValueError(f"invalid measured work {group} {policy} {i}")
            tape.append((n,f,q,row["status"] != "success",bool(row["truncated"])))
        for cohort, start, end in COHORTS:
            subset = tape[start:end]
            n,f,q,failed,truncated = map(sum,zip(*subset))
            tpf,eff = fmean(x[0]/x[1] for x in subset),n/q
            correct,*reference = published(model,method,setting,policy,cohort)
            if not all(math.isclose(x,y,rel_tol=1e-8) for x,y in zip((n,f,q,tpf,eff),reference)):
                raise ValueError(f"raw GPU records do not reproduce scorer table: {model}/{policy}/{cohort}")
            results.append(dict(model=model,method=method,setting=setting,policy=policy,cohort=cohort,
                                requests=len(subset),correct=int(correct),macro_tpf=tpf,token_efficiency=eff,
                                N=n,F=f,physical_Q=q,failed=failed,truncated=truncated,source=group.relative_to(base)))
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = audit(args.archive)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)//len(COHORTS)} configurations reconciled on all 90 requests", flush=True)
