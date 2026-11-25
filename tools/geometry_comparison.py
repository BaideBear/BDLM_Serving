"""Compare published LLaDA B32 geometry-v1 and exact-budget archive arms on CPU."""

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import json
from math import isclose
from pathlib import Path
from statistics import mean
from tempfile import NamedTemporaryFile

from bdlm_serving.evaluation.frozen import MANIFEST


MODEL = "llada21"
BASE = Path("experiments/t2t")
PLACEMENT = BASE / "placement"
DEFAULTS = dict(geometry=PLACEMENT / "geometry-online", adaptive=BASE / "adaptive",
                hard=BASE / "hard", placebo=PLACEMENT)
COUNTS = ("correct", "failed", "truncated", "N", "F", "Q", "physical_Q")
RATES = ("accuracy", "tpf", "token_efficiency", "q_per_f")
METRICS = ("requests", *COUNTS, *RATES, "max_q")
RESERVES = ("reserved_status", "reserved_source", "reserved_seats",
            "reserved_hits", "reserved_hit_rate")
ARM_COLUMNS = ("source", "policy", "nominal_q", "edit", "status", "detail", "request_rows",
               "summary_present", *METRICS, *RESERVES)


@dataclass(frozen=True)
class Arm:
    source: str
    q: int
    edit: int
    selection: str

    @property
    def mask(self):
        return self.q - self.edit

    @property
    def policy(self):
        return f"t2t_q{self.q}_f0_m{self.mask}_{self.selection}"


def comparisons():
    for q, edit, geometry in ((4, 1, "hard_slack_geometry"), (8, 1, "hard_slack_geometry"),
                              (16, 2, "hard_slack_geometry"), (16, 2, "hard_geometry")):
        candidate = Arm("geometry", q, edit, geometry)
        if geometry == "hard_slack_geometry":
            yield candidate, Arm("adaptive", q, edit, "hard_slack")
            # Q8 placebo was scored with the geometry run, unlike Q4/Q16.
            yield candidate, Arm("geometry" if q == 8 else "placebo",
                                 q, edit, "hard_slack_placebo")
        else:
            yield candidate, Arm("hard", q, edit, "hard_frontier")
            yield candidate, Arm("hard", q, edit, "hard_random")
        yield candidate, Arm("hard", q, 0, "hard_noedit")


def _table(path, *, aggregate=False):
    found = defaultdict(list)
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["model"] == MODEL and (not aggregate or row["dataset"] == "all"):
                    found[row["policy"]].append(row)
    return found


def _single(rows, policy):
    matching = rows.get(policy, ())
    if len(matching) > 1:
        raise ValueError(f"{policy}: duplicate aggregate rows")
    return matching[0] if matching else None


def _check_config(row, arm, q_col="nominal_q"):
    columns = q_col, "fresh", "mask", "edit"
    if tuple(int(row[c]) for c in columns) != (arm.q, 0, arm.mask, arm.edit):
        raise ValueError(f"{arm.policy}: budget columns disagree with policy")


def _paired_cost_reserves(row, metrics, arm):
    keys = ("requests", "N", "F", "Q")
    if int(row["nominal_q"]) != arm.q or any(int(row[k]) != metrics[k] for k in keys):
        raise ValueError(f"{arm.policy}: paired-cost work does not match published requests")
    return row["reserved_seats"], row["reserved_strict_hits"]


def _reserves(audit, paired_cost, arm, metrics, directory):
    if audit is not None:
        _check_config(audit, arm, "budget")
        name = directory / f"{MODEL}-audit.csv"
        if int(audit["requests"]) != metrics["requests"] or audit["ok"] != "1":
            return dict(reserved_status="unverified_audit", reserved_source=str(name))
        stages = ("bootstrap", "selected", "verify", "commit", "reencode", "dense")
        for counter in ("F", "Q"):
            keys = [f"{stage}_{counter.lower()}" for stage in stages]
            if counter == "F":
                keys[1] = "selected_steps"
            if sum(int(audit[k] or 0) for k in keys) != metrics[counter]:
                raise ValueError(f"{arm.policy}: audit {counter} does not match published requests")
        seats, hits = audit["reserved_seats"], audit["reserved_hits"]
    elif paired_cost is not None:
        name = directory / f"{MODEL}-paired-cost.csv"
        seats, hits = _paired_cost_reserves(paired_cost, metrics, arm)
    else:
        return dict(reserved_status="missing_evidence")
    if seats == "" or hits == "":
        return dict(reserved_status="missing_counters", reserved_source=str(name))
    seats, hits = int(seats), int(hits)
    if not 0 <= hits <= seats or (arm.edit == 0 and seats != 0):
        raise ValueError(f"{arm.policy}: impossible reservation counters")
    return dict(reserved_status="ok", reserved_source=str(name), reserved_seats=seats,
                reserved_hits=hits, reserved_hit_rate=hits / seats if seats else None)


def _load_arm(arm, directory, tables, manifest):
    requests, summaries, audits, paired_costs = tables
    source = str(directory)
    summary = _single(summaries, arm.policy)
    indexed = {}
    for row in requests.get(arm.policy, ()):
        _check_config(row, arm)
        index = int(row["index"])
        if manifest.get(index) != (row["dataset"], row["task_id"]):
            raise ValueError(f"{arm.policy}: request {index} does not match frozen manifest")
        if index in indexed:
            raise ValueError(f"{arm.policy}: duplicate request {index}")
        numeric = {k: int(row[k]) for k in (*COUNTS, "max_q")}
        if (any(numeric[k] not in (0, 1) for k in COUNTS[:3])
                or numeric["correct"] + numeric["failed"] > 1
                or not 0 < numeric["F"] <= numeric["Q"] == numeric["physical_Q"]
                or numeric["N"] < 0 or not 1 <= numeric["max_q"] <= 32):
            raise ValueError(f"{arm.policy}: invalid work or score for request {index}")
        indexed[index] = numeric

    info = dict(source=source, policy=arm.policy, nominal_q=arm.q, edit=arm.edit,
                request_rows=len(indexed), summary_present=int(summary is not None))
    if summary is not None:
        _check_config(summary, arm)
    if len(indexed) != len(manifest) or summary is None:
        reason = []
        if len(indexed) != len(manifest):
            reason.append(f"requests {len(indexed)}/{len(manifest)}")
        if summary is None:
            reason.append("missing all-summary")
        info.update(status="incomplete" if indexed or summary is not None else "missing",
                    detail="; ".join(reason), reserved_status="arm_unavailable")
        return info, indexed

    rows = list(indexed.values())
    metrics = {k: sum(r[k] for r in rows) for k in COUNTS}
    metrics.update(requests=len(rows), max_q=max(r["max_q"] for r in rows),
                   accuracy=metrics["correct"] / len(rows),
                   tpf=mean(r["N"] / r["F"] for r in rows),
                   token_efficiency=metrics["N"] / metrics["Q"],
                   q_per_f=metrics["Q"] / metrics["F"])
    for k, value in metrics.items():
        if k in RATES:
            if k != "accuracy" and not isclose(float(summary[k]), value, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"{arm.policy}: summary {k} does not match requests")
        elif int(summary[k]) != value:
            raise ValueError(f"{arm.policy}: summary {k} does not match requests")
    info.update(status="ok", **metrics,
                **_reserves(_single(audits, arm.policy), _single(paired_costs, arm.policy),
                            arm, metrics, directory))
    return info, indexed


def _compare(candidate, reference):
    a, ax = candidate
    b, bx = reference
    missing = [side for side, info in (("geometry", a), ("reference", b))
               if info["status"] != "ok"]
    out = dict(nominal_q=a["nominal_q"], status="missing_" + "_and_".join(missing) if missing else "ok")
    side_cols = ("policy", "source", "edit", "status", "detail", "request_rows", *METRICS, *RESERVES)
    for side, info in (("geometry", a), ("reference", b)):
        out.update({f"{side}_{k}": info.get(k) for k in side_cols})
    stats = ("paired_requests", "wins", "losses", "both_correct", "neither_correct",
             *(f"delta_{k}" for k in (*RATES, "correct", "F", "Q", "physical_Q")))
    out.update(dict.fromkeys(stats))
    if not missing:
        outcomes = [(ax[i]["correct"], bx[i]["correct"]) for i in ax]
        out.update(paired_requests=len(outcomes), wins=outcomes.count((1, 0)),
                   losses=outcomes.count((0, 1)), both_correct=outcomes.count((1, 1)),
                   neither_correct=outcomes.count((0, 0)))
        for k in (*RATES, "correct", "F", "Q", "physical_Q"):
            out[f"delta_{k}"] = a[k] - b[k]
    return out


def build(directories, manifest):
    tables = {}
    for label, directory in directories.items():
        tables[label] = (_table(directory / f"{MODEL}-online-requests.csv"),
                         _table(directory / f"{MODEL}-online-summary.csv", aggregate=True),
                         _table(directory / f"{MODEL}-audit.csv"),
                         _table(directory / f"{MODEL}-paired-cost.csv"))
    pairs = list(comparisons())
    arms = {}
    for arm in dict.fromkeys(arm for pair in pairs for arm in pair):
        arms[arm] = _load_arm(arm, directories[arm.source], tables[arm.source], manifest)
    return [info for info, _ in arms.values()], [_compare(arms[a], arms[b]) for a, b in pairs]


def _write(path, rows, fields):
    with NamedTemporaryFile(mode="w", dir=path.parent, newline="", encoding="utf-8",
                            delete=False) as f:
        tmp = Path(f.name)
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for source, default in DEFAULTS.items():
        p.add_argument(f"--{source}-dir", type=Path, default=default)
    p.add_argument("--out", type=Path, default=DEFAULTS["geometry"],
                   help="Directory for arms.csv and paired.csv (default: geometry scoring directory)")
    args = p.parse_args(argv)
    samples = json.loads(MANIFEST.read_text())["samples"]
    manifest = {r["index"]: (r["dataset"], r["task_id"]) for r in samples}
    if set(manifest) != set(range(30)) or len(samples) != 30:
        raise ValueError("Expected frozen30 manifest with unique indices 0 through 29")
    arms, pairs = build({k: getattr(args, f"{k}_dir") for k in DEFAULTS}, manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    _write(args.out / "arms.csv", arms, ARM_COLUMNS)
    _write(args.out / "paired.csv", pairs, list(pairs[0]))
    ready = sum(r["status"] == "ok" for r in pairs)
    print(f"{ready}/{len(pairs)} complete pairs -> {args.out}")


if __name__ == "__main__":
    main()
