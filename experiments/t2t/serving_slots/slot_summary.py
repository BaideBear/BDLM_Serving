#!/usr/bin/env python3
"""Compact long-format summary of measured frozen90 serving slots.

Inputs: --fixed-k0 and --controls (explicit allowlist below).  One row per
(endpoint, cohort); every value is read or arithmetic on published columns
(efficiency=N/physical_Q, q_per_f=physical_Q/F).  No gaps/inferred anchors/nominal-Q-as-work; Gemma Q16 excluded.
"""
from __future__ import annotations
import argparse, csv, math, re, sys
from pathlib import Path

COHORTS = ("original-0-29", "added-30-89", "all-0-89")
REQ = dict(zip(COHORTS, (30, 60, 90)))
ORIG, ADD, ALL = COHORTS
TOL = 1e-6
INTERNAL = {"k0_zero", "k0_slack", "native_fullwidth", "dual64"}
VORDER = ("k0_zero", "k0_slack", "native_fullwidth", "dual64", "ibs_original", "obs_original", "ibs_capped", "obs_capped", "flow_w1", "flow_w2")
FIELDS = "model endpoint variant family offer cohort requests correct N F physical_Q raw_Q q_per_f macro_tpf efficiency cap policy source".split()
DEFAULT = Path(__file__).resolve().parent
# Intentional, not a directory scan.
ALLOWLIST = (
    "llada21-q4-original-ibs.csv", "llada21-q4-original-obs.csv",
    "llada21-q8-original-ibs.csv", "llada21-q8-original-obs.csv",
    "llada21-q16-original-ibs.csv", "llada21-q16-original-obs.csv",
    "llada21-q32-original-ibs.csv", "llada21-q32-original-obs.csv",
    "gemma-ibs_32-zero.csv", "gemma-ibs_64-zero.csv", "gemma-ibs_128-zero.csv",
    "gemma-ibs_256-zero.csv", "gemma-obs_32-zero.csv", "gemma-obs_64-zero.csv",
    "gemma-obs_128-zero.csv", "gemma-obs_256-zero.csv",
    "gemma-original-vs-capped-fixed.csv", "gemma-wave-vs-fixed.csv",
    "llada21-native32-vs-flow-w1.csv", "llada21-dual64-vs-flow-w2.csv",
    "llada21-true64.csv",
)

class SummaryError(ValueError):
    pass

def _num(row, key, where, cast=int, required=True):
    value = row.get(key)
    if value is None or not str(value).strip():
        if required:
            raise SummaryError("%s: missing %s" % (where, key))
        return None
    try:
        out = cast(str(value).strip())
    except ValueError:
        raise SummaryError("%s: bad %s=%r" % (where, key, value))
    if cast is float and not math.isfinite(out):
        raise SummaryError("%s: non-finite %s" % (where, key))
    return out

def _close(a, b):
    return math.isclose(a, b, rel_tol=TOL, abs_tol=1e-12)

def variant(model, policy):
    """Return (variant, offer label, cap label) from a published policy."""
    p = policy.strip()
    if p == "fullwidth":
        return "native_fullwidth", 32, "native_full"
    if p == "dual64_noedit":
        return "dual64", 64, "dual_block"
    m = re.fullmatch(r"flow_w([12])", p)
    if m:
        w = int(m.group(1)); offer = ((256, 512) if model == "gemma" else (32, 64))[w - 1]
        return "flow_w%d" % w, offer, "flow_fixed_w%d" % w
    for rx, lab, cap in (
            (r"optimus_(ibs|obs)_c(\d+)", "original", "nominal_only"),
            (r"(ibs|obs)_cap_(\d+)", "capped", "hard_cap"),
            (r"(ibs|obs)_(\d+)", "original", "original")):
        m = re.fullmatch(rx, p)
        if m:
            return m.group(1) + "_" + lab, int(m.group(2)), cap
    m = re.fullmatch(r"t2t_q(\d+)_f\d+_m(\d+)_hard_prefix_(\w+)", p)
    if m:
        label = "k0_zero" if m.group(3) == "zero" else "k0_" + m.group(3)
        return label, int(m.group(1)), "sparse_nominal"
    m = re.fullmatch(r"slottrace_(\d+)", p)
    if m:
        return "k0_zero", int(m.group(1)), "sparse_nominal"
    raise SummaryError("unrecognized policy %r for model %r" % (policy, model))

def _groups(rows, keyfn):
    groups = {}
    for row in rows:
        key = keyfn(row)
        cohort = row["cohort"].strip()
        if cohort in groups.get(key, {}):
            raise SummaryError("duplicate cohort %s" % cohort)
        groups.setdefault(key, {})[cohort] = row
    return groups

def _metrics(row, prefix, cohort, where):
    req = _num(row, "requests", where)
    if req != REQ[cohort]:
        raise SummaryError("%s: requests %s != %s" % (where, req, REQ[cohort]))
    cor, n, f, qp = (_num(row, prefix + k, where)
                     for k in ("correct", "N", "F", "physical_Q"))
    raw = _num(row, prefix + "Q", where, required=False)
    tpf = _num(row, prefix + "macro_tpf", where, float); eff = _num(row, prefix + "pooled_token_efficiency", where, float)
    if not 0 <= cor <= req or f <= 0 or qp <= 0 or (
            raw is not None and not 0 <= raw <= qp):
        raise SummaryError("%s: invalid counts or F/Q" % where)
    if not _close(eff, n / qp):
        raise SummaryError("%s: efficiency != N/physical_Q" % where)
    qpf = qp / f; claimed = _num(row, prefix + "q_per_f", where, float, required=False)
    if claimed is not None and not _close(claimed, qpf):
        raise SummaryError("%s: q_per_f != physical_Q/F" % where)
    return dict(requests=req, correct=cor, N=n, F=f, physical_Q=qp,
                raw_Q=raw if raw is not None else "", q_per_f=qpf,
                macro_tpf=tpf, efficiency=eff)

def _emit(by, prefix, model, source, var=None, offer=None, cap=None,
          policy=""):
    if var is None:
        policy = by[ORIG][prefix + "policy"].strip()
        var, offer, cap = variant(model, policy)
    items = {c: _metrics(by[c], prefix, c, "%s/%s" % (source, c))
             for c in COHORTS}
    for key in ("correct", "N", "F", "physical_Q"):
        if items[ORIG][key] + items[ADD][key] != items[ALL][key]:
            raise SummaryError("%s: %s cohorts do not sum" % (source, key))
    if all(items[c]["raw_Q"] != "" for c in COHORTS) and (
            items[ORIG]["raw_Q"] + items[ADD]["raw_Q"]
            != items[ALL]["raw_Q"]):
        raise SummaryError("%s: raw_Q cohorts do not sum" % source)
    endpoint = "%s:%s:%s" % (model, var, offer); family = "internal" if var in INTERNAL else "external"
    return [dict(model=model, endpoint=endpoint, variant=var, family=family,
                 offer=offer, cohort=c, cap=cap, policy=policy, source=source,
                 **items[c]) for c in COHORTS]

def _fixed(path):
    path = Path(path)
    if not path.is_file():
        raise SummaryError("missing fixed-k0.csv: %s" % path)
    with path.open(newline="", encoding="utf-8") as fh: rows = list(csv.DictReader(fh))
    groups = _groups(rows, lambda r: (r["model"].strip(), _num(r, "slot", "fixed-k0 slot")))
    out = []
    for (model, slot), by in sorted(groups.items()):
        if (model, slot) == ("gemma", 16):
            continue
        if set(by) != set(COHORTS):
            raise SummaryError("fixed-k0 %s/%s: mixed cohorts" % (model, slot))
        out += _emit(by, "", model, "fixed-k0.csv", var="k0_zero",
                     offer=slot, cap="sparse_nominal")
    return out

def _control(path):
    path = Path(path)
    if not path.is_file():
        raise SummaryError("missing allowlisted control: %s" % path.name)
    with path.open(newline="", encoding="utf-8") as fh: rows = [r for r in csv.DictReader(fh) if (r.get("scope") or "all") == "all"]
    groups = _groups(rows, lambda r: (r["model"].strip(), r["arm_policy"].strip(), r["baseline_policy"].strip()))
    out = []
    for (model, _arm, _base), by in groups.items():
        if set(by) != set(COHORTS):
            raise SummaryError("%s: mixed cohorts %s" % (path.name, sorted(by)))
        out += _emit(by, "arm_", model, "controls/%s#arm" % path.name)
        out += _emit(by, "baseline_", model,
                     "controls/%s#baseline" % path.name)
    return out

def _consolidate(items):
    seen = {}
    for row in items:
        if row["model"] == "gemma" and row["offer"] == 16:
            continue
        key = (row["endpoint"], row["cohort"])
        if key not in seen:
            seen[key] = dict(row)
            continue
        old = seen[key]
        for f in ("requests", "correct", "N", "F", "physical_Q"):
            if old[f] != row[f]:
                raise SummaryError("inconsistent %s: %s vs %s" %
                                   (key, old["source"], row["source"]))
        if not _close(old["macro_tpf"], row["macro_tpf"]): raise SummaryError("inconsistent macro_tpf for %s" % (key,))
        if old["raw_Q"] and row["raw_Q"] and old["raw_Q"] != row["raw_Q"]: raise SummaryError("inconsistent raw_Q for %s" % (key,))
        for f in ("source", "policy", "cap"):
            old[f] = " | ".join(dict.fromkeys(
                v for v in (old[f], row[f]) if v))
        if not old["raw_Q"]:
            old["raw_Q"] = row["raw_Q"]
    return list(seen.values())

def summarize(fixed_path, controls, allowlist=None):
    items = _fixed(fixed_path)
    for name in (ALLOWLIST if allowlist is None else allowlist):
        items += _control(Path(controls) / name)
    order = {v: i for i, v in enumerate(VORDER)}; cohort = {c: i for i, c in enumerate(COHORTS)}
    return sorted(_consolidate(items), key=lambda r: (
        r["model"], order.get(r["variant"], 99), r["offer"],
        cohort[r["cohort"]]))

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-k0", type=Path, default=DEFAULT / "fixed-k0.csv")
    parser.add_argument("--controls", type=Path, default=DEFAULT / "controls")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        rows = summarize(args.fixed_k0, args.controls)
    except SummaryError as exc:
        print("slot_summary: FAIL-CLOSED: %s" % exc, file=sys.stderr)
        return 2
    stream = args.out.open("w", newline="", encoding="utf-8") if args.out else sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=FIELDS,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.out:
            stream.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
