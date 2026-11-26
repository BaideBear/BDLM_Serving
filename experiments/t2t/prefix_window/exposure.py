"""Audit prefix exposure against each event's own endogenous zero plan.

Executable Q is ``prefix_plan.budget``; actual physical rows (Gemma ``q``, LLaDA
``actual_q``) must equal ``actual_width``.  ``slack`` is unused fixed-Q seats in
the zero plan, ``pool`` is ``eligible_prefix_rows``, and opportunity is
``min(K, slack, pool)`` summed for K=1/4.  Fresh-only overflow is separate.  No
cross-run row-signature identity is inferred.  Supported: LLaDA21 B32 and Gemma
B256; unknown models, widths, policies, missing fields, or non-endogenous plans
fail closed.  Aggregate CSV only: one row per (policy, cohort), 25 columns.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re
from collections import Counter
from pathlib import Path

POLICY = re.compile(r"^t2t_q(\d+)_f(\d+)_m(\d+)_(.+)$")
REQUEST = re.compile(r"^(\d{2})-(.+)\.json$")
COHORTS = (("original30", range(30)), ("added60", range(30, 90)),
           ("all90", range(90)))
DATASETS = ("gsm8k", "math500", "humaneval")
MODES = frozenset(("hard_prefix_zero", "hard_prefix_frontier_right",
                   "hard_prefix_random", "hard_prefix_slack_right",
                   "hard_prefix_slack_random", "hard_prefix_slack_oldest"))
FIELDS = ("model", "policy", "kind", "cohort", "requests", "events",
          "mean_q_rows", "positive_slack_events", "positive_slack_requests",
          "slack_rows", "fresh_overflow_events", "fresh_overflow_rows",
          "eligible_events", "eligible_rows", "opp_k1_requests", "opp_k1_rows",
          "opp_k4_requests", "opp_k4_rows", "nominal_k_rows", "added_rows",
          "mask_displaced", "ordinary_displaced", "noop_events",
          "saturated_events", "untouched_requests")


def need(ok, label, message):
    if not ok:
        raise ValueError("%s: %s" % (label, message))


def parse_policy(name):
    m = POLICY.match(name or "")
    return None if m is None else dict(budget=int(m.group(1)),
        fresh=int(m.group(2)), mask=int(m.group(3)), mode=m.group(4))


def schema(model):
    if model in ("gemma", "diffusiongemma"):
        return "gemma"
    if model in ("llada21", "llada", "llada2.1"):
        return "llada"
    raise ValueError("unsupported model %r" % (model,))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cases(path):
    data = json.loads(Path(path).read_text())
    samples = data.get("samples") if isinstance(data, dict) else None
    need(isinstance(samples, list) and len(samples) == 90, "cases",
         "need exactly 90 samples")
    cases = {}
    for case in samples:
        i = case.get("index")
        need(type(i) is int and i not in cases and case.get("dataset") in DATASETS
             and isinstance(case.get("task_id"), str) and type(case.get("seed")) is int,
             "cases", "bad/duplicate case")
        cases[i] = case
    need(sorted(cases) == list(range(90)) and
         len({case["task_id"] for case in cases.values()}) == 90, "cases",
         "indices must be 0..89 with unique task_id")
    counts = Counter(case["dataset"] for case in cases.values())
    need(counts == Counter({name: 30 for name in DATASETS}), "cases",
         "need 30 per dataset, got %r" % dict(counts))
    for name, indices in COHORTS[:-1]:
        expected = 10 if name == "original30" else 20
        need(Counter(cases[i]["dataset"] for i in indices)
             == Counter({dataset: expected for dataset in DATASETS}),
             "cases", "%s needs %d per dataset" % (name, expected))
    return cases


def load_run(run_dir, cases_path):
    run = json.loads((Path(run_dir) / "run.json").read_text())
    kind = schema(run.get("model"))
    width = 256 if kind == "gemma" else 32
    need(run.get("block") == width, "run", "%s requires block=%d" % (kind, width))
    digest = sha256(cases_path)
    need(run.get("samples_sha256") == digest, "run", "samples SHA mismatch")
    return run, kind, width, digest


def discover(run_dir, run):
    found = set()
    for path in Path(run_dir).iterdir():
        m = REQUEST.match(path.name)
        if m and (parsed := parse_policy(m.group(2))) is not None:
            need(parsed["mode"] in MODES, "policy",
                 "unsupported mode %r" % parsed["mode"])
            found.add(m.group(2))
    declared = run.get("policies")
    if isinstance(declared, list) and declared:
        found &= set(declared)
    need(found, "run", "no supported prefix policies")
    return sorted(found)


def load_requests(run_dir, policy, cases, model):
    pattern = re.compile(r"^(\d{2})-%s\.json$" % re.escape(policy))
    files = [(int(m.group(1)), p) for p in sorted(Path(run_dir).iterdir())
             if (m := pattern.match(p.name))]
    need(len(files) == 90, policy, "need 90 request files, got %d" % len(files))
    requests = {}
    for index, path in files:
        request = json.loads(path.read_text())
        need(request.get("index") == index and index not in requests,
             path.name, "filename/index mismatch or duplicate")
        case = cases[index]
        for key in ("dataset", "task_id", "seed"):
            need(request.get(key) == case.get(key), path.name, key + " mismatch")
        need(request.get("policy") == policy and request.get("model") == model
             and request.get("status") == "success", path.name,
             "policy/model/status mismatch")
        requests[index] = request
    need(sorted(requests) == list(range(90)), policy, "indices must be 0..89")
    marker = Path(run_dir) / ("complete-%s.json" % policy)
    need(marker.exists(), policy, "missing complete marker")
    payload = json.loads(marker.read_text())
    need(payload.get("policy") == policy and payload.get("requests") == 90,
         policy, "invalid complete marker")
    return requests


def pint(plan, key, label):
    value = plan.get(key)
    need(type(value) is int, label, key + " is not an integer")
    return value


def prows(plan, key, label):
    value = plan.get(key)
    need(isinstance(value, (list, tuple)) and all(type(row) is int for row in value),
         label, key + " is not an integer row list")
    need(len(set(value)) == len(value), label, key + " has duplicate rows")
    return tuple(value)


def classify(kind_schema, width, policy, index, event):
    label = "%s request %d block %r" % (kind_schema, index, event.get("block"))
    need(event.get("stage") == "selected", label, "not selected")
    plan = event.get("prefix_plan")
    need(isinstance(plan, dict), label, "missing prefix_plan")
    b, q, bs, cl, end = [pint(plan, k, label) for k in (
        "budget", "quota", "block_start", "clock", "prefix_end")]
    fr, el, zr, zm, zb, ad, ev, sel = [prows(plan, k, label) for k in (
        "fresh_rows", "eligible_prefix_rows", "zero_rows", "zero_mask_rows",
        "zero_backfill_rows", "added_rows", "evicted_rows", "selected_rows")]
    fm = plan.get("first_mask")
    need(fm is None or type(fm) is int, label, "bad first_mask")
    zw, aw = len(zr), len(sel)
    need((plan.get("zero_width"), plan.get("actual_width"), plan.get("underfill"),
          plan.get("overflow")) == (zw, aw, max(0, b - aw), max(0, aw - b)),
         label, "logged width/underfill/overflow mismatch")
    need(zr and 0 <= end <= width and bs >= 0, label, "bad plan bounds")
    for rows in (fr, el, zr, zm, zb, ad, ev, sel):
        need(all(0 <= row < width for row in rows), label, "row out of block")
    fs, ms, bf, zs = map(set, (fr, zm, zb, zr))
    es, ads, evs, ss = map(set, (el, ad, ev, sel))
    need(not (fs & ms or fs & bf or ms & bf) and (fs | ms | bf) == zs,
         label, "zero partition mismatch")
    need(fs <= ss and all(row < end for row in el) and ads <= es
         and not ads & zs and evs <= zs, label, "row pool mismatch")
    need(fm == (min(zm) if zm else None), label, "first_mask mismatch")

    parsed = parse_policy(policy)
    need(parsed is not None and parsed["mode"] in MODES, label, "bad policy")
    nominal = parsed["budget"]
    effective = 32 if (kind_schema, width, nominal) == ("llada", 32, 64) else nominal
    need(effective == b and 1 <= b <= width and parsed["fresh"] == 0
         and 0 <= q < b and q == nominal - parsed["mask"], label, "policy/plan mismatch")

    if kind_schema == "gemma":
        need(all(k in event for k in ("q", "block", "clock", "positions")),
             label, "missing Gemma fields")
        physical = event["q"]
        need(event["block"] == bs, label, "block mismatch")
        need(event["clock"] == cl, label, "clock mismatch")
        need(event["positions"] == list(sel), label, "positions mismatch")
    else:
        need(all(k in event for k in ("actual_q", "physical_q", "q_budget",
                                      "block", "step_in_block",
                                      "absolute_positions")), label,
             "missing LLaDA fields")
        physical = event["actual_q"]
        need(event["actual_q"] == event["physical_q"]
             and event["q_budget"] == b, label, "LLaDA Q mismatch")
        need(event["block"] * width == bs and event["step_in_block"] - 1 == cl,
             label, "LLaDA block/clock mismatch")
        need(event["absolute_positions"] == [bs + row for row in sel],
             label, "LLaDA absolute positions mismatch")
    need(type(physical) is int and physical == aw, label,
         "physical Q != actual width")

    mode = parsed["mode"]
    if mode == "hard_prefix_zero":
        shape = "zero"
        need(q == 0 and not ad and not ev, label, "zero with add-back")
    elif mode.startswith("hard_prefix_slack"):
        shape = "slack"
        need(not ev and ss == zs | ads and aw == zw + len(ad), label,
             "slack shape mismatch")
        target = min(q, len(el), max(0, b - zw))
        need(len(ad) == target, label, "slack did not use exact eligible seats")
        if mode == "hard_prefix_slack_oldest":
            stamps = plan.get("eligible_last_query")
            need(isinstance(stamps, list) and len(stamps) == len(el)
                 and all(type(t) is int and t >= 0 for t in stamps), label, "invalid age evidence")
            ranked = sorted(zip(stamps, el), key=lambda pair: (pair[0], -pair[1]))
            need(ads == {i for _, i in ranked[:target]}, label, "oldest order mismatch")
    elif not ad and not ev:
        shape = "noop" if q > 0 else "zero"
    else:
        shape = "swap"
        need(len(ev) == len(ad) and ss == (zs - evs) | ads and aw == zw,
             label, "swap shape mismatch")
    if shape in ("zero", "noop"):
        need(ss == zs and aw == zw, label, "no-add-back selected != zero")
    md, od = len(evs & ms), len(evs & bf)
    need(md + od == len(evs) and not evs & ss, label, "bad displacement")
    slack = max(0, b - zw)
    overflow = max(0, len(fr) - b)
    pool = len(el)
    return dict(kind=shape, quota=q, physical=physical, slack=slack,
                fresh_overflow=overflow, pool=pool, opp1=min(1, slack, pool),
                opp4=min(4, slack, pool), added=len(ad), mask=md, ordinary=od,
                noop=q > 0 and not ad, saturated=q > 0 and len(ad) >= q)


def verify_query_history(events, kind, width):
    """Rebuild physical-query timestamps BEFORE applying the current forward."""
    stamps = None
    block = None
    for index, event in enumerate(events, 1):
        stage = event["stage"]
        if stage == "prefill":
            continue
        if stage == "bootstrap":
            block, stamps, tick = event["block"], [-1] * width, 0
        need(stamps is not None and event["block"] == block, kind, "missing block bootstrap")
        tick = tick + 1 if kind == "gemma" else index
        need(event.get("hard_clock") == tick, kind, "physical history clock mismatch")
        if stage == "selected":
            plan = event["prefix_plan"]
            need(plan["eligible_last_query"] == [stamps[i] for i in plan["eligible_prefix_rows"]],
                 kind, "physical history does not reproduce eligible timestamps")
        queried = event["positions"] if kind == "gemma" else [
            p - block * width for p in event["absolute_positions"]]
        for row in queried:
            need(0 <= row < width, kind, "history position outside physical block")
            stamps[row] = tick


def audit(run_dir, policy, cases, model, kind_schema, width):
    requests = load_requests(run_dir, policy, cases, model)
    totals = {name: Counter() for name, _ in COHORTS}
    kinds = {name: set() for name, _ in COHORTS}
    for index in sorted(requests):
        trace = requests[index].get("events", [])
        if policy.endswith("_slack_oldest"):
            verify_query_history(trace, kind_schema, width)
        events = [e for e in trace if e.get("stage") == "selected"]
        per = Counter()
        for event in events:
            item = classify(kind_schema, width, policy, index, event)
            updates = (("events", 1), ("physical_q", item["physical"]),
                ("positive_slack_events", item["slack"] > 0),
                ("slack_rows", item["slack"]),
                ("fresh_overflow_events", item["fresh_overflow"] > 0),
                ("fresh_overflow_rows", item["fresh_overflow"]),
                ("eligible_events", item["pool"] > 0),
                ("eligible_rows", item["pool"]), ("opp_k1_rows", item["opp1"]),
                ("opp_k4_rows", item["opp4"]),
                ("nominal_k_rows", item["quota"]),
                ("added_rows", item["added"]),
                ("mask_displaced", item["mask"]),
                ("ordinary_displaced", item["ordinary"]),
                ("noop_events", item["noop"]),
                ("saturated_events", item["saturated"]))
            for name, indices in COHORTS:
                if index in indices:
                    kinds[name].add(item["kind"])
                    for key, value in updates:
                        totals[name][key] += value
            per["slack"] += item["slack"] > 0
            per["opp1"] += item["opp1"] > 0
            per["opp4"] += item["opp4"] > 0
            per["added"] += item["added"]
        for name, indices in COHORTS:
            if index in indices:
                row = totals[name]
                row["requests"] += 1
                row["positive_slack_requests"] += per["slack"] > 0
                row["opp_k1_requests"] += per["opp1"] > 0
                row["opp_k4_requests"] += per["opp4"] > 0
                row["untouched_requests"] += per["added"] == 0
    rows = []
    for name, _ in COHORTS:
        row = totals[name]
        out = {field: row.get(field, 0) for field in FIELDS}
        out.update(model=model, policy=policy,
                   kind="+".join(sorted(kinds[name])) or "none", cohort=name,
                   mean_q_rows=(row["physical_q"] / row["events"])
                   if row["events"] else 0.0)
        rows.append(out)
    return rows


def write_csv(rows, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(run_dir, cases_path, out, policies=None):
    run, kind_schema, width, digest = load_run(run_dir, cases_path)
    cases = load_cases(cases_path)
    found = discover(run_dir, run)
    selected = list(policies) if policies else found
    need(set(selected) <= set(found), "policies", "policy not present in run")
    rows = []
    for policy in selected:
        rows.extend(audit(run_dir, policy, cases, run["model"], kind_schema, width))
    write_csv(rows, out)
    return rows, run, kind_schema, width, digest


def main(argv=None):
    default = Path("/data1/project_data/gx-erqw-20261123/runtime/"
                   "t2t-native-1125/prefix90/gemma-q32")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=default)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--out", type=Path, default=Path(
        "experiments/t2t/prefix_window/exposure-gemma-q32.csv"))
    parser.add_argument("--policy", action="append", default=[])
    args = parser.parse_args(argv)
    cases_path = args.cases or (args.run.parent / "samples.json")
    rows, run, kind_schema, width, digest = build(
        args.run, cases_path, args.out, args.policy or None)
    summary = [{key: row[key] for key in FIELDS if key != "cohort"}
               for row in rows if row["cohort"] == "all90"]
    print(json.dumps(dict(run=str(args.run), model=run["model"],
                          schema=kind_schema, width=width,
                          samples_sha256=digest, rows=summary), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
