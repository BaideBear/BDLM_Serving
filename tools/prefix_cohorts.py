"""Auditable 30/60/90 paired cohort scorer for prefix-window policies.

The scorer consumes two request sets over the registered frozen90 cases: one
baseline (arbitrary zero/control arm) and one arm.  Cohorts are disjoint or
exhaustive slices of the canonical frozen90 order, never overlapping variants:

* window 30  -> original only: cases 0-29 (10 per dataset)
* window 60  -> added only:    cases 30-89 (20 per dataset)
* window 90  -> union:         cases 0-89 (30 per dataset)

Original 0-29 and added 30-89 are disjoint and together exhaust the 90 cases.
There is deliberately no "original + first half of added" 60-case cohort.
The baseline may be any zero/control arm at the same nominal Q, not only a
policy named native; policy names are free-form and nominal Q is parsed when present.

For every window and every dataset (plus the pooled "all" scope) it reports
quality, macro TPF, pooled token efficiency, actual F/Q and Q/F, paired
correctness rescued/lost, and 95% intervals for the quality and macro-TPF
deltas.  Quality uses a labelled Newcombe hybrid-score paired interval with a
Wilson square-and-add fallback when a stratum has zero discordance, so it does
not collapse to zero width when the arms agree on every case; TPF keeps the
paired stratified normal interval.  All request work is checked with the same
``verify_work`` used by the existing T2T scorer; correctness is re-scored from
``output_text`` with the existing frozen-task scorer.

Missing, duplicate and metadata-mismatched requests are rejected. The one
documented LLaDA Optimus progress-limit outcome counts as wrong, with every
attempted forward charged. Other failure modes are not treated as scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from bdlm_serving.evaluation import frozen90
from bdlm_serving.evaluation.scoring import score
from bdlm_serving.methods.t2t.protocol import verify_work
from tools.t2t_results import summarize, table


WINDOWS = (30, 60, 90)
COHORTS = {30: "original-0-29", 60: "added-30-89", 90: "all-0-89"}
WINDOW_INDEX_SETS = {30: set(range(0, 30)), 60: set(range(30, 90)),
                     90: set(range(0, 90))}
WINDOW_DATASET_COUNTS = {30: 10, 60: 20, 90: 30}
MODEL_BLOCK = {"llada21": 32, "gemma": 256}
MODEL_ALIASES = {"diffusiongemma": "gemma"}
Z95 = 1.959963984540054
TPF_CI_METHOD = "paired stratified normal (z=1.959963984540054, strata=dataset)"
QUALITY_CI_METHOD = ("Newcombe hybrid-score paired interval; Wilson square-and-add "
                     "fallback at zero discordance; stratified by dataset")
CI_METHOD = f"{TPF_CI_METHOD}; quality: {QUALITY_CI_METHOD}"


def load_cases(path):
    """Load and validate the registered frozen90 samples snapshot."""
    return frozen90.load_tasks(Path(path))


def canonical_model(model):
    """Normalize only exact, known online/offline model-name aliases."""
    return MODEL_ALIASES.get(model, model)


def select_window_cases(cases, window):
    """Return the disjoint/exhaustive cohort for ``window``.

    window 30 is exactly original 0-29; window 60 is exactly added 30-89;
    window 90 is their union 0-89.  No window overlaps the original and added
    sets.  Each cohort must keep the frozen90 10/20/30-per-dataset balance.
    """
    if window not in WINDOW_INDEX_SETS:
        raise ValueError(f"Unsupported window: {window}")
    ordered = sorted(cases, key=lambda case: case["index"])
    datasets = sorted({case["dataset"] for case in ordered})
    if [case["index"] for case in ordered] != list(range(90)):
        raise ValueError("Frozen90 case indices must be exactly 0..89")
    selected = [case for case in ordered if case["index"] in WINDOW_INDEX_SETS[window]]
    if [case["index"] for case in selected] != sorted(WINDOW_INDEX_SETS[window]):
        raise ValueError(f"Window {window}: noncanonical case set")
    counts = Counter(case["dataset"] for case in selected)
    expected = WINDOW_DATASET_COUNTS[window]
    if set(counts) != set(datasets) or any(counts[dataset] != expected for dataset in datasets):
        raise ValueError(f"Window {window}: expected {expected} per dataset, got {dict(counts)}")
    original = {case["index"] for case in ordered[:30]}
    added = {case["index"] for case in ordered[30:]}
    if original != set(range(30)) or added != set(range(30, 90)):
        raise ValueError("Original 0-29 and added 30-89 are not disjoint/exhaustive")
    return selected


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_digest(path):
    path = Path(path)
    if not path.is_dir():
        return _sha256(path)
    digest = hashlib.sha256()
    for child in sorted(p for p in path.iterdir() if p.is_file()):
        digest.update(child.name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(child)))
    return digest.hexdigest()


def _looks_like_request(value):
    return (isinstance(value, dict) and "index" in value and
            "dataset" in value and "task_id" in value)


def _expand_request_data(data):
    if isinstance(data, list):
        rows = []
        for item in data:
            rows.extend(_expand_request_data(item))
        return rows
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("requests"), list):
        return _expand_request_data(data["requests"])
    if _looks_like_request(data):
        return [data]
    return []


def _read_request_file(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {error}") from error
            rows.extend(_expand_request_data(payload))
        return rows
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error
    return _expand_request_data(payload)


def read_request_rows(path, policy=None):
    """Read request objects from a JSON aggregate, JSONL file, or run folder.

    In a folder, valid JSON files that are not request objects (for example
    ``run.json`` or ``complete-<policy>.json``) are ignored.  When ``policy`` is
    supplied only requests carrying that policy are retained; otherwise exactly
    one policy must be present.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Request path does not exist: {path}")
    if path.is_dir():
        files = sorted(p for p in path.iterdir()
                       if p.is_file() and p.suffix in (".json", ".jsonl"))
        if not files:
            raise ValueError(f"No .json/.jsonl request files in {path}")
        rows = []
        for child in files:
            rows.extend(_read_request_file(child))
    elif path.suffix in (".json", ".jsonl"):
        rows = _read_request_file(path)
    else:
        raise ValueError(f"Request path must be a .json/.jsonl file or folder: {path}")
    if policy is not None:
        rows = [row for row in rows if row.get("policy") == policy]
    if not rows:
        suffix = "" if policy is None else f" for policy {policy!r}"
        raise ValueError(f"No request objects{suffix} in {path}")
    policies = sorted({row.get("policy") for row in rows})
    if policy is None and len(policies) != 1:
        raise ValueError(f"Multiple policies in {path}: {policies}; pass --arm-policy/--baseline-policy")
    return rows


def _as_bool(value, field, index):
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("0", "false", "no", ""):
            return False
        if text in ("1", "true", "yes"):
            return True
    raise ValueError(f"request {index}: invalid {field}: {value!r}")


def _int_field(raw, names, label, index):
    for name in names:
        if name in raw:
            value = raw[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"request {index}: invalid {label}: {value!r}")
            return value
    raise ValueError(f"request {index}: missing {label}")


def _number_field(raw, name, index):
    if name not in raw:
        return None
    value = raw[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"request {index}: invalid {name}: {value!r}")
    return float(value)


_POLICY_Q = re.compile(r"(?:^|_)q(\d+)(?:_|$)")
_GEMMA_OPTIMUS = re.compile(r"(ibs|obs)_(4|8|16|32|64|128|256)")
_LL_OPT = re.compile(r"optimus_(ibs|obs)_c(4|8|16|32)")


def llada_reference_chunk(raw, block):
    match = _LL_OPT.fullmatch(raw["policy"])
    opt = bool(match)
    nominal = int(match[2]) if opt else int(raw["policy"][-1])*block
    work, prefill = [], 0
    if raw["phase"]!="formal" or raw["generation_cap"]!=2048:
        raise ValueError("Wrong LLaDA generation protocol")
    if opt and (raw["extra"]["fixed_chunk"]!=nominal or raw["extra"]["mode"]!=match[1]
                or raw["extra"]["window_switches"]!=0):
        raise ValueError("Wrong fixed Optimus configuration")
    for e,c in zip(raw["events"],raw["calls"],strict=True):
        q = c["rows"]
        if (type(q) is not int or q<1 or c["batch"]!=1 or q!=c["physical_rows"] or
                e["physical_forwards"]!=1 or e["q_tokens"]!=q):
            raise ValueError("LLaDA trace disagrees with physical call")
        if e["phase"]=="prefill":
            if work or prefill:
                raise ValueError("Only the initial prompt prefill is free")
            prefill=q
            continue
        if len(e["absolute_positions"])!=q:
            raise ValueError("Wrong number of LLaDA positions")
        if opt:
            if (e["phase"]!="optimus" or e["fixed_chunk"]!=nominal or e["mode"]!=match[1]
                    or len(set(e["absolute_positions"]))!=q):
                raise ValueError("Optimus changed mode or chunk")
        elif (e["phase"]!="flowblock_packed" or q!=nominal or e["max_physical_q"]!=q or
              e["query_rows_over_layers"] != e["physical_layer_attention_calls"]*q):
            raise ValueError("Wrong packed FlowBlock width")
        work.append(q)
    if (not work or prefill!=raw["prefill_rows"] or len(work)!=raw["generation_forwards"]
            or sum(work)!=raw["generation_query_rows"] or sum(work)!=raw["generation_physical_padded_rows"]
            or max(work)!=raw["max_q"] or Counter(map(str,work))!=raw["q_histogram"]):
        raise ValueError("Incomplete LLaDA work ledger")
    failure=raw["request_failure"]
    if raw["status"]!="success":
        if (not opt or raw["status"]!="generation_failed" or not isinstance(failure,dict)
                or failure.get("type")!="optimus_progress_limit" or failure!=raw["extra"].get("request_failure")
                or failure.get("delivered_output_tokens")!=0 or raw["output_tokens"] or raw["output_ids"]
                or raw["output_text"]!="" or raw["eos"] or raw["truncated"] or
                failure.get("steps_in_block")!=failure.get("step_limit") or
                failure.get("step_limit")!=2*raw["events"][-1]["active_width"]+4 or
                not 0<=failure.get("cached_prefix",-1)<block):
            raise ValueError("Unrecognized or inconsistent LLaDA generation failure")
    elif failure is not None:
        raise ValueError("Unexpected failure record on successful generation")
    return nominal


def optimus_chunk(raw, block):
    """Optimus reports observed max Q, not a model block-size setting."""
    match = _GEMMA_OPTIMUS.fullmatch(raw["policy"])
    mode, c = match.group(1), int(match.group(2))
    events, trace = raw["events"], raw["trace"]
    if not events or events[0][0] != "encoder" or not trace:
        raise ValueError("Missing Optimus prefill or decoder trace")
    gen = events[1:]
    dec = [q for tag, q in gen if tag == "decoder"]
    if (not dec or any(tag not in {"encoder", "decoder"} or type(q) is not int or q < 1
                       or (tag == "encoder" and q != block) for tag, q in gen)
            or any(width > block + (c if mode == "obs" else 0) for width in dec)
            or dec != [step["q"] for step in trace]
            or any(not 0 <= step["cross"] <= min(c if mode == "obs" else 0, step["q"]) for step in trace)
            or raw["max_q"] != max(q for _, q in gen) or raw["max_decoder_q"] != max(dec)
            or raw["generation_forwards"] != len(gen)
            or raw["generation_query_rows"] != sum(q for _, q in gen)
            or _nominal_q(raw, raw["index"]) not in (None, c)):
        raise ValueError("Optimus policy or work counters disagree with observed events")
    return c


def check_gemma_wave(raw, block):
    # The archived adapter pads a fixed W-block decoder. A resident but idle
    # slot still enters the transformer; encoder re-encodes remain one block.
    w = int(raw["policy"].removeprefix("flow_w"))
    events, trace = raw["events"], raw["trace"]
    if not events or events[0][0] != "encoder" or not trace:
        raise ValueError("Missing Gemma wave prefill or decoder trace")
    gen = events[1:]
    if (any(tag not in ("encoder", "decoder") or type(q) is not int or
            q != (block if tag == "encoder" else w*block) for tag, q in gen)
            or [q for tag,q in gen if tag == "decoder"] != [s["q"] for s in trace]
            or any(s["width"] != w or not 1 <= s["active"] <= w for s in trace)
            or raw["max_q"] != w * block or raw["generation_forwards"] != len(gen)
            or raw["generation_query_rows"] != sum(q for _,q in gen)):
        raise ValueError("Gemma wave work does not match packed physical rows")


def _nominal_q(raw, index):
    """Best-effort nominal budget from an explicit field or t2t_q<N> policy."""
    for name in ("nominal_q", "budget"):
        if name in raw:
            value = raw[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"request {index}: invalid {name}: {value!r}")
            return value
    policy = raw.get("policy")
    if isinstance(policy, str):
        match = _POLICY_Q.search(policy)
        if match:
            return int(match.group(1))
    return None


def _request_failed(raw, index):
    status = raw.get("status")
    failed_present = "failed" in raw
    if status is not None:
        if not isinstance(status, str):
            raise ValueError(f"request {index}: invalid status: {status!r}")
        failed = status != "success"
        if failed_present and _as_bool(raw["failed"], "failed", index) and not failed:
            raise ValueError(f"request {index}: status is success but failed flag is set")
        return failed
    if not failed_present:
        raise ValueError(f"request {index}: missing status/failed marker")
    return _as_bool(raw["failed"], "failed", index)


def _identity(row):
    return (row["index"], row["dataset"], row["task_id"])


def normalize_request(raw, case, role):
    """Validate one request against its frozen90 case and re-score it."""
    index = case["index"]
    for key in ("index", "dataset", "task_id", "seed"):
        if raw.get(key) != case[key]:
            raise ValueError(f"{role} request {index}: {key} mismatch "
                             f"({raw.get(key)!r} != {case[key]!r})")
    raw_model = raw.get("model")
    policy = raw.get("policy")
    if not isinstance(raw_model, str) or not raw_model:
        raise ValueError(f"{role} request {index}: missing model")
    if not isinstance(policy, str) or not policy:
        raise ValueError(f"{role} request {index}: missing policy")
    model = canonical_model(raw_model)
    if model not in MODEL_BLOCK:
        raise ValueError(f"{role} request {index}: unsupported model {raw_model!r}")
    block = MODEL_BLOCK[model]
    max_q = raw.get("max_q", block)
    fixed_chunk = None
    ll_ref = model=="llada21" and (_LL_OPT.fullmatch(policy) or policy in ("flow_w1","flow_w2"))
    if ll_ref:
        fixed_chunk = llada_reference_chunk(raw, block)
    elif model=="llada21" and policy=="dual64_noedit":
        from tools.run_dual_t2t import audit_dual64
        audit_dual64(raw)
        fixed_chunk=64
    elif model == "gemma" and _GEMMA_OPTIMUS.fullmatch(policy):
        fixed_chunk = optimus_chunk(raw, block)
    elif model == "gemma" and policy in ("flow_w1", "flow_w2"):
        check_gemma_wave(raw, block)
    elif isinstance(max_q, bool) or not isinstance(max_q, int) or max_q != block:
        raise ValueError(f"{role} request {index}: max_q {max_q!r} does not match {model} B{block}")
    failed = _request_failed(raw, index)
    if failed and not ll_ref:
        raise ValueError(f"{role} request {index}: failed status {raw.get('status')!r}")
    n = _int_field(raw, ("output_tokens", "N"), "output_tokens", index)
    f = _int_field(raw, ("generation_forwards", "F"), "generation_forwards", index)
    q = _int_field(raw, ("generation_query_rows", "Q"), "generation_query_rows", index)
    physical_q = _int_field(raw, ("generation_physical_padded_rows", "physical_Q"),
                            "generation_physical_padded_rows", index)
    tpf = _number_field(raw, "tpf", index)
    token_efficiency = _number_field(raw, "token_efficiency", index)
    if tpf is None:
        if f <= 0:
            raise ValueError(f"{role} request {index}: generation_forwards must be positive")
        tpf = n / f
    if token_efficiency is None:
        if q <= 0:
            raise ValueError(f"{role} request {index}: generation_query_rows must be positive")
        token_efficiency = n / q
    work = dict(output_tokens=n, generation_forwards=f, generation_query_rows=q,
                generation_physical_padded_rows=physical_q, tpf=tpf,
                token_efficiency=token_efficiency)
    try:
        verify_work(work)
    except ValueError as error:
        raise ValueError(f"{role} request {index}: {error}") from error
    text = raw.get("output_text")
    if not isinstance(text, str):
        raise ValueError(f"{role} request {index}: missing output_text")
    try:
        result = {"correct":False} if failed else score(text, case)
    except Exception as error:  # scoring failures are audit failures, not silent zeros
        raise ValueError(f"{role} request {index}: scoring failed: {error}") from error
    correct = bool(result["correct"])
    if "correct" in raw and _as_bool(raw["correct"], "correct", index) != correct:
        raise ValueError(f"{role} request {index}: provided correct disagrees with scorer")
    truncated = _as_bool(raw.get("truncated", False), "truncated", index)
    nominal_q = fixed_chunk or _nominal_q(raw, index)
    return dict(index=index, dataset=case["dataset"], task_id=case["task_id"],
                seed=case["seed"], model=model, policy=policy, max_q=max_q,
                N=n, F=f, Q=q, physical_Q=physical_q, tpf=tpf,
                token_efficiency=token_efficiency, correct=int(correct),
                failed=int(failed), truncated=int(truncated), nominal_q=nominal_q)


def load_request_set(path, cases, role, policy=None):
    """Load exactly one complete request set over all frozen90 cases."""
    case_by_index = {case["index"]: case for case in cases}
    raw_rows = read_request_rows(path, policy)
    rows_by_index = {}
    tasks = set()
    for raw in raw_rows:
        index = raw.get("index")
        if index not in case_by_index:
            raise ValueError(f"{role}: unknown case index {index!r}")
        row = normalize_request(raw, case_by_index[index], role)
        if index in rows_by_index:
            raise ValueError(f"{role}: duplicate case index {index}")
        task = (row["dataset"], row["task_id"])
        if task in tasks:
            raise ValueError(f"{role}: duplicate task {task}")
        tasks.add(task)
        rows_by_index[index] = row
    expected = set(case_by_index)
    missing = sorted(expected - set(rows_by_index))
    if missing:
        raise ValueError(f"{role}: missing case indices {missing}")
    extra = sorted(set(rows_by_index) - expected)
    if extra:
        raise ValueError(f"{role}: unexpected case indices {extra}")
    rows = [rows_by_index[index] for index in sorted(rows_by_index)]
    models = {row["model"] for row in rows}
    policies = {row["policy"] for row in rows}
    nominal_qs = sorted({row["nominal_q"] for row in rows if row["nominal_q"] is not None})
    if len(models) != 1:
        raise ValueError(f"{role}: mixed models {sorted(models)}")
    if len(policies) != 1:
        raise ValueError(f"{role}: mixed policies {sorted(policies)}")
    if len(nominal_qs) > 1:
        raise ValueError(f"{role}: mixed nominal Q {nominal_qs}")
    return (rows, next(iter(models)), next(iter(policies)),
            nominal_qs[0] if nominal_qs else None)


def paired_ci(values_by_group, *, bounds=None):
    """Paired stratified normal 95% CI for a mean difference.

    Each group is a dataset stratum.  The stratum weights follow the observed
    cohort sizes (proportional allocation), so equal-sized balanced cohorts
    reduce to the unweighted stratified mean.
    """
    groups = [list(values) for values in values_by_group.values() if values]
    if not groups:
        raise ValueError("Cannot form a CI from no paired values")
    total = sum(len(group) for group in groups)
    estimate = sum((len(group) / total) * mean(group) for group in groups)
    variance = 0.0
    for group in groups:
        if len(group) < 2:
            raise ValueError("Each stratum needs at least two paired values")
        center = mean(group)
        sample_variance = sum((value - center) ** 2 for value in group) / (len(group) - 1)
        weight = len(group) / total
        variance += weight * weight * sample_variance / len(group)
    se = math.sqrt(max(variance, 0.0))
    low, high = estimate - Z95 * se, estimate + Z95 * se
    if bounds is not None:
        low, high = max(bounds[0], low), min(bounds[1], high)
    return estimate, se, low, high


def _wilson(k, n):
    """Two-sided 95% Wilson score interval for one binomial proportion."""
    if n <= 0:
        raise ValueError("Wilson interval needs at least one observation")
    p = k / n
    z2 = Z95 * Z95
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    half = Z95 * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _newcombe_paired(both, baseline_only, arm_only, neither):
    """Newcombe hybrid-score interval for a paired difference in proportions."""
    n = both + baseline_only + arm_only + neither
    if n <= 0:
        raise ValueError("Newcombe interval needs at least one observation")
    baseline_correct = both + baseline_only
    arm_correct = both + arm_only
    p1 = baseline_correct / n
    p2 = arm_correct / n
    l1, u1 = _wilson(baseline_correct, n)
    l2, u2 = _wilson(arm_correct, n)
    denominator = math.sqrt((both + baseline_only) * (arm_only + neither) *
                            (both + arm_only) * (baseline_only + neither))
    phi = 0.0 if denominator == 0 else (both * neither - baseline_only * arm_only) / denominator
    phi = max(-1.0, min(1.0, phi))
    lower_term = ((p2 - l2) ** 2
                  - 2.0 * phi * (p2 - l2) * (u1 - p1)
                  + (u1 - p1) ** 2)
    upper_term = ((u2 - p2) ** 2
                  - 2.0 * phi * (u2 - p2) * (p1 - l1)
                  + (p1 - l1) ** 2)
    low = (p2 - p1) - math.sqrt(max(lower_term, 0.0))
    high = (p2 - p1) + math.sqrt(max(upper_term, 0.0))
    return p2 - p1, low, high


def quality_ci(counts_by_group):
    """Score-based paired quality interval from labelled 2x2 counts.

    ``counts_by_group`` maps a dataset stratum to
    ``(both_correct, baseline_only, arm_only, both_wrong)``.  With at least one
    discordant case the interval is Newcombe's hybrid-score paired interval.
    When a stratum has zero discordance, that paired interval can collapse to
    zero width, so the stratum falls back to the Wilson square-and-add interval
    ``[arm_low - baseline_high, arm_high - baseline_low]``.  The fallback is
    conservative for paired data but non-degenerate.  Strata are combined with
    proportional weights and the returned count reports fallback strata.
    """
    parts = []
    fallback_strata = 0
    for counts in counts_by_group.values():
        both, baseline_only, arm_only, neither = counts
        n = both + baseline_only + arm_only + neither
        if n == 0:
            continue
        baseline_correct = both + baseline_only
        arm_correct = both + arm_only
        estimate = (arm_correct - baseline_correct) / n
        if baseline_only + arm_only:
            estimate, low, high = _newcombe_paired(
                both, baseline_only, arm_only, neither)
            if high - low <= 1e-12:
                baseline_low, baseline_high = _wilson(baseline_correct, n)
                arm_low, arm_high = _wilson(arm_correct, n)
                low = arm_low - baseline_high
                high = arm_high - baseline_low
                fallback_strata += 1
        else:
            baseline_low, baseline_high = _wilson(baseline_correct, n)
            arm_low, arm_high = _wilson(arm_correct, n)
            low = arm_low - baseline_high
            high = arm_high - baseline_low
            fallback_strata += 1
        low = max(-1.0, low)
        high = min(1.0, high)
        parts.append((n, estimate, low, high))
    if not parts:
        raise ValueError("Cannot form a quality interval from no paired values")
    total = sum(part[0] for part in parts)
    estimate = sum(part[0] / total * part[1] for part in parts)
    lower_var = sum((part[0] / total) ** 2 * (part[1] - part[2]) ** 2
                    for part in parts)
    upper_var = sum((part[0] / total) ** 2 * (part[3] - part[1]) ** 2
                    for part in parts)
    low = max(-1.0, estimate - math.sqrt(max(lower_var, 0.0)))
    high = min(1.0, estimate + math.sqrt(max(upper_var, 0.0)))
    se = ((estimate - low) + (high - estimate)) / (2.0 * Z95)
    return estimate, se, low, high, fallback_strata


def _paired_counts(baseline_rows, arm_rows):
    counts = Counter(both_correct=0, both_wrong=0, rescued=0, lost=0)
    for baseline, arm in zip(baseline_rows, arm_rows):
        if _identity(baseline) != _identity(arm):
            raise ValueError("Cannot pair different cases")
        if arm["correct"] and not baseline["correct"]:
            counts["rescued"] += 1
        elif baseline["correct"] and not arm["correct"]:
            counts["lost"] += 1
        elif baseline["correct"] and arm["correct"]:
            counts["both_correct"] += 1
        else:
            counts["both_wrong"] += 1
    return counts


def _window_row(window_cases, baseline_by_index, arm_by_index, window, scope,
                model, baseline_policy, arm_policy, nominal_q):
    subset = [case for case in window_cases
              if scope == "all" or case["dataset"] == scope]
    if not subset:
        raise ValueError(f"Empty scope {scope!r} at window {window}")
    baseline_rows = [baseline_by_index[case["index"]] for case in subset]
    arm_rows = [arm_by_index[case["index"]] for case in subset]
    baseline_summary = summarize(baseline_rows)
    arm_summary = summarize(arm_rows)
    paired = _paired_counts(baseline_rows, arm_rows)
    requests = baseline_summary["requests"]
    baseline_quality = baseline_summary["correct"] / requests
    arm_quality = arm_summary["correct"] / requests
    quality_delta = arm_quality - baseline_quality
    tpf_delta = arm_summary["tpf"] - baseline_summary["tpf"]
    quality_counts = defaultdict(lambda: [0, 0, 0, 0])
    tpf_groups = defaultdict(list)
    for case, baseline, arm in zip(subset, baseline_rows, arm_rows):
        cell = quality_counts[case["dataset"]]
        if baseline["correct"] and arm["correct"]:
            cell[0] += 1
        elif baseline["correct"] and not arm["correct"]:
            cell[1] += 1
        elif not baseline["correct"] and arm["correct"]:
            cell[2] += 1
        else:
            cell[3] += 1
        tpf_groups[case["dataset"]].append(arm["tpf"] - baseline["tpf"])
    (quality_estimate, quality_se, quality_low, quality_high,
     quality_fallback_strata) = quality_ci(quality_counts)
    tpf_estimate, tpf_se, tpf_low, tpf_high = paired_ci(tpf_groups)
    if abs(quality_estimate - quality_delta) > 1e-12:
        raise AssertionError("Paired quality delta disagrees with summary quality")
    if abs(tpf_estimate - tpf_delta) > 1e-9:
        raise AssertionError("Paired TPF delta disagrees with summary TPF")
    dataset_counts = Counter(case["dataset"] for case in subset)
    return dict(
        model=model, arm_policy=arm_policy, baseline_policy=baseline_policy,
        nominal_q="" if nominal_q is None else nominal_q,
        window=window, cohort=COHORTS[window], scope=scope, requests=requests,
        dataset_counts=";".join(f"{name}={dataset_counts[name]}"
                                for name in sorted(dataset_counts)),
        strata=len(quality_counts),
        baseline_correct=baseline_summary["correct"], arm_correct=arm_summary["correct"],
        baseline_quality=baseline_quality, arm_quality=arm_quality,
        quality_delta=quality_delta,
        quality_delta_ci_low=quality_low, quality_delta_ci_high=quality_high,
        quality_delta_se=quality_se,
        quality_ci_method=QUALITY_CI_METHOD,
        quality_ci_fallback_strata=quality_fallback_strata,
        rescued=paired["rescued"], lost=paired["lost"],
        both_correct=paired["both_correct"], both_wrong=paired["both_wrong"],
        baseline_macro_tpf=baseline_summary["tpf"], arm_macro_tpf=arm_summary["tpf"],
        macro_tpf_delta=tpf_delta,
        macro_tpf_delta_ci_low=tpf_low, macro_tpf_delta_ci_high=tpf_high,
        macro_tpf_delta_se=tpf_se,
        baseline_pooled_token_efficiency=baseline_summary["token_efficiency"],
        arm_pooled_token_efficiency=arm_summary["token_efficiency"],
        pooled_token_efficiency_delta=(arm_summary["token_efficiency"]
                                       - baseline_summary["token_efficiency"]),
        baseline_N=baseline_summary["N"], baseline_F=baseline_summary["F"],
        baseline_Q=baseline_summary["Q"], baseline_physical_Q=baseline_summary["physical_Q"],
        baseline_q_per_f=baseline_summary["q_per_f"],
        arm_N=arm_summary["N"], arm_F=arm_summary["F"],
        arm_Q=arm_summary["Q"], arm_physical_Q=arm_summary["physical_Q"],
        arm_q_per_f=arm_summary["q_per_f"],
        delta_N=arm_summary["N"] - baseline_summary["N"],
        delta_F=arm_summary["F"] - baseline_summary["F"],
        delta_Q=arm_summary["Q"] - baseline_summary["Q"],
        failed=baseline_summary["failed"] + arm_summary["failed"],
        truncated=baseline_summary["truncated"] + arm_summary["truncated"],
        ci_method=CI_METHOD,
        tpf_ci_method=TPF_CI_METHOD,
    )


def _paired_case_row(case, baseline, arm, window, model, baseline_policy,
                     arm_policy, nominal_q):
    return dict(
        model=model, arm_policy=arm_policy, baseline_policy=baseline_policy,
        nominal_q="" if nominal_q is None else nominal_q,
        window=window, cohort=COHORTS[window], index=case["index"],
        dataset=case["dataset"], task_id=case["task_id"], seed=case["seed"],
        baseline_correct=baseline["correct"], arm_correct=arm["correct"],
        rescued=int(bool(arm["correct"]) and not bool(baseline["correct"])),
        lost=int(bool(baseline["correct"]) and not bool(arm["correct"])),
        baseline_N=baseline["N"], baseline_F=baseline["F"], baseline_Q=baseline["Q"],
        baseline_tpf=baseline["tpf"],
        baseline_token_efficiency=baseline["token_efficiency"],
        arm_N=arm["N"], arm_F=arm["F"], arm_Q=arm["Q"],
        arm_tpf=arm["tpf"], arm_token_efficiency=arm["token_efficiency"],
        baseline_truncated=baseline["truncated"], arm_truncated=arm["truncated"],
    )


def _audit(baseline_path, arm_path, cases_path, cases, model, baseline_policy,
           arm_policy, baseline_rows, arm_rows, baseline_q, arm_q, nominal_q):
    window_indices = {str(window): [case["index"] for case in select_window_cases(cases, window)]
                      for window in WINDOWS}
    counts = {str(window): dict(Counter(case["dataset"]
                                        for case in select_window_cases(cases, window)))
              for window in WINDOWS}
    return dict(
        tool="prefix_cohorts",
        model=model,
        block=MODEL_BLOCK[model],
        baseline_policy=baseline_policy,
        arm_policy=arm_policy,
        nominal_q=dict(baseline=baseline_q, arm=arm_q, resolved=nominal_q,
                       matched=(None if baseline_q is None or arm_q is None
                                else baseline_q == arm_q)),
        cases=len(cases),
        case_order="frozen90.load_tasks: 0-29 original, 30-89 added",
        windows=list(WINDOWS),
        cohorts={str(window): COHORTS[window] for window in WINDOWS},
        window_policy=("disjoint/exhaustive: 30=original 0-29, 60=added 30-89, "
                       "90=union 0-89"),
        dataset_counts_per_window=counts,
        window_indices=window_indices,
        requests=dict(baseline=len(baseline_rows), arm=len(arm_rows)),
        failed=dict(baseline=sum(row["failed"] for row in baseline_rows),
                    arm=sum(row["failed"] for row in arm_rows)),
        truncated=dict(baseline=sum(row["truncated"] for row in baseline_rows),
                       arm=sum(row["truncated"] for row in arm_rows)),
        ci_method=CI_METHOD,
        quality_ci_method=QUALITY_CI_METHOD,
        tpf_ci_method=TPF_CI_METHOD,
        inputs=dict(baseline=str(baseline_path), arm=str(arm_path), cases=str(cases_path)),
        sha256=dict(baseline=_path_digest(baseline_path), arm=_path_digest(arm_path),
                    cases=_path_digest(cases_path)),
    )


def score_files(baseline, arm, cases, out, *, paired_out=None, audit_out=None,
                baseline_policy=None, arm_policy=None, nominal_q=None):
    """Score one baseline/arm pair over the frozen90 30/60/90 cohorts.

    The baseline may be any arbitrary zero/control arm, not just a policy named
    ``native``.  If both request sets expose a nominal Q, they must match; an
    explicit ``nominal_q`` can pin a budget for arms whose policy names do not
    encode it.
    """
    cases_path = Path(cases)
    cases = load_cases(cases_path)
    baseline_rows, baseline_model, baseline_label, baseline_q = load_request_set(
        baseline, cases, "baseline", baseline_policy)
    arm_rows, arm_model, arm_label, arm_q = load_request_set(
        arm, cases, "arm", arm_policy)
    if baseline_model != arm_model:
        raise ValueError(f"Baseline model {baseline_model!r} != arm model {arm_model!r}")
    if baseline_q is not None and arm_q is not None and baseline_q != arm_q:
        raise ValueError(f"Baseline nominal Q {baseline_q} != arm nominal Q {arm_q}")
    if nominal_q is not None:
        if baseline_q not in (None, nominal_q):
            raise ValueError(f"Baseline nominal Q {baseline_q} != requested {nominal_q}")
        if arm_q not in (None, nominal_q):
            raise ValueError(f"Arm nominal Q {arm_q} != requested {nominal_q}")
        resolved_q = nominal_q
    else:
        resolved_q = baseline_q if baseline_q is not None else arm_q
    baseline_by_index = {row["index"]: row for row in baseline_rows}
    arm_by_index = {row["index"]: row for row in arm_rows}
    datasets = sorted({case["dataset"] for case in cases})
    summary, paired = [], []
    for window in WINDOWS:
        window_cases = select_window_cases(cases, window)
        for scope in ("all", *datasets):
            summary.append(_window_row(window_cases, baseline_by_index, arm_by_index,
                                       window, scope, baseline_model,
                                       baseline_label, arm_label, resolved_q))
        for case in window_cases:
            paired.append(_paired_case_row(case, baseline_by_index[case["index"]],
                                           arm_by_index[case["index"]], window,
                                           baseline_model, baseline_label, arm_label,
                                           resolved_q))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paired_out = Path(paired_out) if paired_out else out.with_name(out.stem + "-paired.csv")
    audit_out = Path(audit_out) if audit_out else out.with_name(out.stem + "-audit.json")
    paired_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    table(out, summary)
    table(paired_out, paired)
    audit = _audit(baseline, arm, cases_path, cases, baseline_model, baseline_label,
                   arm_label, baseline_rows, arm_rows, baseline_q, arm_q, resolved_q)
    tmp = audit_out.with_name(audit_out.name + ".tmp")
    tmp.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(audit_out)
    return summary, paired, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", "--native", dest="baseline", type=Path, required=True,
                        help="Baseline zero/control request JSON/JSONL file or run folder")
    parser.add_argument("--arm", type=Path, required=True,
                        help="Arm request JSON/JSONL file or run folder")
    parser.add_argument("--cases", type=Path, required=True,
                        help="Registered frozen90 samples snapshot")
    parser.add_argument("--out", type=Path, required=True,
                        help="Summary CSV path")
    parser.add_argument("--paired-out", type=Path,
                        help="Per-case paired CSV (default: <out>-paired.csv)")
    parser.add_argument("--audit-out", type=Path,
                        help="Audit JSON (default: <out>-audit.json)")
    parser.add_argument("--baseline-policy", "--native-policy", dest="baseline_policy",
                        help="Baseline policy name when the input contains several")
    parser.add_argument("--arm-policy",
                        help="Arm policy name when the input contains several")
    parser.add_argument("--nominal-q", type=int,
                        help="Require this nominal Q when policy names do not encode it")
    args = parser.parse_args(argv)
    summary, paired, audit = score_files(
        args.baseline, args.arm, args.cases, args.out,
        paired_out=args.paired_out, audit_out=args.audit_out,
        baseline_policy=args.baseline_policy, arm_policy=args.arm_policy,
        nominal_q=args.nominal_q)
    print(f"model={audit['model']} windows={len(summary)} paired_rows={len(paired)} "
          f"baseline={audit['baseline_policy']} arm={audit['arm_policy']} "
          f"q={audit['nominal_q']['resolved']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
