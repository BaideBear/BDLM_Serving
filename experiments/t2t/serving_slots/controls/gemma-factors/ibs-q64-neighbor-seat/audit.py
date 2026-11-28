"""Reconstruct row eligibility on the archived A0 paid IBS Q64 trace (no inference)."""
import argparse
import ast
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import types

import numpy as np

B, Q = 256, 64
PILOT = [7, 13, 25, 33, 55, 71]
PINS = {
    "raw": "2741ef9c07b95145f49378f35a53ccd317abf35ae582735169eef4d7ac30657c",
    "summary": "d3bdd6dd97235973752df513ef2e3c4832fac41cb1cf54d9115e1e0f75e0344e",
    "sequence": "99c80047d517113579e04fd57443471a759a738dc30b206ba1342e0d4f9b3da2",
    "limits": "842f078e06b5f79b336165b060d1212e947e01182e167379a1d1dbb9f62b5376",
    "runtime": "a09d04477361cb2a260dac85ce2195c0bfee9218306b7cfb96d7ddf4f1dba9af",
}


def check(ok, detail):
    if not ok:
        raise ValueError(detail)


def pinned(path, key):
    data = path.read_bytes()
    check(sha256(data).hexdigest() == PINS[key], f"Unexpected {key}: {path}")
    return data


def load_planner(sequence, limits):
    # Only the pinned NumPy frontier path is executed. Host arrays replace the
    # allocation-only .new() method; any Tensor-dependent branch fails closed.
    names = {"ProcessingPlan", "DelayedCacheState"}
    classes = [n for n in ast.parse(sequence).body if isinstance(n, ast.ClassDef) and n.name in names]
    funcs = [n for n in ast.parse(limits).body if isinstance(n, ast.FunctionDef) and n.name == "limited_rows"]
    check({n.name for n in classes} == names and len(funcs) == 1, "Missing replay source")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *classes, *funcs], type_ignores=[]))
    # dataclasses resolves annotations through the actual module.
    import sys
    mod = types.ModuleType("q64_pinned_frontier")
    sys.modules[mod.__name__] = mod
    mod.__dict__.update(np=np, dataclass=dataclass, DLLM_MASKED=0)
    exec(compile(tree, "<pinned Optimus planner>", "exec"), mod.__dict__)
    return mod


def replay(arm, planner):
    out = Counter()
    head = -1
    trigger = arm["trigger"]
    terminal_tick = arm["payment"]["tick"] if arm["payment"] else None
    origin = len(arm["encoder"][0][0])
    trace, states, decoder = (arm[k] for k in ("trace", "states", "decoder"))
    check(len(trace) == len(states) == len(decoder) == arm["bill"]["D"], "Physical length")
    for tick, (row, state, dec) in enumerate(zip(trace, states, decoder)):
        start = row["block"] * B
        if row["block"] != head:
            check(row["block"] == head+1 and (head == -1 or ctl.cached_prefix_len == B and not dirty),
                  "Bad block transition")
            head = row["block"]
            ctl = planner.DelayedCacheState(
                B, B, None, chunked_decode_enabled=True, chunk_size=Q, _full_indices_np=np.arange(B))
            prior = np.zeros(B, dtype=np.uint8)
            dirty, previous_writes, seen, initialized = set(), set(), set(), set()
        out["decoder_calls"] += 1
        out["decoder_rows"] += row["q"]
        check(row["tick"] == tick and row["offer"] == Q and row["cross"] == 0, "Wrong tick or offer")
        buf = np.empty(B, dtype=np.int64)
        n, _ = ctl.fill_processing_buffers(buf, 0, prior, chunk_size_override=Q)
        candidate = buf[:n].copy()
        dirty_mask = np.isin(np.arange(B), list(dirty))
        normal = planner.limited_rows(candidate, prior, dirty_mask, Q).tolist()
        full_window = set(range(ctl.cached_prefix_len, B))
        masks = {p for p in full_window if not prior[p]}
        rows = [p-start for p in row["positions"]]
        terminal = tick == terminal_tick
        expected = sorted(p-start for p in trigger["debt"]) if terminal else normal
        check(n == row["proposed_q"] and rows == expected and (not terminal or set(rows) <= set(normal))
              and len(rows) == row["q"] == len(dec["ids"]) == len(dec["pos"]) and 0 < len(rows) <= Q
              and dec["pos"] == [origin+p for p in row["positions"]]
              and row["visible_end"] == start + max(rows) + 1, "Exact row reconstruction failed")
        selected = set(rows)
        initialized.update(rows)
        check(initialized.issuperset(range(max(rows)+1)), "Physical KV hole")
        first = min(masks, default=None)
        donors = selected - dirty - {first, max(rows)}
        # The outer-cap drop set is distinct from positions the upstream
        # prefix planner omitted. Audit the wider sets separately, never run them.
        if terminal:
            out["terminal_payments"] += 1
        else:
            out["ordinary_calls"] += 1
            if len(rows) == Q:
                out["ordinary_full_width_calls"] += 1
            cap_dropped = (set(candidate) - selected) & masks
            if n > Q:
                out["hardcap_binding_calls"] += 1
                if not masks:
                    out["binding_no_mask_calls"] += 1
                if len(dirty) >= Q:
                    out["binding_full_dirty_calls"] += 1
                check(len(set(candidate) & masks) <= 1, "Unexpected upstream second MASK")
                check(not cap_dropped or (cap_dropped == {first} and not donors), "Unexpected free swap")
                if cap_dropped:
                    out["binding_dropped_mask_calls"] += 1
                if cap_dropped & seen:
                    out["binding_dropped_measured_calls"] += 1
                if cap_dropped & seen and donors:
                    out["binding_has_clean_pair_before_rank_or_neighbor"] += 1
            resident = (masks - selected) & seen & set(range(max(rows)))
            if n > Q and resident:
                out["binding_resident_omitted_calls_ignoring_upstream"] += 1
                if donors:
                    out["binding_resident_donor_calls_ignoring_upstream"] += 1
            if len(rows) == Q and resident:
                out["full_width_resident_omitted_calls_ignoring_upstream"] += 1
                # Superset: don't test ordinary/special IDs or age ranking.
                paid_neighbors = (previous_writes - dirty) | (previous_writes & selected)
                neighbor = {p for p in resident if {p-1, p+1} & paid_neighbors}
                if donors and neighbor:
                    out["full_width_neighbor_pairs_ignoring_upstream_rank_and_special"] += 1
        sampled = sorted(p for p in selected if not prior[p])
        check((dec["sample"]["rows"] if dec["sample"] else []) == sampled, "Wrong sampled history")
        seen.update(sampled)
        dirty -= selected
        frontier = prior.copy()
        if dirty:
            frontier[list(dirty)] = 0
            check(not sampled, "Sampling with unserved debt")
        previous_writes = {p-start for p in row["admitted"]}
        check(previous_writes <= set(sampled) and len(previous_writes) == row["accepted"], "Wrong admissions")
        prior[list(previous_writes)] = 1
        dirty |= previous_writes
        ctl.update_from_mask(frontier, mark_ready_if_all_left_non_mask=True, dllm_mask_cur=prior)
        check(ctl.cached_prefix_len == row["ready"] == state["ready"] and
              sorted(start+p for p in dirty) == state["debt"], "Wrong frontier or dirty history")
    check(not dirty and out["decoder_rows"] == arm["bill"]["Q"], "End-of-run physical bill")
    check(arm["bill"]["F"] == len(arm["events"])-1 == out["decoder_calls"]+arm["bill"]["E"],
          "Generation bill or prompt prefill mismatch")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("snapshot", type=Path, help="Archived gemma-ibs512-1127 source")
    a = parser.parse_args()
    seq = a.snapshot / "third_party/Optimus/engine/lmdeploy/pytorch/strategies/dllm/sequence.py"
    runtime = a.snapshot / "bdlm_serving/runtime"
    planner = load_planner(pinned(seq, "sequence"), pinned(runtime / "optimus_limits.py", "limits"))
    pinned(runtime / "diffusiongemma_optimus.py", "runtime")
    summary = json.loads(pinned(a.summary, "summary"))
    records = [json.loads(line[8:]) for line in pinned(a.raw, "raw").splitlines() if line.startswith(b"RECEIPT ")]
    cases = {rec["index"]: rec["arms"]["paid"] for rec in records if rec["kind"] == "case"}
    check(len(cases) == sum(r["kind"] == "case" for r in records) == 90 and sorted(cases) == list(range(90)),
          "Incomplete frozen cohort")
    check(records[0]["q"] == summary["q"] == Q and records[0]["gpu"] == summary["gpu"]
          and records[0]["optimus_files"]["sequence.py"] == PINS["sequence"], "Wrong history registration")
    compact = {r["index"]: r["arms"]["paid"] for r in summary["cases"]}
    checks = {}
    for i, arm in cases.items():
        check(all(arm[k] == compact[i][k] for k in compact[i]), "Wrong compact receipt")
        checks[i] = replay(arm, planner)
    metrics = sorted(set(k for counter in checks.values() for k in counter) | {
        "binding_dropped_mask_calls", "binding_dropped_measured_calls", "binding_no_mask_calls",
        "binding_full_dirty_calls", "binding_has_clean_pair_before_rank_or_neighbor",
        "binding_resident_omitted_calls_ignoring_upstream", "binding_resident_donor_calls_ignoring_upstream",
        "full_width_resident_omitted_calls_ignoring_upstream",
        "full_width_neighbor_pairs_ignoring_upstream_rank_and_special"})
    group = {}
    for name, indices in (("frozen90", list(range(90))), ("fixed_six", PILOT)):
        counts = {k: dict(total=sum(checks[i][k] for i in indices),
                               requests=sum(bool(checks[i][k]) for i in indices)) for k in metrics}
        group[name] = dict(counts=counts, historical_paid_baseline=dict(requests=len(indices),
            correct=sum(cases[i]["correct"] for i in indices), N=sum(cases[i]["N"] for i in indices),
            **{k: sum(cases[i]["bill"][k] for i in indices) for k in ("D", "E", "F", "Q")}))
    print(json.dumps(dict(schema="gemma-ibs-q64-neighbor-seat-audit-v1", pins=PINS,
                          reference_pid=records[0]["pid"], reference_gpu=records[0]["gpu"],
                          fixed_indices=PILOT, cohorts=group,
                          per_fixed_case={i: {k: checks[i][k] for k in metrics} for i in PILOT}),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
