"""Formal frozen90 external-offer runner for the LLaDA2.1 TopTransitionKernel.

This module is a self-contained runner and reporter.  It does not modify the
pinned single-block or dual-block executors, the protected launcher, or any
existing queue.  It runs four arms by default:

    toptrace_32_32_64_64_fused   (fuse_growth=True)
    toptrace_32_32_64_64_split   (fuse_growth=False)
    toptrace_64_64_32_32_fused
    toptrace_64_64_32_32_split

Named traces: fixed32, fixed64, llhh, hhll and wave.  A comma-separated 32/64
sequence is also accepted.  The protected launcher can select traces with the
space-separated TOP_OFFER_TRACES environment variable.  Each request is driven by the same request-local
external offer clock for the fused and split arms.  The offer is a function of the generation-forward tick
only; it never reads logits, task identity, or the score.  Prefill is outside
the clock.  Commit and reencode each consume one offer and one real forward.

The runner reuses the frozen90 task loader, the online package/verify_work
accounting, the pinned checkpoint manifest, and the TopTransitionKernel.  It
writes request files in the standard package shape plus:

    run.json
    complete-<policy>.json       (formal 90-request GPU/2048 runs only)
    qualified.json               (trace-contract marker; not a historical reproduction)
    smoke.json                   (limited, CPU, or non-2048 runs; never formal)
    top-transition-summary.csv   (standard three metrics per arm)
    top-transition-pairs.csv     (per-request fused/split pairs)
    top-transition-audit.json    (fail-closed audit and exposure counts)

The per-request files are readable by tools.prefix_cohorts.read_request_rows.
A policy branch for toptrace_* is still needed before normalize_request can
score them; until then the self-contained report above is the qualified
report.  No old baseline is re-run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

from experiments.t2t.serving_slots.expanded_handoff import SAMPLES as SAMPLES_SHA
from experiments.t2t.serving_slots.expanded_handoff import SHA as LAUNCHER_SHA
from tools.run_t2t import durable

BLOCK = 32
PAIR = 64
MODEL = "llada21"
SUITE = "frozen90"
GEN_LENGTH = 2048
VARIANTS = ("fused", "split")
DEFAULT_TRACES = ("llhh", "hhll")
TRACES = {
    "fixed32": (32,),
    "fixed64": (64,),
    "llhh": (32, 32, 64, 64),
    "hhll": (64, 64, 32, 32),
    "wave": (32, 64, 32, 32, 64),
}
ACTIONS = {
    "head_bootstrap", "pair_bootstrap", "fused_growth", "split_growth",
    "pair_denoise", "head_denoise", "commit_head", "promote_verify",
}
SUMMARY_COLUMNS = (
    "policy", "trace", "variant", "requests", "correct", "macro_tpf",
    "token_efficiency", "N", "F", "Q", "physical_Q", "max_q",
    "growth_requests", "fused_growth_forwards", "split_growth_forwards",
    "refresh_only_rows", "overflow_rows", "max_overflow", "sampled_q",
)
PAIR_COLUMNS = (
    "trace", "index", "dataset", "task_id", "seed", "fused_policy", "split_policy",
    "fused_correct", "split_correct", "fused_N", "split_N", "fused_F", "split_F",
    "fused_Q", "split_Q", "fused_tpf", "split_tpf", "same_output",
    "same_offers_prefix", "delta_correct", "delta_tpf", "delta_Q", "delta_F",
)


def read(path):
    return json.loads(Path(path).read_text())


def policy_name(trace, variant):
    return "toptrace_" + "_".join(map(str, trace)) + "_" + variant


def resolve_traces(values):
    """Resolve named traces or comma-separated 32/64 offer sequences."""
    if not values:
        values = list(DEFAULT_TRACES)
    resolved, seen = [], set()
    for value in values:
        if value in TRACES:
            trace = TRACES[value]
        else:
            try:
                trace = tuple(int(item) for item in str(value).split(","))
            except ValueError as error:
                raise ValueError(f"Unknown offer trace: {value!r}") from error
        if not trace or any(q not in (BLOCK, PAIR) for q in trace):
            raise ValueError(f"Offer trace must contain only 32/64: {value!r}")
        if trace in seen:
            raise ValueError(f"Duplicate offer trace: {value!r}")
        seen.add(trace)
        resolved.append(trace)
    return tuple(resolved)


def resolve_variants(values):
    if not values:
        return VARIANTS
    if len(set(values)) != len(values) or any(v not in VARIANTS for v in values):
        raise ValueError(f"Variants must be unique and drawn from {VARIANTS}")
    return tuple(values)


def resolve_args_traces(args):
    """Read --trace, or the protected-launcher TOP_OFFER_TRACES environment."""
    values = getattr(args, "trace", None)
    if not values:
        values = os.environ.get("TOP_OFFER_TRACES", "").split()
    return resolve_traces(values)


def offer(trace):
    """Return the request-local external offer clock for one trace."""
    values = tuple(trace)
    if not values or any(q not in (BLOCK, PAIR) for q in values):
        raise ValueError("Invalid offer trace")

    def next_budget(info):
        return values[info.tick % len(values)]

    return next_budget


def top_metadata(events, trace, variant):
    generation = [event for event in events if event["phase"] != "prefill"]
    actions = Counter(event.get("top_action") for event in generation)
    return dict(
        trace=list(trace), variant=variant, fuse_growth=(variant == "fused"),
        offers=[event["q_budget"] for event in generation],
        offer_ticks=len(generation), top_actions=dict(actions),
        growth_forwards=actions.get("fused_growth", 0) + actions.get("split_growth", 0),
        refresh_only_rows=sum(len(event.get("refresh_only_positions", []))
                              for event in generation),
        overflow_rows=sum(int(event.get("overflow", 0)) for event in generation),
        max_overflow=max((int(event.get("overflow", 0)) for event in generation), default=0),
        sampled_q=sum(int(event.get("sampled_q", 0)) for event in generation),
    )


def audit_top(row, trace, fuse_growth, *, policy=None):
    """Fail-closed audit of one TopTransition request against its offer clock."""
    from bdlm_serving.methods.t2t.protocol import verify_work

    trace = tuple(trace)
    if not trace or any(q not in (BLOCK, PAIR) for q in trace):
        raise ValueError("Invalid offer trace")
    if row.get("model") != MODEL:
        raise ValueError("Wrong model")
    if policy is not None and row.get("policy") != policy:
        raise ValueError("Wrong policy")
    if row.get("status") != "success":
        raise ValueError("Request did not succeed")
    if not isinstance(row.get("output_ids"), list) or not isinstance(row.get("output_text"), str):
        raise ValueError("Missing output")
    if not isinstance(row.get("eos"), bool) or not isinstance(row.get("truncated"), bool):
        raise ValueError("Missing stop flags")
    if row["eos"] == row["truncated"]:
        raise ValueError("EOS and truncation must be exclusive")
    if not isinstance(row.get("output_tokens"), int) or row["output_tokens"] < 0:
        raise ValueError("Invalid output token count")
    verify_work(row)
    events = row.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("Missing event stream")
    generation = [event for event in events if event.get("phase") != "prefill"]
    prefill = [event for event in events if event.get("phase") == "prefill"]
    if not generation:
        raise ValueError("No generation events")
    if row.get("generation_forwards") != len(generation):
        raise ValueError("generation_forwards disagrees with the event stream")
    q_sum = sum(int(event["physical_q"]) for event in generation)
    if row.get("generation_query_rows") != q_sum:
        raise ValueError("generation_query_rows disagrees with the event stream")
    if row.get("generation_physical_padded_rows") != q_sum:
        raise ValueError("physical Q disagrees with the event stream")
    if row.get("max_q") != max(int(event["physical_q"]) for event in generation):
        raise ValueError("max_q disagrees with the event stream")
    if row.get("prefill_rows") != sum(int(event["physical_q"]) for event in prefill):
        raise ValueError("prefill_rows disagrees with the event stream")
    stages, queries = Counter(), Counter()
    for event in generation:
        stages[event["stage"]] += 1
        queries[event["stage"]] += int(event["physical_q"])
    if row.get("forwards_by_phase") != dict(stages):
        raise ValueError("forwards_by_phase disagrees with the event stream")
    if row.get("q_by_phase") != dict(queries):
        raise ValueError("q_by_phase disagrees with the event stream")
    if row.get("q_histogram") != dict(Counter(str(event["physical_q"]) for event in generation)):
        raise ValueError("q_histogram disagrees with the event stream")
    if row.get("t2t_edits") != sum(int(event.get("edited", 0)) for event in generation):
        raise ValueError("t2t_edits disagrees with the event stream")
    meta = row.get("top_transition")
    if meta is not None:
        if meta.get("trace") != list(trace) or meta.get("variant") not in VARIANTS:
            raise ValueError("top_transition metadata has the wrong trace or variant")
        if meta.get("fuse_growth") != (meta.get("variant") == "fused"):
            raise ValueError("top_transition fuse_growth flag disagrees with the variant")
    actions, offers = Counter(), Counter()
    refresh_only_rows = overflow_rows = sampled_q = 0
    max_overflow = 0
    context_version = -1
    block = -1
    for tick, event in enumerate(generation):
        expected = trace[tick % len(trace)]
        if event.get("offer_tick") != tick:
            raise ValueError(f"offer_tick {event.get('offer_tick')} != {tick}")
        if event.get("q_budget") != expected or event.get("nominal_q") != expected:
            raise ValueError(f"q_budget {event.get('q_budget')} != {expected}")
        previous = None if tick == 0 else trace[(tick - 1) % len(trace)]
        if event.get("previous_offer") != previous:
            raise ValueError(f"previous_offer {event.get('previous_offer')} != {previous}")
        q = int(event["physical_q"])
        positions = event.get("absolute_positions")
        if not isinstance(positions, list) or not positions:
            raise ValueError("Missing absolute positions")
        if positions != sorted(set(positions)):
            raise ValueError("Positions are not sorted unique")
        if event.get("F") != 1 or q != len(positions) or q != event.get("actual_q"):
            raise ValueError("Physical Q disagrees with positions")
        if not event.get("layer_physical_q") or any(int(v) != q for v in event["layer_physical_q"]):
            raise ValueError("Per-layer Q disagrees with the event")
        if not 0 < q <= PAIR:
            raise ValueError("Physical Q outside the top-transition ceiling")
        if event.get("q_tokens") != q:
            raise ValueError("q_tokens disagrees with physical Q")
        if event.get("q_start") != positions[0] or event.get("q_end") != positions[-1] + 1:
            raise ValueError("q_start/q_end disagree with positions")
        if min(positions) < event["block"] * BLOCK or max(positions) >= (event["block"] + 2) * BLOCK:
            raise ValueError("Positions escape the active pair")
        if event.get("prefix_kv_length") != event["block"] * BLOCK:
            raise ValueError("prefix_kv_length disagrees with the head block")
        visible = event.get("visible_kv")
        if (not isinstance(visible, int) or visible % BLOCK
                or not max(positions) + 1 <= visible <= (event["block"] + 2) * BLOCK):
            raise ValueError("visible_kv disagrees with the active pair")
        if event["block"] < block:
            raise ValueError("Head block moved backwards")
        block = event["block"]
        if event.get("context_version", 0) < context_version:
            raise ValueError("context_version moved backwards")
        context_version = event.get("context_version", 0)
        refresh = event.get("refresh_only_positions", [])
        if len(refresh) != len(set(refresh)) or not set(refresh) <= set(positions):
            raise ValueError("refresh_only_positions escape the projected rows")
        if not set(refresh) <= set(event.get("refreshed_pending", [])) <= set(positions):
            raise ValueError("refreshed_pending disagrees with the projected rows")
        if int(event.get("overflow", 0)) != max(0, q - expected):
            raise ValueError("overflow disagrees with the offer")
        if int(event["overflow"]) > len(refresh):
            raise ValueError("overflow exceeds refresh-only rows")
        expected_sampled = 0 if event["phase"] == "commit" else q - len(refresh)
        if int(event.get("sampled_q", -1)) != expected_sampled:
            raise ValueError("sampled_q disagrees with the physical rows")
        if event["phase"] != "commit" and int(event.get("effective_q", -1)) != expected_sampled:
            raise ValueError("effective_q disagrees with sampled rows")
        action = event.get("top_action")
        if action not in ACTIONS:
            raise ValueError(f"Unknown top_action: {action}")
        if action == "fused_growth" and not fuse_growth:
            raise ValueError("fused_growth in a split arm")
        if action == "split_growth" and fuse_growth:
            raise ValueError("split_growth in a fused arm")
        if action == "fused_growth" and q != PAIR:
            raise ValueError("Fused growth must project a full pair")
        if action == "split_growth" and q != BLOCK + len(refresh):
            raise ValueError("Split growth must project one tail block plus refresh-only head rows")
        if action in ("fused_growth", "split_growth", "head_bootstrap", "pair_bootstrap"):
            if event["phase"] != "denoise" or event["stage"] != "bootstrap":
                raise ValueError("Growth/bootstrap phase or stage is wrong")
        elif action in ("pair_denoise", "head_denoise"):
            if event["phase"] != "denoise" or event["stage"] != "selected":
                raise ValueError("Denoise phase or stage is wrong")
        elif action == "commit_head":
            if event["phase"] != "commit" or event["stage"] != "commit":
                raise ValueError("Commit phase or stage is wrong")
        elif action == "promote_verify":
            if event["phase"] != "denoise" or event["stage"] != "reencode":
                raise ValueError("Promotion phase or stage is wrong")
        if bool(event.get("store_kv")) != (action == "commit_head"):
            raise ValueError("store_kv disagrees with the action")
        actions[action] += 1
        offers[expected] += 1
        refresh_only_rows += len(refresh)
        overflow_rows += int(event["overflow"])
        max_overflow = max(max_overflow, int(event["overflow"]))
        sampled_q += int(event["sampled_q"])
    if meta is not None:
        if meta.get("offers") != [event["q_budget"] for event in generation]:
            raise ValueError("top_transition offers disagree with the event stream")
        if meta.get("top_actions") != dict(actions):
            raise ValueError("top_transition actions disagree with the event stream")
    return dict(
        F=len(generation), Q=q_sum, physical_Q=q_sum,
        max_q=max(int(event["physical_q"]) for event in generation),
        offers=dict(offers), top_actions=dict(actions),
        growth_requests=int(any(a in ("fused_growth", "split_growth") for a in actions)),
        fused_growth_forwards=actions.get("fused_growth", 0),
        split_growth_forwards=actions.get("split_growth", 0),
        refresh_only_rows=refresh_only_rows, overflow_rows=overflow_rows,
        max_overflow=max_overflow, sampled_q=sampled_q,
    )


def summarize(rows):
    if not rows:
        raise ValueError("No rows")
    count = len(rows)
    n = sum(int(row["output_tokens"]) for row in rows)
    q = sum(int(row["generation_query_rows"]) for row in rows)
    physical = sum(int(row["generation_physical_padded_rows"]) for row in rows)
    return dict(
        requests=count,
        correct=sum(int(bool(row["correct"])) for row in rows),
        macro_tpf=sum(row["output_tokens"] / row["generation_forwards"] for row in rows) / count,
        token_efficiency=n / physical, N=n,
        F=sum(int(row["generation_forwards"]) for row in rows), Q=q,
        physical_Q=physical,
        max_q=max(int(row["max_q"]) for row in rows),
    )


def aggregate_audits(audits):
    if not audits:
        raise ValueError("No audits")
    out = dict(
        F=sum(a["F"] for a in audits), Q=sum(a["Q"] for a in audits),
        physical_Q=sum(a["physical_Q"] for a in audits),
        max_q=max(a["max_q"] for a in audits),
        growth_requests=sum(a["growth_requests"] for a in audits),
        fused_growth_forwards=sum(a["fused_growth_forwards"] for a in audits),
        split_growth_forwards=sum(a["split_growth_forwards"] for a in audits),
        refresh_only_rows=sum(a["refresh_only_rows"] for a in audits),
        overflow_rows=sum(a["overflow_rows"] for a in audits),
        max_overflow=max(a["max_overflow"] for a in audits),
        sampled_q=sum(a["sampled_q"] for a in audits),
    )
    out["offers"] = dict(sum((Counter(a["offers"]) for a in audits), Counter()))
    out["top_actions"] = dict(sum((Counter(a["top_actions"]) for a in audits), Counter()))
    return out


def build_report(rows_by_policy, arms, meta):
    summary, pairs, pair_audit = [], [], {}
    for arm in arms:
        policy, trace, variant = arm["policy"], tuple(arm["trace"]), arm["variant"]
        rows = rows_by_policy[policy]
        audits = [audit_top(row, trace, variant == "fused", policy=policy) for row in rows]
        package = summarize(rows)
        audit = aggregate_audits(audits)
        for key in ("F", "Q", "physical_Q", "max_q"):
            if package[key] != audit[key]:
                raise ValueError(f"{policy}: {key} disagrees between package and audit")
        row = dict(package)
        row.update(audit)
        row.update(policy=policy, trace=",".join(map(str, trace)), variant=variant)
        summary.append(row)
    by_trace = {}
    for arm in arms:
        by_trace.setdefault(tuple(arm["trace"]), {})[arm["variant"]] = arm["policy"]
    for trace, variants in by_trace.items():
        if set(variants) != set(VARIANTS):
            continue
        fused_rows = rows_by_policy[variants["fused"]]
        split_rows = rows_by_policy[variants["split"]]
        if len(fused_rows) != len(split_rows):
            raise ValueError("Fused/split request counts differ")
        rescued = lost = both = neither = same_output = same_offers = 0
        delta_correct = delta_q = delta_f = 0
        delta_tpf = 0.0
        for fused, split in zip(fused_rows, split_rows, strict=True):
            if any(fused[key] != split[key] for key in ("index", "dataset", "task_id", "seed")):
                raise ValueError("Fused/split identity mismatch")
            offers_fused = fused["top_transition"]["offers"]
            offers_split = split["top_transition"]["offers"]
            common = min(len(offers_fused), len(offers_split))
            prefix = offers_fused[:common] == offers_split[:common]
            if not prefix:
                raise ValueError("Fused/split offer clocks diverge")
            same_offers += int(prefix)
            same_output += int(fused["output_ids"] == split["output_ids"])
            if split["correct"] and not fused["correct"]:
                rescued += 1
            elif fused["correct"] and not split["correct"]:
                lost += 1
            elif fused["correct"] and split["correct"]:
                both += 1
            else:
                neither += 1
            delta_correct += int(bool(split["correct"])) - int(bool(fused["correct"]))
            delta_tpf += split["tpf"] - fused["tpf"]
            delta_q += split["generation_query_rows"] - fused["generation_query_rows"]
            delta_f += split["generation_forwards"] - fused["generation_forwards"]
            pairs.append(dict(
                trace=",".join(map(str, trace)), index=fused["index"],
                dataset=fused["dataset"], task_id=fused["task_id"], seed=fused["seed"],
                fused_policy=variants["fused"], split_policy=variants["split"],
                fused_correct=int(bool(fused["correct"])),
                split_correct=int(bool(split["correct"])),
                fused_N=fused["output_tokens"], split_N=split["output_tokens"],
                fused_F=fused["generation_forwards"], split_F=split["generation_forwards"],
                fused_Q=fused["generation_query_rows"], split_Q=split["generation_query_rows"],
                fused_tpf=fused["tpf"], split_tpf=split["tpf"],
                same_output=int(fused["output_ids"] == split["output_ids"]),
                same_offers_prefix=int(prefix),
                delta_correct=int(bool(split["correct"])) - int(bool(fused["correct"])),
                delta_tpf=split["tpf"] - fused["tpf"],
                delta_Q=split["generation_query_rows"] - fused["generation_query_rows"],
                delta_F=split["generation_forwards"] - fused["generation_forwards"],
            ))
        pair_audit[",".join(map(str, trace))] = dict(
            requests=len(fused_rows), rescued=rescued, lost=lost,
            both_correct=both, both_wrong=neither, delta_correct=delta_correct,
            delta_tpf_mean=delta_tpf / len(fused_rows), delta_Q=delta_q, delta_F=delta_f,
            same_output=same_output, same_offers_prefix=same_offers,
        )
    audit = dict(
        model=MODEL, suite=SUITE, samples_sha256=meta.get("samples_sha256"),
        requests=meta.get("requests"), gen_length=meta.get("gen_length"),
        arms={row["policy"]: {key: row[key] for key in (
            "requests", "correct", "macro_tpf", "token_efficiency", "N", "F", "Q",
            "physical_Q", "max_q", "growth_requests", "fused_growth_forwards",
            "split_growth_forwards", "refresh_only_rows", "overflow_rows",
            "max_overflow", "sampled_q", "offers", "top_actions")} for row in summary},
        pairs=pair_audit,
        checks=dict(
            all_rows_audited=True, fused_split_offer_prefix=True,
            growth_exposed={row["policy"]: row["growth_requests"] > 0 for row in summary},
        ),
    )
    if meta.get("formal"):
        subsets = (("original30", 0, 30), ("added60", 30, 90), ("all90", 0, 90))
        audit["cohorts"] = []
        for policy, rows in rows_by_policy.items():
            if [r["index"] for r in rows] != list(range(90)):
                raise ValueError("Formal report needs the full frozen90 manifest order")
            for cohort, lo, hi in subsets:
                audit["cohorts"].append(dict(policy=policy, cohort=cohort, **summarize(rows[lo:hi])))
    return summary, pairs, audit


def write_csv(path, columns, rows):
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def write_report(out, summary, pairs, audit):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "top-transition-summary.csv", SUMMARY_COLUMNS, summary)
    write_csv(out / "top-transition-pairs.csv", PAIR_COLUMNS, pairs)
    if "cohorts" in audit:
        write_csv(out / "top-transition-cohorts.csv", tuple(audit["cohorts"][0]), audit["cohorts"])
    durable(out / "top-transition-audit.json", audit)


def execute(args, tasks, run_one, *, reference=None):
    """Run the selected tasks and write the self-contained report."""
    traces = resolve_args_traces(args)
    variants = resolve_variants(getattr(args, "variant", None))
    arms = [dict(policy=policy_name(trace, variant), trace=list(trace),
                 offers=sorted(set(trace)), variant=variant)
            for trace in traces for variant in variants]
    policies = [arm["policy"] for arm in arms]
    limit = getattr(args, "limit", None)
    selected = list(tasks) if limit is None else list(tasks)[:limit]
    if not selected:
        raise ValueError("No tasks selected")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    samples_sha = getattr(args, "samples_sha256", None)
    if samples_sha is None and getattr(args, "samples", None) and Path(args.samples).is_file():
        from bdlm_serving.evaluation.frozen import sha
        samples_sha = sha(args.samples)
    gen_length = int(getattr(args, "gen_length", GEN_LENGTH))
    device = str(getattr(args, "device", "cuda"))
    on_cuda = device.split(":", 1)[0] == "cuda"
    formal = (limit is None and len(selected) == 90
              and gen_length == GEN_LENGTH and on_cuda
              and samples_sha == SAMPLES_SHA)
    meta = dict(
        model=MODEL, suite=SUITE, samples_sha256=samples_sha,
        block=BLOCK, pair_blocks=2, max_offer=PAIR, batch_size=1,
        mode="llada_top_transition_external_offer", gen_length=gen_length,
        requests=len(selected), limit=limit, device=device, formal=formal,
        arms=arms, policies=policies,
        traces={",".join(map(str, trace)): list(trace) for trace in traces},
        variants=list(variants),
        checkpoint=str(getattr(args, "checkpoint", "")),
        reference=str(reference) if reference else "",
        note=("Same request-local external offer clock for the fused and split arms; "
              "prefill is outside the clock. Standard metrics are correct, macro TPF "
              "and pooled token efficiency; N/F/physical Q are audit counters. "
              "max_offer is the executor ceiling; each arm carries its own external "
              "trace. qualified.json is a trace-contract marker, not a reproduction "
              "of a historical endpoint; CPU, non-2048 and non-90 runs write smoke.json."),
    )
    durable(out / "run.json", meta)
    rows_by_policy = {policy: [] for policy in policies}
    for task in selected:
        for arm in arms:
            policy, trace, variant = arm["policy"], tuple(arm["trace"]), arm["variant"]
            row = run_one(task, trace=trace, variant=variant, policy=policy,
                          gen_length=gen_length)
            audit = audit_top(row, trace, variant == "fused", policy=policy)
            durable(out / f"{task['index']:02d}-{policy}.json", row)
            rows_by_policy[policy].append(row)
            print(f"{policy} {task['index']:02d} N={row['output_tokens']} "
                  f"F={row['generation_forwards']} Q={row['generation_query_rows']} "
                  f"correct={int(row['correct'])} growth={audit['growth_requests']}", flush=True)
    if formal:
        for policy in policies:
            durable(out / f"complete-{policy}.json", dict(policy=policy, requests=90))
        durable(out / "qualified.json", dict(
            model=MODEL, suite=SUITE, trace_contract_checked=True,
            completed_requests=90, gen_length=GEN_LENGTH, policies=policies))
    else:
        durable(out / "smoke.json", dict(
            model=MODEL, suite=SUITE, trace_contract_checked=True,
            completed_requests=len(selected), limit=limit, gen_length=gen_length,
            device=device, policies=policies,
            reason=("device" if not on_cuda else
                    "generation_cap" if gen_length != GEN_LENGTH else
                    "requests" if limit is not None or len(selected) != 90 else
                    "samples" if samples_sha != SAMPLES_SHA else
                    "unspecified")))
    summary, pairs, audit = build_report(rows_by_policy, arms, meta)
    write_report(out, summary, pairs, audit)
    for row in summary:
        print(f"{row['policy']} correct={row['correct']}/{row['requests']} "
              f"macro_tpf={row['macro_tpf']:.3f} "
              f"token_efficiency={row['token_efficiency']:.2%} "
              f"growth={row['growth_requests']}", flush=True)
    return summary


def resolve_reference(args):
    explicit = getattr(args, "reference", None)
    candidates = [Path(explicit)] if explicit else []
    samples = Path(args.samples)
    candidates.extend([
        samples.parent.parent / "frozen90-native" / "llada21",
        samples.parent / "frozen90-native" / "llada21",
    ])
    for candidate in candidates:
        if all((candidate / f"{index:02d}-native32.json").is_file() for index in range(90)):
            return candidate
    raise FileNotFoundError("No complete 90-task native32 prompt reference")


def validate_prompts(tok, tasks, reference):
    for task in tasks:
        prompt = tok.apply_chat_template(
            [dict(role="user", content=task["prompt"])],
            tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
        expected = read(reference / f"{task['index']:02d}-native32.json")["prompt_ids"]
        if ids != expected:
            raise ValueError(f"Changed prompt on request {task['index']:02d}")


def top_backend(args, prepare=None, device="cuda", tokenizer=None, prompt_reference=None):
    """Load the real LLaDA2.1 model and return a TopTransition request runner."""
    import torch
    import transformers.utils as utils
    if not hasattr(utils, "LossKwargs"):
        from typing import TypedDict

        class LossKwargs(TypedDict, total=False):
            num_items_in_batch: int

        utils.LossKwargs = LossKwargs
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bdlm_serving.evaluation.frozen import ROOT, sha
    from bdlm_serving.evaluation.scoring import score
    from bdlm_serving.methods.t2t.online import package
    from bdlm_serving.methods.t2t.protocol import verify_work
    from bdlm_serving.runtime.serial import install_llada_rectangular_mask
    from experiments.t2t.serving_slots.llada_top_transition import TopTransitionKernel
    from tools.run_dual_t2t import saturation_summary

    for entry in json.loads((ROOT / "benchmarks/checkpoints/llada21.json").read_text()):
        path = args.checkpoint / entry["name"]
        if path.stat().st_size != entry["bytes"] or ("sha256" in entry and sha(path) != entry["sha256"]):
            raise ValueError(f"Checkpoint mismatch: {path.name}")
    torch.set_num_threads(int(getattr(args, "threads", 0) or 1))
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    tok = tokenizer or AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    install_llada_rectangular_mask(model, allow_selected_cache=True)
    if prepare:
        for parameter in model.parameters():
            parameter.data = parameter.data.clone()
        prepare(model)
    model.to(device)
    eos = [tok.eos_token_id]
    mask = tok.convert_tokens_to_ids("<|mask|>")
    special = set(tok.all_special_ids) | set(eos) | {mask}
    cfg = dict(confidence_threshold=.5, editing_threshold=.5, max_post_steps=16)

    @torch.inference_mode()
    def run(task, *, trace, variant, policy, gen_length=GEN_LENGTH):
        prompt = tok.apply_chat_template(
            [dict(role="user", content=task["prompt"])],
            tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        expected = read(prompt_reference / f"{task['index']:02d}-native32.json")["prompt_ids"]
        if ids[0].tolist() != expected:
            raise ValueError(f"Changed prompt on request {task['index']:02d}")
        kernel = TopTransitionKernel(model, fuse_growth=(variant == "fused"))
        state = kernel.create(ids, cfg, gen_length, task["seed"], mask, eos)
        kernel.run(state, offer(trace))
        output_ids = kernel.output(state)[0].tolist()
        work = [(event["stage"], event["physical_q"]) for event in state.events
                if event["phase"] != "prefill"]
        prefill = sum(event["physical_q"] for event in state.events
                      if event["phase"] == "prefill")
        row = package(tok, output_ids, work, eos, special, prefill,
                      events=state.events,
                      t2t_edits=sum(int(event.get("edited", 0)) for event in state.events))
        row.update({key: task[key] for key in ("index", "dataset", "task_id", "seed")},
                   model=MODEL, policy=policy)
        offers = sorted(set(trace))
        if len(offers) == 1:
            row["nominal_q"] = offers[0]
        row["correct"] = bool(score(row["output_text"], task)["correct"])
        row["top_transition"] = top_metadata(state.events, trace, variant)
        row["dual"] = dict(max_offer=PAIR, pair_blocks=2, gen_length=gen_length,
                           mode="rolling_pair", external_trace=list(trace))
        row["dual_saturation"] = saturation_summary(state.events)
        verify_work(row)
        audit_top(row, trace, variant == "fused", policy=policy)
        return row

    return run


def run(args, prepare=None):
    """Formal entry: validate frozen90 identity, then execute.

    TOP_OFFER_TRACES is read when --trace is absent, so the protected holder
    can select fixed32/fixed64 without bypassing the launcher.
    """
    from bdlm_serving.evaluation.frozen import sha
    from bdlm_serving.evaluation.frozen90 import load_tasks

    tasks = load_tasks(args.samples)
    if sha(args.samples) != SAMPLES_SHA:
        raise ValueError("Unexpected frozen90 sample digest")
    if len(tasks) != 90:
        raise ValueError("Top-transition formal run requires the registered 90 tasks")
    reference = resolve_reference(args)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True, local_files_only=True)
    validate_prompts(tok, tasks, reference)
    args.samples_sha256 = SAMPLES_SHA
    args.gen_length = int(getattr(args, "gen_length", GEN_LENGTH))
    run_one = top_backend(args, prepare, device=getattr(args, "device", "cuda"),
                          tokenizer=tok, prompt_reference=reference)
    return execute(args, tasks, run_one, reference=reference)


def score(args):
    """Re-audit a completed run and rewrite the self-contained report."""
    run_dir = Path(args.run)
    meta = read(run_dir / "run.json")
    arms = meta["arms"]
    requests = int(meta["requests"])
    formal = bool(meta.get("formal"))
    if formal:
        if meta.get("samples_sha256") != SAMPLES_SHA:
            raise ValueError("Formal run is not the registered frozen90 snapshot")
        marker = read(run_dir / "qualified.json")
        if (marker.get("trace_contract_checked") is not True
                or marker.get("completed_requests") != requests
                or marker.get("policies") != meta["policies"]):
            raise ValueError("Missing or wrong trace-contract qualification")
    else:
        marker = read(run_dir / "smoke.json")
        if (marker.get("trace_contract_checked") is not True
                or marker.get("completed_requests") != requests
                or marker.get("policies") != meta["policies"]):
            raise ValueError("Missing or wrong smoke marker")
    rows_by_policy = {}
    for arm in arms:
        policy, trace, variant = arm["policy"], tuple(arm["trace"]), arm["variant"]
        rows = []
        for index in range(requests):
            row = read(run_dir / f"{index:02d}-{policy}.json")
            if row.get("policy") != policy:
                raise ValueError("Wrong request policy")
            audit_top(row, trace, variant == "fused", policy=policy)
            rows.append(row)
        rows_by_policy[policy] = rows
    summary, pairs, audit = build_report(rows_by_policy, arms, meta)
    write_report(args.out or run_dir, summary, pairs, audit)
    for row in summary:
        print(f"{row['policy']} correct={row['correct']}/{row['requests']} "
              f"macro_tpf={row['macro_tpf']:.3f} "
              f"token_efficiency={row['token_efficiency']:.2%}", flush=True)
    return summary


def launcher_main(argv):
    """Re-exec the pinned protected launcher with this module as its worker.

    TOP_OFFER_TRACES may hold a space-separated list of named traces (fixed32,
    fixed64, llhh, hhll, wave) or comma-separated 32/64 sequences.  This keeps
    fixed32/fixed64 inside the protected holder instead of bypassing it.
    """
    if argv not in (["llada21", "4"], ["llada21", "4", "worker"]):
        raise ValueError("Only the registered n164 GPU4 LLaDA entry is supported")
    samples = Path(os.environ["FROZEN90_SAMPLES"]).resolve()
    launcher = samples.parents[1] / "launch_prefix90.py"
    source = launcher.read_bytes()
    key = b"from experiments.t2t.serving_slots.llada_reference import run"
    if hashlib.sha256(source).hexdigest() != LAUNCHER_SHA or source.count(key) != 1:
        raise RuntimeError("Protected launcher SHA256 or import target changed")
    if os.environ.get("PREFIX_MODE") != "llrefs":
        raise RuntimeError("Set PREFIX_MODE=llrefs for the protected llrefs path")
    replacement = b"from experiments.t2t.serving_slots.top_transition90 import run"
    exec(compile(source.replace(key, replacement), str(launcher), "exec"), globals())


def queue(root):
    """One successor to the verified-commit GPU4 job, never a second dispatcher."""
    import subprocess
    import time
    data = root / "prefix90"
    while True:
        pred = subprocess.run(["tmux", "has-session", "-t", "=prefix90-verified-commit"],
                              capture_output=True)
        if pred.returncode == 1:
            break
        pred.check_returncode()
        time.sleep(8)
    for q in (4, 8, 16, 32):
        name = f"promote_q{q}_zero"
        if read(data / "llada21-verified-commit90" / f"complete-{name}.json") != dict(
                policy=name, requests=90):
            raise ValueError("Unfinished GPU4 predecessor")
    out = data / "llada21-top-offer32-64-v1"
    if out.exists():
        raise FileExistsError(out)
    env = dict(os.environ, FROZEN90_SAMPLES=str(data / "samples.json"), TRACE_OUT=str(out),
               PREFIX_MODE="llrefs", LL_BASELINE_POLICIES="top", TOP_OFFER_TRACES="fixed32 fixed64 llhh hhll")
    with out.with_suffix(".log").open("x") as log:
        subprocess.run([sys.executable, "-u", __file__, "llada21", "4"], env=env, cwd=root,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run_parser = sub.add_parser("run", help="Run the frozen90 external-offer arms")
    for key in ("checkpoint", "samples", "output"):
        run_parser.add_argument("--" + key, type=Path, required=True)
    run_parser.add_argument("--reference", type=Path)
    run_parser.add_argument("--trace", action="append",
                            help="Named trace (fixed32, fixed64, llhh, hhll, wave) "
                                 "or comma-separated 32/64 offers; repeatable")
    run_parser.add_argument("--variant", action="append", choices=VARIANTS)
    run_parser.add_argument("--gen-length", type=int, default=GEN_LENGTH)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--threads", type=int)
    score_parser = sub.add_parser("score", help="Re-audit and rewrite the report")
    score_parser.add_argument("--run", type=Path, required=True)
    score_parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.action == "run":
        return run(args)
    return score(args)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) == 2 and argv[0] == "queue":
        queue(Path(argv[1]).resolve())
    elif argv[:2] == ["llada21", "4"]:
        launcher_main(argv)
    else:
        main(argv)
