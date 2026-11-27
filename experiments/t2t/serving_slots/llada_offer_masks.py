"""LLaDA2.1 Q4/8/16/32 attributable budget-mask contrast on frozen90.

Fixed high-tier start, progress80-gated block opening, then a downshift to each
target.
Every open arm releases the head->tail quadrant only when the *current offer*
equals its own target. One shared causal arm runs the identical offer trace and
never opens; it is the reference for all four targets. The intervention is a
pure mask change on the existing forward geometry: no extra Q rows, no extra
K/V window, no change to commit, sampling or RNG.

Arms
----
``causal``   the shared reference. Same trace, never releases.
``open4``    releases only at Q4.
``open8``    releases only at Q8.
``open16``   releases only at Q16.
``open32``   releases only at Q32.

Trace
-----
``TRACE = (64, 64, 32, 64, 64, 16, 64, 64, 8, 64, 64, 4)``.
Each target is preceded by its own two Q64 offers, so it gets a fresh tail
debt and a real cross-block opportunity instead of competing with the other
targets for the single first low offer after one prelude. The period repeats.
The tail is admitted only by ``SpawnGatedKernel`` at a Q64 offer when measured
head progress reaches 0.8; a target offer never changes the tail geometry.
The trace and targets come from ``llada_mask_study``. ``LLADA_OFFER_MASK_STUDY``
selects ``replay`` (default) or ``fixed4``/``fixed8``/``fixed16``/``fixed32``
before worker startup; ``observe``, ``audit`` and ``pair`` bind the selected
trace and targets by default.

Row schema
----------
``offer_mask_witness`` maps ``str(target)`` to
``{first_tick, pretrace_sha256, geometry_sha256, rounds, edge_pairs,
released_rounds, released_edge_pairs}`` for all four targets. ``rounds`` and
``edge_pairs`` are **eligible** counts: ticks at the target with a real
opportunity, and the head-query x tail-key pairs that opportunity contains.
They are not opened counts. The actual release is ``released_rounds`` and
``released_edge_pairs``. ``first_tick`` is the first eligible tick, not the
first release. The causal row therefore carries the same first geometry and
pretrace hash as the open row, with ``released_rounds == 0`` and
``released_edge_pairs == 0``. ``trace`` records ``{tick, offer, opportunity,
opened, action, physical_q, mandatory_q}`` for every generation forward.
``opportunity`` is the number of head-query x tail-key pairs that the current
forward already contains and that the block-causal mask hides; ``opened`` is
the number actually released by this arm. The common scorer schema is
``witness(row)`` (eligible); ``released(row)`` exposes the dedicated opened
counts, and ``trace(row)`` exposes the per-forward trace.

Protected GPU0 entry (owner only; never run by the reviewer)
-----------------------------------------------------------
  cd /data1/project_data/gx-erqw-20261123/runtime/t2t-exchange-suite-1127
  PYTHONPATH=.:../fixed-baseline-deps ../venv/bin/python -u \
    experiments/t2t/serving_slots/llada_offer_masks.py llada21 0

Wait for the Gemma replay receipt, then take the same lease:
  PYTHONPATH=.:../fixed-baseline-deps ../venv/bin/python -u \
    experiments/t2t/serving_slots/llada_offer_masks.py queue

Offline, model-free scoring:
  ../venv/bin/python -m experiments.t2t.serving_slots.llada_offer_masks score

CPU interface check (no model, no GPU, no holder, no lease):
  ../venv/bin/python -m experiments.t2t.serving_slots.llada_offer_masks check

Launcher requirements
---------------------
``spawn_mask_gpu.gpu0_lease`` must whitelist this entry:
``entry.stem in ("spawn_mask_gpu", "commit_pair_probe", "llada_offer_masks")``.
``OUT`` must not exist. The predecessor is the Gemma replay study, whose
``returned.json`` is exactly ``{"gpu": 0, "restored": true}``. The launcher
sets ``PREFIX_MODE=llrefs``, ``LL_BASELINE_POLICIES=offer-masks``,
``TRACE_OUT=$OUT``, ``FROZEN90_SAMPLES=$DATA/samples.json`` and
``CUDA_VISIBLE_DEVICES`` to the GPU0 UUID; this module supplies the
``run(args, prepare)`` worker with the ``llrefs`` signature.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.t2t.serving_slots import llada_all_slots90 as reg
from experiments.t2t.serving_slots import spawn_mask_gpu as lease
from experiments.t2t.serving_slots import spawn_frozen as sf
from experiments.t2t.serving_slots.llada_mask_study import study

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
DATA = reg.DATA
PLAN = study(os.environ.get("LLADA_OFFER_MASK_STUDY", "replay"))
OUT = DATA / PLAN.folder
PREV = DATA / "gemma-unified90-q32-q32_unmask32"
GPU = lease.GPU
CAUSAL = "causal"
TARGETS = PLAN.targets
TRACE = PLAN.trace
ARMS = {CAUSAL: None, **{f"open{q}": q for q in TARGETS}}
GATE = 0.8
CFG = dict(confidence_threshold=.5, editing_threshold=.5, max_post_steps=16)
COHORTS = {"original30": range(0, 30), "new60": range(30, 90), "full90": range(90)}


def witness(row):
    """Common scorer schema: target -> {first_tick, pretrace, geometry, rounds, edges}."""
    keys = ("first_tick", "pretrace_sha256", "geometry_sha256", "rounds", "edge_pairs")
    return {str(q): {k: row["offer_mask_witness"][str(q)][k] for k in keys}
            for q in TARGETS}


def trace(row):
    """Common scorer trace: one {tick, offer, opportunity, opened} per forward."""
    return [dict(tick=t["tick"], offer=t["offer"], opportunity=t["opportunity"],
                 opened=t["opened"]) for t in row["trace"]]


def released(row):
    """Dedicated opened counts; the common schema above stays eligible."""
    return {str(q): dict(released_rounds=row["offer_mask_witness"][str(q)]["released_rounds"],
                         released_edge_pairs=row["offer_mask_witness"][str(q)]["released_edge_pairs"])
            for q in TARGETS}


def input_digest(state, positions, end, info):
    """The state at a first target-tier exposure, not the first cross-block."""
    return reg.digest(dict(
        canvas=state.x[0].tolist(), write=state.last_write, query=state.last_query,
        committed=state.committed_end, context=state.context_version,
        cpu=state.rng_cpu.tolist(),
        device=state.rng_cuda.tolist() if state.rng_cuda is not None else None,
        positions=positions, visible=end, block=state.block, geometry=info,
        tick=state.offer_tick))


def target_witness(target, events, rows, prefill, input_sha256=None):
    """Summarise one target from the full events, never from the offer label."""
    ticks = [t for t in rows if t["offer"] == target]
    first = next((t for t in ticks if t["opportunity"] > 0), None)
    trajectory = reg.digest(events)
    if first is None:
        return dict(first_tick=None, pretrace_sha256=None, geometry_sha256=None,
                    rounds=0, edge_pairs=0, released_rounds=0, released_edge_pairs=0,
                    input_sha256=None, trajectory_sha256=trajectory)
    index = first["tick"] + prefill
    return dict(
        first_tick=first["tick"],
        pretrace_sha256=reg.digest(events[:index]),
        geometry_sha256=reg.digest(events[index]["mask_window"]),
        rounds=sum(t["opportunity"] > 0 for t in ticks),
        edge_pairs=sum(t["opportunity"] for t in ticks),
        released_rounds=sum(t["opened"] > 0 for t in ticks),
        released_edge_pairs=sum(t["opened"] for t in ticks),
        input_sha256=input_sha256, trajectory_sha256=trajectory)


def mask_allowed(bias, expected):
    """Return the allowed-visibility bool matrix; reject unknown mask dtypes.

    The real LLaDA path can hand the attention layer a fully visible additive
    float matrix (all zeros). Distinguish bool, integer binary and float
    additive by dtype; never infer the convention from the values.
    """
    import torch
    if bias.dtype == torch.bool:
        reg.require(torch.equal(bias, expected), "Bad effective bool mask in attention")
        return bias
    if bias.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        reg.require(bool(torch.all((bias == 0) | (bias == 1))),
                    "Non-binary integer attention mask")
        allowed = bias != 0
        reg.require(torch.equal(allowed, expected), "Bad effective binary mask in attention")
        return allowed
    if bias.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        additive = torch.zeros_like(bias).masked_fill(
            ~expected, torch.finfo(bias.dtype).min)
        reg.require(torch.equal(bias, additive), "Bad effective additive mask in attention")
        return bias == 0
    raise ValueError(f"Unsupported attention mask dtype: {bias.dtype}")


def observe(kernel, state, arm, *, trace=TRACE):
    """Bind the release to the real attention mask and record the exact quote menu."""
    import torch
    from experiments.t2t.serving_slots.llada_quote import with_quotes
    from experiments.t2t.serving_slots.mask_pair_llada import head_to_tail_mask

    target = ARMS[arm]
    calls, quotes, witnesses = [], [], []
    first_inputs = {q: None for q in TARGETS}
    current = dict(release=False, info=None, offer=None)
    first_input = None

    def swap(_module, _args, kw):
        nonlocal first_input
        mask, ids = kw["attention_mask"], kw["position_ids"]
        positions, end = ids[0].tolist(), mask.shape[-1]
        reg.require(tuple(mask.shape) == (1, 1, len(positions), end),
                    "Not a rectangular mask")
        expected = reg.geometry(state.block, state.prompt_length, positions, end)
        updated, info = head_to_tail_mask(mask, ids, state.prompt_length, state.block)
        reg.require(info == expected, "Mask hook disagrees with actual query/key geometry")
        offer = trace[state.offer_tick % len(trace)]
        release = target is not None and offer == target and info is not None
        for q in TARGETS:
            if offer == q and info is not None and first_inputs[q] is None:
                first_inputs[q] = input_digest(state, positions, end, info)
        if info and first_input is None:
            first_input = reg.digest(dict(
                canvas=state.x[0].tolist(), write=state.last_write,
                query=state.last_query, committed=state.committed_end,
                context=state.context_version, cpu=state.rng_cpu.tolist(),
                device=state.rng_cuda.tolist() if state.rng_cuda is not None else None))
        current.update(release=release, info=info, offer=offer)
        calls.append((state.block, positions, end, info, release, offer))
        if release:
            return _args, dict(kw, attention_mask=updated)
        return None

    def witness(_module, _args, kw):
        bias, pos = kw["attention_mask"][0, 0], kw["position_ids"][0]
        keys = torch.arange(bias.shape[-1], device=pos.device)
        native = pos[:, None] // 32 >= keys[None, :] // 32
        boundary = (state.block + 1) * 32
        window = ((pos >= max(boundary - 32, state.prompt_length)) & (pos < boundary))[:, None] & (
            (keys >= boundary) & (keys < boundary + 32))[None, :]
        expected = native | window if current["release"] else native
        allowed = mask_allowed(bias, expected)
        witnesses.append(int((~native & allowed).sum()))

    def budget(tick, prices):
        quotes.append(prices)
        return trace[tick.tick % len(trace)]

    inner = getattr(kernel.model, "model", kernel.model)
    swap_target = inner if type(inner).forward is not torch.nn.Module.forward else kernel.model
    attention = inner.layers[0].attention
    handle_swap = swap_target.register_forward_pre_hook(swap, with_kwargs=True)
    handle_witness = attention.register_forward_pre_hook(witness, with_kwargs=True)
    try:
        kernel.run(state, with_quotes(kernel, state, budget))
    finally:
        handle_witness.remove()
        handle_swap.remove()
    events = state.events
    prefill = int(bool(events) and events[0]["phase"] == "prefill")
    gen = events[prefill:]
    reg.require(len(quotes) == len(gen) and len(calls) == len(events) == len(witnesses),
                "Quote, hook, witness and event clocks differ")
    for event, call, seen in zip(events, calls, witnesses, strict=True):
        block, positions, end, info, release, offer = call
        reg.require((block, positions, end) == (event["block"], event["absolute_positions"],
                                                event["visible_kv"]),
                    "Model invocation differs from kernel event")
        event.update(mask_window=info, opened_edges=seen)
        reg.require(seen == (info["opened_edges"] if release else 0),
                    "Witness disagrees with execution")
    for tick, (event, prices) in enumerate(zip(gen, quotes, strict=True)):
        chosen = prices[event["q_budget"]]
        reg.require(event["physical_q"] == chosen["physical_q"]
                    and event["top_action"] == chosen["action"],
                    "Quote disagrees with the executed forward")
        event["quote"] = {k: chosen[k] for k in (
            "offer", "action", "physical_q", "mandatory_q", "required_q",
            "dense_required_q", "prefill_q")}
    rows = [dict(tick=t, offer=e["q_budget"],
                 opportunity=e["mask_window"]["opened_edges"] if e["mask_window"] else 0,
                 opened=e["opened_edges"], action=e["top_action"],
                 physical_q=e["physical_q"], mandatory_q=e["mandatory_q"])
            for t, e in enumerate(gen)]
    return dict(
        first_input_sha256=first_input, trace=rows,
        offer_mask_witness={str(q): target_witness(q, events, rows, prefill, first_inputs[q])
                            for q in TARGETS},
        offer_mask_target=target)


def audit(row, arm, *, trace=TRACE, cap=reg.CAP):
    """Reconcile the hooked geometry, physical work and witness for one row."""
    reg.require(row["policy"] == arm and row["model"] == "llada21"
                and row["status"] == "success", "Wrong arm/model/status")
    target = ARMS[arm]
    reg.require(row["offer_mask_target"] == target, "Wrong target label")
    events = row["events"]
    prompt = len(row["prompt_ids"])
    prefill = int(bool(events) and events[0]["phase"] == "prefill")
    gen = events[prefill:]
    rows = row["trace"]
    reg.require(gen and len(gen) == len(rows), "Missing generation or wrong trace length")
    reg.require(row["prefill_rows"] == (prompt // 32 * 32 if prefill else 0),
                "Wrong prompt work")
    if prefill:
        e = events[0]
        reg.require(e["phase"] == "prefill"
                    and e["absolute_positions"] == list(range(prompt // 32 * 32))
                    and e["visible_kv"] == prompt // 32 * 32 and "offer_tick" not in e,
                    "Unexpected prompt forward")
    for tick, e in enumerate(gen):
        pos, end, q = e["absolute_positions"], e["visible_kv"], e["physical_q"]
        reg.require(pos and pos == sorted(set(pos))
                    and len(pos) == e["actual_q"] == q == e["q_tokens"]
                    and 0 <= pos[0] <= pos[-1] < end and e["F"] == 1
                    and all(v == q for v in e["layer_physical_q"]),
                    "Invalid physical work")
        geom = reg.geometry(e["block"], prompt, pos, end)
        reg.require(e["mask_window"] == geom, "Wrong mask geometry")
        offer = trace[tick % len(trace)]
        reg.require(e["offer_tick"] == tick and e["nominal_q"] == e["q_budget"] == offer
                    and e["previous_offer"] == (trace[(tick - 1) % len(trace)] if tick else None),
                    "Wrong external offer clock")
        expected = geom["opened_edges"] if (target is not None and offer == target and geom) else 0
        reg.require(e["opened_edges"] == expected, "Wrong release decision")
        record = rows[tick]
        reg.require(record["tick"] == tick and record["offer"] == offer
                    and record["opportunity"] == (geom["opened_edges"] if geom else 0)
                    and record["opened"] == e["opened_edges"],
                    "Trace disagrees with the event")
        chosen = e["quote"]
        reg.require(chosen["offer"] == offer and chosen["physical_q"] == q
                    and chosen["action"] == e["top_action"], "Quote disagrees with the event")
        if "tail_admission" in e:
            reg.require(e["tail_admission"] in ("open", "defer"), "Bad tail admission")
            reg.require(e["q_budget"] == 64, "Tail admission outside Q64")
            if e["tail_admission"] == "defer":
                reg.require(e["top_action"] not in ("pair_bootstrap", "pair_denoise",
                                                    "fused_growth", "split_growth"),
                            "Deferred Q64 opened the tail")
            else:
                reg.require(e["top_action"] in ("pair_bootstrap", "fused_growth"),
                            "Open admission did not open a tail")
        debt, dense, refresh = map(set, (e["refreshed_pending"], e["dense_required_positions"],
                                         e["refresh_only_positions"]))
        reg.require(refresh <= debt <= set(pos) and e["mandatory_q"] == len(debt),
                    "Bad pending rows")
        over, dense_over = max(0, q - offer), max(0, e["dense_required_q"] - offer)
        why = {e["dense_reason"]: dense_over} if dense_over else {}
        if over > dense_over:
            why["mandatory_refresh"] = over - dense_over
        reg.require(e["overflow"] == over and e["overflow_reasons"] == why
                    and e["underfill"] == max(0, offer - q)
                    and (q <= max(offer, len(debt | dense)) if offer < 32 else over <= len(refresh)),
                    "Unexplained physical overrun")
    for q in TARGETS:
        stored = row["offer_mask_witness"][str(q)]
        reg.require(stored == target_witness(q, events, rows, prefill, stored.get("input_sha256")),
                    f"Wrong witness for target {q}")
        reg.require((stored["input_sha256"] is None) == (stored["first_tick"] is None),
                    "Missing or spurious target input hash")
    if target is None:
        reg.require(all(e["opened_edges"] == 0 for e in gen), "Causal arm opened edges")
        reg.require(all(w["released_rounds"] == 0 and w["released_edge_pairs"] == 0
                        for w in row["offer_mask_witness"].values()),
                    "Causal arm released a target")
    else:
        reg.require(all(e["opened_edges"] == 0 for e in gen if e["q_budget"] != target),
                    "Open arm released at a non-target offer")
    f, q_sum, n = len(gen), sum(e["physical_q"] for e in gen), row["output_tokens"]
    ids, eos, special = row["output_ids"], row["eos_id"], row["special_ids"]
    reg.require(bool(ids) and len(ids) <= cap and eos not in ids[:-1]
                and row["eos"] == (ids[-1] == eos) and row["truncated"] != row["eos"]
                and (row["eos"] or len(ids) == cap)
                and n == sum(t not in special for t in ids), "Wrong output/EOS convention")
    reg.require(row["generation_forwards"] == f and row["generation_query_rows"] ==
                row["generation_physical_padded_rows"] == q_sum and row["tpf"] == n/f
                and row["token_efficiency"] == n/q_sum, "Wrong packaged metrics")
    return dict(N=n, F=f, Q=q_sum, opened_edges=sum(e["opened_edges"] for e in gen),
                witness=dict(row["offer_mask_witness"]), trace=rows)


def pair(causal, trial, target, *, trace=TRACE, cap=reg.CAP):
    """Fail-closed: the causal reference and the trial must agree until release."""
    audit(causal, CAUSAL, trace=trace, cap=cap)
    audit(trial, f"open{target}", trace=trace, cap=cap)
    reg.require(all(causal[k] == trial[k] for k in (
        *reg.KEYS, "prompt_ids", "special_ids", "eos_id", "first_input_sha256")),
        "Different task or input")
    wa = causal["offer_mask_witness"][str(target)]
    wb = trial["offer_mask_witness"][str(target)]
    prefill = int(bool(causal["events"]) and causal["events"][0]["phase"] == "prefill")
    if wa["first_tick"] is None:
        reg.require(wb["first_tick"] is None and wb["released_rounds"] == 0
                    and wb["released_edge_pairs"] == 0
                    and wa["trajectory_sha256"] == wb["trajectory_sha256"],
                    "Trial released a target with no opportunity or changed the trajectory")
        skip = ("policy", "offer_mask_witness", "trace", "offer_mask_target")
        reg.require({k: v for k, v in causal.items() if k not in skip} ==
                    {k: v for k, v in trial.items() if k not in skip},
                    "Unexposed pair differs")
        return dict(target=target, exposed=False, first_tick=None, rounds=0, edge_pairs=0,
                    released_rounds=0, released_edge_pairs=0)
    tick = wa["first_tick"]
    index = tick + prefill
    reg.require(wb["first_tick"] == tick and wb["pretrace_sha256"] == wa["pretrace_sha256"]
                and wb["geometry_sha256"] == wa["geometry_sha256"]
                and wa["input_sha256"] is not None
                and wb["input_sha256"] == wa["input_sha256"]
                and wb["released_rounds"] >= 1,
                "Different first exposure or pretrace")
    reg.require(causal["events"][:index] == trial["events"][:index],
                "Different pre-exposure trajectory")
    for k in ("stage", "phase", "top_action", "block", "offer_tick", "q_budget",
              "context_version", "visible_kv", "absolute_positions", "sampled_positions",
              "refreshed_pending", "refresh_only_positions", "mask_window", "quote"):
        reg.require(causal["events"][index].get(k) == trial["events"][index].get(k),
                    f"First geometry differs: {k}")
    reg.require(trial["events"][index]["opened_edges"] > 0
                and causal["events"][index]["opened_edges"] == 0
                and wb["released_edge_pairs"] >= trial["events"][index]["opened_edges"],
                "Trial did not release at the first opportunity")
    return dict(target=target, exposed=True, first_tick=tick, rounds=wb["rounds"],
                edge_pairs=wb["edge_pairs"], released_rounds=wb["released_rounds"],
                released_edge_pairs=wb["released_edge_pairs"])


def generate(model, tok, task, prompt, arm, *, cap=reg.CAP):
    """One arm on one request: create, observe, package and audit.

    Shared by the protected worker and the CPU real-weight preflight. The
    preflight uses ``cap=96`` on frozen index 0; it is a prefix/mask/billing
    check, not a quality score.
    """
    import torch
    from bdlm_serving.methods.t2t.online import package
    from experiments.t2t.serving_slots.llada_spawn import SpawnGatedKernel

    mask = tok.convert_tokens_to_ids("<|mask|>")
    eos = tok.eos_token_id
    special = sorted(set(tok.all_special_ids) | {mask, eos})
    ids = torch.tensor([prompt], device=model.device, dtype=torch.long)
    kernel = SpawnGatedKernel(model, spawn_threshold=GATE, fuse_growth=True,
                              prefix_mode="zero")
    state = kernel.create(ids, CFG, cap, task["seed"], mask, [eos])
    extra = observe(kernel, state, arm)
    work = [(e["stage"], e["physical_q"]) for e in state.events if e["phase"] != "prefill"]
    row = package(tok, kernel.output(state)[0].tolist(), work, [eos], special,
                  sum(e["physical_q"] for e in state.events if e["phase"] == "prefill"),
                  events=state.events, prompt_ids=prompt,
                  first_input_sha256=extra["first_input_sha256"],
                  special_ids=special, eos_id=eos, policy=arm, model="llada21",
                  **{k: task[k] for k in reg.KEYS})
    row.update(offer_mask_target=extra["offer_mask_target"],
               offer_mask_witness=extra["offer_mask_witness"], trace=extra["trace"])
    audit(row, arm, cap=cap)
    return row


def run(args, prepare):
    """Protected worker: one loaded model, five arms per request, sealed 30/90."""
    import torch

    reg.require(prepare is not None and os.environ["CUDA_VISIBLE_DEVICES"] == GPU
                and args.output == OUT and args.samples == DATA/"samples.json"
                and reg.sha(args.samples) == reg.SAMPLES_SHA,
                "Not the protected offer-mask worker")
    model, tok, tasks, reference = sf.load_gpu(args, prepare)
    out = args.output
    out.mkdir(exist_ok=False)
    reg.save(out/"run.json", spec())
    files = {}
    with torch.inference_mode():
        for index in range(90):
            task = tasks[index]
            prompt = reg.read(reference / f"{index:02d}-native32.json")["prompt_ids"]
            rows = {arm: generate(model, tok, task, prompt, arm) for arm in ARMS}
            for target in TARGETS:
                pair(rows[CAUSAL], rows[f"open{target}"], target)
            for arm, row in rows.items():
                name = reg.filename(index, arm)
                reg.save(out/name, row)
                files[name] = reg.sha(out/name)
            print(f"offer-mask request {index:02d}: {','.join(ARMS)}", flush=True)
            if index in (29, 89):
                reg.save(out/f"sealed-{index+1}.json",
                         dict(run=reg.sha(out/"run.json"), files=files))
    reg.save(out/"worker-complete.json",
             dict(run_sha256=reg.sha(out/"run.json"), files=files))


def provenance():
    paths = [HERE, HERE.with_name("llada_all_slots.py"), HERE.with_name("llada_quote.py"),
             HERE.with_name("llada_top_transition.py"), HERE.with_name("llada_all_slots90.py"),
             HERE.with_name("mask_pair_llada.py"), HERE.with_name("llada_spawn.py"),
             HERE.with_name("llada_mask_study.py"), HERE.with_name("spawn_frozen.py"),
             HERE.with_name("spawn_mask_gpu.py"),
             REPO/"bdlm_serving/methods/t2t/llada_dual.py",
             REPO/"bdlm_serving/methods/t2t/llada_sparse.py",
             REPO/"bdlm_serving/runtime/serial.py",
             REPO/"bdlm_serving/evaluation/frozen90.py",
             REPO/"bdlm_serving/evaluation/scoring.py",
             REPO/"bdlm_serving/methods/t2t/online.py"]
    return {str(p.relative_to(REPO)): reg.sha(p) for p in paths}


def predecessor():
    receipt = reg.read(PREV/"returned.json")
    reg.require(receipt == {"gpu": 0, "restored": True},
                "Predecessor GPU0 study has not returned")
    run = reg.read(PREV/"run.json")
    reg.require(run["samples_sha256"] == reg.SAMPLES_SHA and run["suite"] == "frozen90"
                and run["model"] == "diffusiongemma", "Wrong predecessor study")
    return dict(run=PREV.name, returned_sha256=reg.sha(PREV/"returned.json"),
                run_sha256=reg.sha(PREV/"run.json"))


def spec():
    return dict(model="llada21", suite="frozen90", gpu=0, gpu_uuid=GPU, cap=reg.CAP,
                samples_sha256=reg.SAMPLES_SHA, launcher_sha256=reg.LAUNCHER_SHA,
                config=CFG, trace=list(TRACE), targets=list(TARGETS),
                kernel="SpawnGatedKernel", spawn_threshold=GATE, gate="progress80",
                arms={arm: dict(target=ARMS[arm], trace=list(TRACE), spawn_threshold=GATE)
                      for arm in ARMS},
                indices=list(range(90)), fuse_growth=True, prefix_mode="zero",
                predecessor=predecessor(), sources=provenance())


def sealed(out=OUT, n=90):
    reg.require(n in (30, 90), "Unknown cohort")
    manifest = reg.read(out/f"sealed-{n}.json")
    reg.require(manifest["run"] == reg.sha(out/"run.json")
                and manifest["files"] == {reg.filename(i, a): reg.sha(out/reg.filename(i, a))
                                          for i in range(n) for a in ARMS},
                "Wrong sealed cohort")
    return reg.sha(out/f"sealed-{n}.json")


def acquire():
    try:
        return lease.gpu0_lease(HERE, OUT, "offer-masks")
    except ValueError as error:
        if "Unregistered lease" in str(error):
            raise ValueError(
                "spawn_mask_gpu.gpu0_lease must whitelist llada_offer_masks: "
                "entry.stem in ('spawn_mask_gpu','commit_pair_probe','llada_offer_masks')"
            ) from error
        raise


def take_lease():
    predecessor()
    holder = acquire()
    reg.save(OUT/"returned.json", dict(
        gpu=0, uuid=GPU, pid=holder["pid"], restored=True,
        original30=sealed(OUT, 30), full90=sealed(OUT, 90)))


def protected():
    reg.require(sys.argv[1:] in (["llada21", "0"], ["llada21", "0", "worker"])
                and not sys.flags.optimize, "Wrong protected invocation")
    take_lease()


def queue():
    """Owner-only: wait for the Gemma replay receipt, then take the lease."""
    while not (PREV/"returned.json").exists():
        time.sleep(15)
    predecessor()
    subprocess.run([sys.executable, "-u", str(HERE), "llada21", "0"],
                   cwd=REPO, check=True)


def cohort_of(index):
    return "original30" if index < 30 else "new60"


def question(rows, pairs, tasks, index, target):
    a, b = rows[CAUSAL][index], rows[f"open{target}"][index]
    p = pairs[target][index]
    opportunities = [t["opportunity"] for t in a["trace"] if t["offer"] == target]
    return dict(
        cohort=cohort_of(index), index=index, dataset=tasks[index]["dataset"],
        task_id=tasks[index]["task_id"], target=target,
        exposed=p["exposed"], first_tick=p["first_tick"], rounds=p["rounds"],
        edge_pairs=p["edge_pairs"], released_rounds=p["released_rounds"],
        released_edge_pairs=p["released_edge_pairs"], target_ticks=len(opportunities),
        opportunity_ticks=sum(v > 0 for v in opportunities),
        max_opportunity=max(opportunities, default=0),
        causal_correct=a["correct"], open_correct=b["correct"],
        delta_correct=int(b["correct"]) - int(a["correct"]),
        causal_N=a["output_tokens"], causal_F=a["generation_forwards"],
        causal_Q=a["generation_query_rows"], open_N=b["output_tokens"],
        open_F=b["generation_forwards"], open_Q=b["generation_query_rows"],
        causal_tpf=a["tpf"], open_tpf=b["tpf"], delta_tpf=b["tpf"] - a["tpf"],
        causal_efficiency=a["token_efficiency"], open_efficiency=b["token_efficiency"],
        delta_efficiency=b["token_efficiency"] - a["token_efficiency"],
        same_ids=a["output_ids"] == b["output_ids"],
        rescued=not a["correct"] and b["correct"], lost=a["correct"] and not b["correct"])


def group(rows, pairs, tasks, indices, target, name):
    sub = [question(rows, pairs, tasks, i, target) for i in indices]
    exposed = [r for r in sub if r["exposed"]]

    def mean(key, values):
        return sum(r[key] for r in values) / len(values) if values else None

    return dict(
        cohort=name, target=target, requests=len(sub), exposed=len(exposed),
        triggered=sum(r["released_rounds"] > 0 for r in sub),
        rounds=sum(r["rounds"] for r in sub), edge_pairs=sum(r["edge_pairs"] for r in sub),
        released_rounds=sum(r["released_rounds"] for r in sub),
        released_edge_pairs=sum(r["released_edge_pairs"] for r in sub),
        target_ticks=sum(r["target_ticks"] for r in sub),
        opportunity_ticks=sum(r["opportunity_ticks"] for r in sub),
        max_opportunity=max((r["max_opportunity"] for r in sub), default=0),
        rescued=[r["index"] for r in sub if r["rescued"]],
        lost=[r["index"] for r in sub if r["lost"]],
        same_ids=sum(r["same_ids"] for r in sub),
        delta_correct=mean("delta_correct", sub), delta_tpf=mean("delta_tpf", sub),
        delta_efficiency=mean("delta_efficiency", sub),
        exposed_delta_correct=mean("delta_correct", exposed),
        exposed_delta_tpf=mean("delta_tpf", exposed),
        exposed_delta_efficiency=mean("delta_efficiency", exposed),
        negative_control=None if exposed else dict(
            identical=sum(r["same_ids"] for r in sub),
            max_opportunity=max((r["max_opportunity"] for r in sub), default=0),
            trace_proof=all(r["max_opportunity"] == 0 for r in sub)))


def summarize(rows, pairs, tasks):
    return dict(
        study=str(OUT), trace=list(TRACE), targets=list(TARGETS), arms=list(ARMS),
        cohorts={name: {str(q): group(rows, pairs, tasks, indices, q, name)
                        for q in TARGETS}
                 for name, indices in COHORTS.items()})


def write_tables(out, rows, pairs, tasks):
    per = [question(rows, pairs, tasks, i, q) for q in TARGETS for i in range(90)]
    columns = ["cohort", "index", "dataset", "task_id", "target", "exposed", "first_tick",
               "rounds", "edge_pairs", "released_rounds", "released_edge_pairs",
               "target_ticks", "opportunity_ticks", "max_opportunity",
               "causal_correct", "open_correct", "delta_correct", "causal_N", "causal_F",
               "causal_Q", "open_N", "open_F", "open_Q", "causal_tpf", "open_tpf",
               "delta_tpf", "causal_efficiency", "open_efficiency", "delta_efficiency",
               "same_ids", "rescued", "lost"]
    with (out/"per_question.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in per)
    summary = summarize(rows, pairs, tasks)
    flat = []
    for name, targets in summary["cohorts"].items():
        for q, g in targets.items():
            flat.append(dict(cohort=name, target=int(q), **{
                k: v for k, v in g.items() if k not in ("cohort", "target")}))
    columns = ["cohort", "target", "requests", "exposed", "triggered", "rounds", "edge_pairs",
               "released_rounds", "released_edge_pairs", "target_ticks", "opportunity_ticks",
               "max_opportunity", "rescued", "lost",
               "same_ids", "delta_correct", "delta_tpf", "delta_efficiency",
               "exposed_delta_correct", "exposed_delta_tpf", "exposed_delta_efficiency",
               "negative_control"]
    with (out/"summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in flat:
            writer.writerow({k: (json.dumps(row[k], sort_keys=True)
                                 if isinstance(row[k], (list, dict)) else row[k])
                             for k in columns})


def score(out=OUT):
    """Use only sealed artifacts and the frozen judge. Never load or run a model."""
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.evaluation.scoring import score as judge

    tasks = load_tasks(DATA/"samples.json")
    reg.require(reg.sha(DATA/"samples.json") == reg.SAMPLES_SHA, "Wrong sample digest")
    run = reg.read(out/"run.json")
    expected = spec()
    reg.require(all(run[k] == v for k, v in expected.items() if k != "sources")
                and run["sources"]["bdlm_serving/evaluation/scoring.py"]
                == expected["sources"]["bdlm_serving/evaluation/scoring.py"],
                "Wrong run protocol or judge")
    receipt = reg.read(out/"returned.json")
    reg.require(receipt["restored"] and receipt["gpu"] == 0 and receipt["uuid"] == GPU
                and receipt["original30"] == sealed(out, 30)
                and receipt["full90"] == sealed(out, 90), "Missing GPU return")
    rows = {arm: {} for arm in ARMS}
    for index in range(90):
        for arm in ARMS:
            row = reg.read(out/reg.filename(index, arm))
            reg.require(all(row[k] == tasks[index][k] for k in reg.KEYS),
                        "Frozen identity mismatch")
            audit(row, arm)
            row["correct"] = bool(judge(row["output_text"], tasks[index])["correct"])
            rows[arm][index] = row
    pairs = {q: {} for q in TARGETS}
    for index in range(90):
        for q in TARGETS:
            pairs[q][index] = pair(rows[CAUSAL][index], rows[f"open{q}"][index], q)
    summary = summarize(rows, pairs, tasks)
    reg.save(out/"summary.json", summary)
    write_tables(out, rows, pairs, tasks)
    return summary


def check():
    """CPU-only interface check: no model, no GPU, no holder, no lease."""
    source = HERE.with_name("spawn_mask_gpu.py").read_text()
    report = dict(
        module=str(HERE), repo=str(REPO), out=str(OUT), predecessor=str(PREV),
        trace=list(TRACE), targets=list(TARGETS),
        arms={k: v for k, v in ARMS.items()}, cap=reg.CAP,
        plan=PLAN.folder,
        kernel="SpawnGatedKernel", spawn_threshold=GATE, gate="progress80",
        samples=str(DATA/"samples.json"),
        samples_ok=reg.sha(DATA/"samples.json") == reg.SAMPLES_SHA,
        launcher_ok=reg.sha(reg.ROOT/"launch_prefix90.py") == reg.LAUNCHER_SHA,
        whitelist="llada_offer_masks" in source,
        predecessor_state="returned" if (PREV/"returned.json").exists() else "running",
        invoke=[str(REPO.parent/"venv/bin/python"), "-u", str(HERE), "llada21", "0"],
        env=dict(PREFIX_MODE="llrefs", LL_BASELINE_POLICIES="offer-masks",
                 TRACE_OUT=str(OUT), FROZEN90_SAMPLES=str(DATA/"samples.json"),
                 CUDA_VISIBLE_DEVICES=GPU))
    if (PREV/"returned.json").exists():
        try:
            report["predecessor_receipt"] = predecessor()
        except (ValueError, FileNotFoundError) as error:
            report["predecessor_receipt"] = dict(error=str(error))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score", "check", "queue"))
    parser.add_argument("--study", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "score":
        score(args.study)
    elif args.command == "check":
        check()
    else:
        queue()


if __name__ == "__main__":
    if sys.argv[1:2] == ["llada21"]:
        protected()
    else:
        main()
