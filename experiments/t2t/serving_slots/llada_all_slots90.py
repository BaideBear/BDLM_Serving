"""Protected frozen90 GPU4 generation and separate, model-free offline scoring.

From the deployed t2t-experiments checkout with PYTHONPATH=.:../fixed-baseline-deps:
  ../venv/bin/python -u -m experiments.t2t.serving_slots.llada_all_slots90 queue
  ../venv/bin/python -m experiments.t2t.serving_slots.llada_all_slots90 score [--mask-only]

Register exactly one queue in tmux after prefix90-llada-top-transition. It can
wait for that session. Disappearance is not a receipt. Verify all eight frozen
predecessor arms, its qualified audit, and current sole holder before starting.
Never resume an incomplete lease or synthesize a protected return marker.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
from random import Random
import subprocess
import sys
import time

ROOT = Path("/data1/project_data/gx-erqw-20261123/runtime/t2t-native-1125")
REPO = ROOT.parent / "t2t-experiments"
DATA = ROOT / "prefix90"
STUDY = DATA / "llada21-all-slots90-v1"
PRED = DATA / "llada21-top-offer32-64-v1"
PARENT = "prefix90-llada-top-transition"
HOLDER = "/data1/project_data/gx-megrez-20261122/gpu-holders/gpu-hold"
UUID = "GPU-d3bdc2bc-d19d-e98f-ac79-ab9c6e887e0e"
SAMPLES_SHA = "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed"
LAUNCHER_SHA = "ba80a293773326950738fd0d4ca0a8a04e6df4b86c6e9a7265bd2ba3a92e6e59"
HERE = Path(__file__).resolve()
KEYS = ("index", "dataset", "task_id", "seed")
CAP = 2048
LAYERS = 20
TRACE = (4, 8, 16, 32, 64)
SHIFT = TRACE[-1:] + TRACE[:-1]
ARMS = {f"fixed{q}": (q,) for q in TRACE}
ARMS.update(replay=TRACE, replay_open=TRACE, shifted=SHIFT, replay_right=TRACE)
RIGHT = {4: 2, 8: 4, 16: 4}
PAIR = ("replay", "replay_open")
# Pair adjacent requests on one loaded model; one arm per remaining lease.
GROUPS = tuple((PAIR, range(i, i+30)) for i in (0, 30, 60)) + tuple(
    ((arm,), range(90)) for arm in ("shifted", "replay_right", *(f"fixed{q}" for q in TRACE)))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def save(path, obj):
    temp = path.with_suffix(".partial")
    with temp.open("x") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    temp.replace(path)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def spec(i):
    arms, indices = GROUPS[i]
    return dict(model="llada21", suite="frozen90", gpu=4, gpu_uuid=UUID, gen_length=CAP,
                samples_sha256=SAMPLES_SHA, launcher_sha256=LAUNCHER_SHA, group=i,
                indices=list(indices), fuse_growth=True, arms={
                    arm: dict(trace=list(ARMS[arm]), opened=(arm == "replay_open"),
                              prefix_mode="right" if arm == "replay_right" else "zero",
                              prefix_slots={str(k): v for k, v in RIGHT.items()}
                              if arm == "replay_right" else {}) for arm in arms})


def folder(study, i):
    return study / f"lease-{i:02d}"


def filename(index, arm):
    return f"{index:02d}-{arm}.json"


def geometry(block, prompt, positions, end):
    boundary = (block + 1) * 32
    rows = [p for p in positions if max(prompt, boundary-32) <= p < boundary]
    right_end = min(end, boundary+32)
    if not rows or right_end <= boundary:
        return None
    return dict(block=block, left_query_count=len(rows), left_query_first=rows[0],
                left_query_last=rows[-1], right_key_start=boundary, right_key_end=right_end,
                opened_edges=len(rows)*(right_end-boundary))


def observe(kernel, state, arm, opener):
    """Bind to the actual forward, not a guessed stage (step may include prefill)."""
    import torch
    calls = []
    first_input = None

    def hook(_module, _args, kw):
        nonlocal first_input
        mask, ids = kw["attention_mask"], kw["position_ids"]
        positions, end = ids[0].tolist(), mask.shape[-1]
        require(tuple(mask.shape) == (1, 1, len(positions), end), "Not a rectangular mask")
        expected = geometry(state.block, state.prompt_length, positions, end)
        updated, info = opener(mask, ids, state.prompt_length, state.block)
        require(info == expected, "Mask hook disagrees with actual query/key geometry")
        opened = info["opened_edges"] if info and arm == "replay_open" else 0
        if info and first_input is None:
            first_input = digest(dict(canvas=state.x[0].tolist(), write=state.last_write,
                                      query=state.last_query, committed=state.committed_end,
                                      context=state.context_version, cpu=state.rng_cpu.tolist(),
                                      device=state.rng_cuda.tolist() if state.rng_cuda is not None else None))
        sparse_offer = ARMS[arm][state.offer_tick % len(ARMS[arm])] < 32
        start = state.block * 32
        head_masks = ([start+n for n, v in enumerate(state.x[0, start:start+32].tolist())
                       if v == state.mask_id] if sparse_offer else None)
        calls.append((state.block, positions, end, info, opened, head_masks))
        if opened:
            return _args, dict(kw, attention_mask=updated)
        return None

    target = getattr(kernel.model, "model", kernel.model)
    if type(target).forward is torch.nn.Module.forward:
        target = kernel.model
    handle = target.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        trace = ARMS[arm]
        kernel.run(state, lambda tick: trace[tick.tick % len(trace)])
    finally:
        handle.remove()
    for e, (block, positions, end, info, opened, masks) in zip(state.events, calls, strict=True):
        require((block, positions, end) == (e["block"], e["absolute_positions"], e["visible_kv"]),
                "Model invocation differs from kernel event")
        e.update(mask_window=info, opened_edges=opened)
        if "prefix_plan" in e:
            e["observed_head_masks"] = masks
    return first_input


def audit(row, arm):
    """Reconcile hooked geometry, physical work, sparse obligations and overflow reasons."""
    require(row["policy"] == arm and row["model"] == "llada21" and row["status"] == "success",
            "Wrong arm/model or unsuccessful request")
    trace, events, prompt = ARMS[arm], row["events"], len(row["prompt_ids"])
    prefill = prompt // 32 * 32
    gen = events[int(prefill > 0):]
    require(gen and row["prefill_rows"] == prefill, "Missing generation or wrong prompt work")
    first = None
    reasons, histogram, stages, queries = Counter(), Counter(), Counter(), Counter()
    optional = opened = parked = sampled = 0
    expected_block = prompt // 32
    for j, e in enumerate(events):
        pos, end, q = e["absolute_positions"], e["visible_kv"], e["physical_q"]
        require(e["block"] == expected_block and pos and pos == sorted(set(pos))
                and len(pos) == e["actual_q"] == q == e["q_tokens"]
                and 0 <= pos[0] <= pos[-1] < end and e["F"] == 1
                and e["layer_physical_q"] == [q]*LAYERS, "Invalid physical work")
        geom = geometry(e["block"], prompt, pos, end)
        edges = geom["opened_edges"] if geom and arm == "replay_open" else 0
        require(e["mask_window"] == geom and e["opened_edges"] == edges, "Wrong mask intervention")
        if prefill and j == 0:
            require(e["phase"] == "prefill" and pos == list(range(prefill)) and end == prefill
                    and not geom and "offer_tick" not in e, "Unexpected prompt forward")
            continue
        tick = j - int(prefill > 0)
        offer = trace[tick % len(trace)]
        require(e["phase"] != "prefill" and e["offer_tick"] == tick
                and e["nominal_q"] == e["q_budget"] == offer and q <= 64
                and e["previous_offer"] == (trace[(tick-1) % len(trace)] if tick else None),
                "Wrong external offer clock")
        if geom and first is None:
            first = j
        debt, dense, refresh = map(set, (e["refreshed_pending"], e["dense_required_positions"],
                                         e["refresh_only_positions"]))
        start = e["block"] * 32
        action = e["top_action"]
        require(e["context_version"] == expected_block - prompt//32 and
                (action not in ("pair_bootstrap", "pair_denoise", "fused_growth") or offer == 64)
                and (tick or action in ("head_bootstrap", "pair_bootstrap"))
                and (j != len(events)-1 or action == "commit_head"), "Invalid generation transition")
        if action == "commit_head":
            expected_block += 1
        dense_action = {"head_bootstrap": ("bootstrap", 32), "pair_bootstrap": ("bootstrap", 64),
                        "fused_growth": ("bootstrap", 64), "promote_verify": ("promotion_verify", 32),
                        "head_verify": ("verify", 32), "commit_head": ("commit", 32)}
        reason, width = dense_action.get(action, (None, 0))
        require((reason or action in ("pair_denoise", "head_denoise", "tail_dirty_drain"))
                and e["dense_reason"] == reason and e["dense_required_q"] == width
                and dense == set(range(start, start+width)) and refresh <= debt <= set(pos)
                and (not width or set(pos) == dense | refresh)
                and e["mandatory_q"] == len(debt)
                and e["mandatory_head_positions"] == sorted(p for p in debt if p < start+32)
                and e["mandatory_tail_positions"] == sorted(p for p in debt if p >= start+32),
                "Wrong mandatory or dense rows")
        sampled_rows = [] if action == "commit_head" else sorted(set(pos)-refresh)
        require(e["sampled_positions"] == sampled_rows and e["sampled_q"] == len(sampled_rows),
                "Wrong sampled/scatter partition")
        over = max(0, q-offer)
        dense_over = max(0, width-offer)
        why = {reason: dense_over} if dense_over else {}
        if over > dense_over:
            why["mandatory_refresh"] = over-dense_over
        require(e["overflow"] == over and e["overflow_reasons"] == why
                and e["underfill"] == max(0, offer-q)
                and (q <= max(offer, len(debt | dense)) if offer < 32 else over <= len(refresh)),
                "Unexplained physical overrun")
        sparse = offer < 32 and action in ("head_denoise", "tail_dirty_drain")
        require(("prefix_plan" in e) == sparse, "Missing or spurious sparse selection")
        if sparse:
            p = e["prefix_plan"]
            base, pool, add = p["zero_rows"], p["eligible_prefix_rows"], p["added_rows"]
            masks = e["observed_head_masks"]
            editable = range(max(start, prompt), start+32)
            require(masks and masks == sorted(set(masks)) and set(masks) <= set(editable),
                    "No observed unresolved head")
            capacity = max(0, offer-len(debt))
            mask_pick = [v for v in masks if v not in debt][:capacity]
            ordinary = [v for v in editable if v >= masks[0] and v not in masks and v not in debt]
            old_pick = ordinary[:capacity-len(mask_pick)]
            require(base == sorted(e["mandatory_head_positions"] + mask_pick + old_pick)
                    and p["zero_mask_rows"] == mask_pick and p["zero_backfill_rows"] == old_pick
                    and p["prefix_end"] == masks[0]
                    and pool == [v for v in editable if v < masks[0] and v not in debt],
                    "Zero plan differs from observed physical head")
            slack = max(0, offer - len(base) - len(refresh))
            quota = RIGHT[offer] if arm == "replay_right" else 0
            n = min(quota, slack, len(pool)) if base else 0
            require(p["mode"] == ("right" if quota else "zero") and p["quota"] == quota
                    and p["slack"] == slack and add == (pool[-n:] if n else [])
                    and pool == sorted(set(pool)) and base == sorted(set(base))
                    and all(max(prompt, start) <= v < p["prefix_end"] < start+32 for v in pool)
                    and not (set(pool) & (debt | set(base))) and not p["evicted_rows"]
                    and p["selected_rows"] == sorted(base+add) == sampled_rows
                    and refresh == {v for v in debt if v >= start+32},
                    "Not a true-slack prefix placement")
            optional += n
        opened += edges
        parked += len(refresh)
        sampled += len(sampled_rows)
        reasons.update(why)
        stages[e["stage"]] += 1
        queries[e["stage"]] += q
        histogram[str(q)] += 1
    f, q, n = len(gen), sum(queries.values()), row["output_tokens"]
    ids, eos, special = row["output_ids"], row["eos_id"], row["special_ids"]
    require(bool(ids) and len(ids) <= CAP and eos not in ids[:-1]
            and row["eos"] == (ids[-1] == eos) and row["truncated"] != row["eos"]
            and (row["eos"] or len(ids) == CAP) and n == sum(t not in special for t in ids),
            "Wrong output/EOS convention")
    require(row["generation_forwards"] == f and row["generation_query_rows"] ==
            row["generation_physical_padded_rows"] == q and row["tpf"] == n/f
            and row["token_efficiency"] == n/q and row["max_q"] == max(map(int, histogram))
            and row["q_histogram"] == histogram and row["forwards_by_phase"] == stages
            and row["q_by_phase"] == queries, "Wrong packaged metrics")
    stamp = row["first_input_sha256"]
    require(stamp is None if first is None else isinstance(stamp, str) and len(stamp) == 64,
            "Missing or unexpected first input fingerprint")
    first_geometry = None if first is None else {k: events[first][k] for k in (
        "offer_tick", "q_budget", "stage", "block", "absolute_positions", "sampled_positions",
        "refresh_only_positions", "refreshed_pending", "visible_kv", "mask_window")}
    return dict(first=first, first_geometry=first_geometry, opened_edges=opened, optional=optional, parked=parked,
                sampled=sampled, overflow=dict(reasons), N=n, F=f, Q=q)


def paired(on, off):
    a, b = audit(on, PAIR[0]), audit(off, PAIR[1])
    i = a["first"]
    require(i == b["first"] and all(on[k] == off[k] for k in
            (*KEYS, "prompt_ids", "special_ids", "eos_id", "first_input_sha256")),
            "Different first exposure or input")
    require(on["events"][:i] == off["events"][:i], "Different pre-exposure trajectory")
    if i is None:
        require({k: v for k, v in on.items() if k != "policy"} ==
                {k: v for k, v in off.items() if k != "policy"}, "Unexposed pair differs")
    else:
        for k in ("stage", "phase", "top_action", "block", "offer_tick", "q_budget", "context_version",
                  "visible_kv", "absolute_positions", "sampled_positions", "refreshed_pending",
                  "refresh_only_positions", "prefix_plan", "mask_window"):
            require(on["events"][i].get(k) == off["events"][i].get(k), f"First geometry differs: {k}")
    return i


def run(args, prepare):
    """Only called from the verified launcher worker. No task judging in the lease."""
    import torch
    import transformers.utils as utils
    from typing import TypedDict
    if not hasattr(utils, "LossKwargs"):
        class LossKwargs(TypedDict, total=False):
            num_items_in_batch: int
        utils.LossKwargs = LossKwargs
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.methods.t2t.online import package
    from bdlm_serving.runtime.serial import install_llada_rectangular_mask
    from experiments.t2t.serving_slots.llada_all_slots import AllSlotKernel
    from experiments.t2t.serving_slots.mask_pair_llada import head_to_tail_mask
    from experiments.t2t.serving_slots.top_transition90 import resolve_reference, validate_prompts

    i = int(os.environ["ALL_SLOTS_GROUP"])
    require(0 <= i < len(GROUPS) and prepare is not None and os.environ["CUDA_VISIBLE_DEVICES"] == UUID
            and args.output == folder(STUDY, i) and args.samples == DATA/"samples.json"
            and sha(args.samples) == SAMPLES_SHA, "Not the protected frozen90 worker")
    tasks = load_tasks(args.samples)
    tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True)
    reference = resolve_reference(args)
    validate_prompts(tok, tasks, reference)
    for item in read(REPO / "benchmarks/checkpoints/llada21.json"):
        path = args.checkpoint/item["name"]
        require(path.stat().st_size == item["bytes"] and
                ("sha256" not in item or sha(path) == item["sha256"]), "Wrong checkpoint")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, trust_remote_code=True,
        local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    install_llada_rectangular_mask(model, allow_selected_cache=True)
    for p in model.parameters():
        p.data = p.data.clone()
    prepare(model)
    model.to("cuda")
    mask, eos = tok.convert_tokens_to_ids("<|mask|>"), tok.eos_token_id
    special = sorted(set(tok.all_special_ids) | {mask, eos})
    cfg = dict(confidence_threshold=.5, editing_threshold=.5, max_post_steps=16)
    out = args.output
    out.mkdir(exist_ok=False)
    save(out/"run.json", spec(i))
    files = {}
    arms, indices = GROUPS[i]
    for index in indices:
        task = tasks[index]
        # validate_prompts already verified these frozen IDs against this tokenizer.
        prompt = read(reference / f"{index:02d}-native32.json")["prompt_ids"]
        ids = torch.tensor([prompt], device=model.device, dtype=torch.long)
        rows = {}
        for arm in arms:
            kernel = AllSlotKernel(model, prefix_slots=RIGHT if arm == "replay_right" else None,
                                   prefix_mode="right" if arm == "replay_right" else "zero")
            state = kernel.create(ids, cfg, CAP, task["seed"], mask, [eos])
            stamp = observe(kernel, state, arm, head_to_tail_mask)
            path = state.events
            work = [(e["stage"], e["physical_q"]) for e in path if e["phase"] != "prefill"]
            row = package(tok, kernel.output(state)[0].tolist(), work, [eos], special,
                          sum(e["physical_q"] for e in path if e["phase"] == "prefill"),
                          events=path, prompt_ids=prompt, first_input_sha256=stamp,
                          special_ids=special, eos_id=eos, policy=arm, model="llada21",
                          **{k: task[k] for k in KEYS})
            audit(row, arm)
            rows[arm] = row
        if tuple(arms) == PAIR:
            paired(rows[PAIR[0]], rows[PAIR[1]])
        for arm, row in rows.items():
            name = filename(index, arm)
            save(out/name, row)
            files[name] = sha(out/name)
        print(f"lease {i:02d} complete request {index:02d}: {','.join(arms)}", flush=True)
    save(out/"worker-complete.json", dict(spec_sha256=digest(spec(i)), files=files))


def predecessor(path):
    policies = ["toptrace_" + "_".join(map(str, trace)) + "_" + variant
                for trace in ((32,), (64,), (32, 32, 64, 64), (64, 64, 32, 32))
                for variant in ("fused", "split")]
    meta, audit_log = read(path/"run.json"), read(path/"top-transition-audit.json")
    require(meta["formal"] is True and meta["suite"] == "frozen90" and meta["model"] == "llada21"
            and meta["requests"] == 90 and meta["gen_length"] == CAP and meta["limit"] is None
            and meta["samples_sha256"] == SAMPLES_SHA and meta["policies"] == policies,
            "Wrong TopTransition predecessor")
    require(read(path/"qualified.json") == dict(model="llada21", suite="frozen90",
            trace_contract_checked=True, completed_requests=90, gen_length=CAP, policies=policies)
            and audit_log["model"] == "llada21" and audit_log["suite"] == "frozen90"
            and audit_log["samples_sha256"] == SAMPLES_SHA and audit_log["requests"] == 90
            and audit_log["checks"]["all_rows_audited"] is True
            and set(audit_log["arms"]) == set(policies), "Missing predecessor qualification")
    for p in policies:
        require(read(path/f"complete-{p}.json") == dict(policy=p, requests=90)
                and audit_log["arms"][p]["requests"] == 90, f"Incomplete predecessor: {p}")
    return dict(session=PARENT, run=path.name, samples_sha256=SAMPLES_SHA, policies=policies,
                qualified_sha256=sha(path/"qualified.json"), audit_sha256=sha(path/"top-transition-audit.json"))


def holder():
    record, = json.loads(subprocess.check_output([HOLDER, "status", "4"]))
    active = subprocess.check_output(["nvidia-smi", "-i", UUID, "--query-compute-apps=pid",
                                      "--format=csv,noheader,nounits"]).split()
    pid = record["pid"]
    require(record["running"] is True and record["gpu_status"]["uuid"] == UUID
            and list(map(int, active)) == [pid], "GPU4 is not solely held by the registered holder")
    return dict(uuid=UUID, sole_holder_pid=pid)


def source_hashes():
    slots = REPO/"experiments/t2t/serving_slots"
    paths = [slots/n for n in ("llada_all_slots90.py", "llada_all_slots.py", "llada_top_transition.py",
                               "top_transition90.py", "mask_pair_llada.py")]
    paths += [REPO/"bdlm_serving/methods/t2t"/n for n in ("llada_dual.py", "llada_sparse.py")]
    return {p.name: sha(p) for p in paths}


def verify_worker(study, i, verify_bytes=False):
    out = folder(study, i)
    arms, indices = GROUPS[i]
    work = read(out/"worker-complete.json")
    require(read(out/"run.json") == spec(i) and work["spec_sha256"] == digest(spec(i))
            and work["files"].keys() == {filename(j, arm) for j in indices for arm in arms},
            "Incomplete or wrong lease manifest")
    if verify_bytes:
        for name, checksum in work["files"].items():
            require(sha(out/name) == checksum, f"Wrong request digest: {name}")
    return work


def return_receipt(study, i, previous):
    out = folder(study, i)
    work = verify_worker(study, i)
    receipt = read(out/"protected-complete.json")
    require(receipt["worker_sha256"] == digest(work) and receipt["previous_sha256"] == previous
            and receipt["spec_sha256"] == digest(spec(i)) and receipt["launcher_exit"] == 0
            and receipt["holder"]["uuid"] == UUID and receipt["holder"]["sole_holder_pid"] > 0,
            "Missing or mismatched protected return receipt")
    return digest(receipt)


def protected(argv):
    i = int(os.environ["ALL_SLOTS_GROUP"])
    require(argv in (["llada21", "4"], ["llada21", "4", "worker"]) and 0 <= i < len(GROUPS)
            and not sys.flags.optimize, "Wrong protected invocation")
    registration = read(STUDY/"registered.json")
    require(registration["sources"] == source_hashes() and
            registration["predecessor"] == predecessor(PRED), "Registration inputs changed")
    previous = digest(registration)
    for group in range(i):
        previous = return_receipt(STUDY, group, previous)
    launcher = ROOT/"launch_prefix90.py"
    source = launcher.read_bytes()
    key = b"from experiments.t2t.serving_slots.llada_reference import run"
    require(hashlib.sha256(source).hexdigest() == LAUNCHER_SHA and source.count(key) == 1,
            "Protected launcher changed")
    os.environ.update(PREFIX_MODE="llrefs", TRACE_OUT=str(folder(STUDY, i)),
                      FROZEN90_SAMPLES=str(DATA/"samples.json"), LL_BASELINE_POLICIES="all_slots")
    replacement = b"from experiments.t2t.serving_slots.llada_all_slots90 import run"
    exec(compile(source.replace(key, replacement), str(launcher), "exec"),
         {"__name__": "__main__", "__file__": str(HERE)})
    # The pinned worker calls sys.exit. Only the launcher parent reaches this point.
    require(len(argv) == 2, "Worker cannot certify holder return")
    returned = holder()
    work = verify_worker(STUDY, i, verify_bytes=True)
    save(folder(STUDY, i)/"protected-complete.json",
         dict(worker_sha256=digest(work), spec_sha256=digest(spec(i)),
              previous_sha256=previous, launcher_exit=0, holder=returned))


def queue():
    require(not STUDY.exists() and sha(DATA/"samples.json") == SAMPLES_SHA and
            sha(ROOT/"launch_prefix90.py") == LAUNCHER_SHA, "Existing run or changed pinned inputs")
    while True:
        probe = subprocess.run(["tmux", "has-session", "-t", "="+PARENT], capture_output=True)
        if probe.returncode == 1 and (b"can't find session:" in probe.stderr
                                      or b"no server running" in probe.stderr):
            break
        probe.check_returncode()
        time.sleep(8)
    receipt = dict(predecessor=predecessor(PRED), holder=holder(), sources=source_hashes())
    STUDY.mkdir(exist_ok=False)
    save(STUDY/"registered.json", receipt)
    previous = digest(receipt)
    for i in range(len(GROUPS)):
        with (STUDY/f"lease-{i:02d}.log").open("x") as log:
            subprocess.run([sys.executable, "-u", str(HERE), "llada21", "4"],
                           cwd=REPO, env=dict(os.environ, ALL_SLOTS_GROUP=str(i)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        previous = return_receipt(STUDY, i, previous)
    save(STUDY/"complete.json", dict(groups=len(GROUPS), requests=90,
                                     policies=list(ARMS), final_receipt_sha256=previous))


def totals(rows):
    n, f, q = (sum(r[k] for r in rows) for k in ("N", "F", "Q"))
    return dict(requests=len(rows), correct=sum(r["correct"] for r in rows), N=n, F=f, physical_Q=q,
                macro_TPF=sum(r["N"]/r["F"] for r in rows)/len(rows) if rows else None,
                token_efficiency=n/q if q else None,
                exposed=sum(r["first"] is not None for r in rows),
                prefix_rows=sum(r["optional"] for r in rows), parked_rows=sum(r["parked"] for r in rows),
                opened_edges=sum(r["opened_edges"] for r in rows))


def paired_summary(pairs):
    def effects(draw):
        a, b = zip(*draw)
        return (sum(y["correct"]-x["correct"] for x, y in draw)/len(draw),
                sum(y["N"]/y["F"]-x["N"]/x["F"] for x, y in draw)/len(draw),
                sum(y["N"] for y in b)/sum(y["Q"] for y in b) -
                sum(x["N"] for x in a)/sum(x["Q"] for x in a))
    if not pairs:
        return dict(requests=0, delta=None, ci95=None)
    rng = Random(17000)
    samples = [effects(rng.choices(pairs, k=len(pairs))) for _ in range(2000)]
    return dict(requests=len(pairs), rescued=sum(not a["correct"] and b["correct"] for a, b in pairs),
                lost=sum(a["correct"] and not b["correct"] for a, b in pairs),
                same_ids=sum(a["output_digest"] == b["output_digest"] for a, b in pairs),
                delta=effects(pairs), ci95=[[v[49], v[1949]] for v in map(sorted, zip(*samples))])


def score(study, samples, *, mask_only=False):
    """Use only sealed artifacts and the frozen judge. Never load or run a model."""
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.evaluation.scoring import score as judge
    require(sha(samples) == SAMPLES_SHA, "Wrong sample digest")
    tasks = load_tasks(samples)
    return score_files(study, tasks, judge, mask_only=mask_only)


def score_files(study, tasks, judge, *, mask_only=False):
    require([t["index"] for t in tasks] == list(range(90)), "Wrong frozen indices")
    reg = read(study/"registered.json")
    pred_dir = study.parent/PRED.name
    require(reg["predecessor"] == predecessor(pred_dir) and reg["holder"]["uuid"] == UUID
            and reg["holder"]["sole_holder_pid"] > 0, "Unverified predecessor registration")
    previous = digest(reg)
    groups = [i for i, (arms, _) in enumerate(GROUPS) if not mask_only or arms == PAIR]
    policies = PAIR if mask_only else tuple(ARMS)
    results = {arm: {} for arm in policies}
    counts = {arm: Counter() for arm in policies}
    offer_work = {arm: {} for arm in policies}
    for i in groups:
        previous = return_receipt(study, i, previous)
        out = folder(study, i)
        manifest = verify_worker(study, i)["files"]
        arms, indices = GROUPS[i]
        for index in indices:
            task = tasks[index]
            rows = {}
            for arm in arms:
                name = filename(index, arm)
                payload = (out/name).read_bytes()
                require(hashlib.sha256(payload).hexdigest() == manifest[name], "Request digest mismatch")
                row = rows[arm] = json.loads(payload)
                require(all(row[k] == task[k] for k in KEYS) and index not in results[arm],
                        "Repeated or different frozen request")
                entry = audit(row, arm)
                counts[arm].update(entry["overflow"])
                for e in row["events"]:
                    if e["phase"] != "prefill":
                        counter = offer_work[arm].setdefault(e["q_budget"], Counter())
                        counter.update(F=1, physical_Q=e["physical_q"], sampled_Q=e["sampled_q"],
                                       underfill=e["underfill"], overflow=e["overflow"])
                entry.update(correct=bool(judge(row["output_text"], task)["correct"]),
                             output_digest=digest(row["output_ids"]), **{k: task[k] for k in KEYS})
                if arm in ("fixed32", "fixed64"):
                    q = ARMS[arm][0]
                    old = read(pred_dir/f"{index:02d}-toptrace_{q}_fused.json")
                    require(all(old[k] == task[k] for k in KEYS), "Unpaired fixed top control")
                    entry["old_ids_equal"] = old["output_ids"] == row["output_ids"]
                    entry["old_work_equal"] = all(row[k] == old[k] for k in (
                        "generation_forwards", "generation_physical_padded_rows", "q_histogram",
                        "prefill_rows", "output_tokens"))
                results[arm][index] = entry
            if tuple(arms) == PAIR:
                paired(rows["replay"], rows["replay_open"])
    if not mask_only:
        require(read(study/"complete.json") == dict(groups=len(GROUPS), requests=90,
                policies=list(ARMS), final_receipt_sha256=previous), "Incomplete final receipt")
    require(all(set(rows) == set(range(90)) for rows in results.values()), "Incomplete arm coverage")

    cohorts = {"full90": range(90), "original30": range(30), "new60": range(30, 90)}
    summary, pairs = [], []
    for cohort, indices in cohorts.items():
        for arm in policies:
            summary.append(dict(arm=arm, cohort=cohort, **totals([results[arm][j] for j in indices])))
        contrasts = [("replay", "replay_open")]
        if not mask_only:
            contrasts += [("replay", "shifted"), ("replay", "replay_right")]
        for base, other in contrasts:
            for scope in (("all", "exposed", "unexposed") if other == "replay_open" else ("all",)):
                cases = [(results[base][j], results[other][j]) for j in indices if
                         scope == "all" or (results[base][j]["first"] is not None) == (scope == "exposed")]
                pairs.append(dict(base=base, other=other, cohort=cohort, scope=scope, **paired_summary(cases)))
    report = dict(summary=summary, contrasts=pairs, overflow_reasons=counts, by_offer=offer_work,
                  delta_order=["accuracy", "macro_TPF", "pooled_token_efficiency"],
                  direction="other minus base", paired_bootstrap_draws=2000, seed=17000,
                  protected_receipt_sha256=previous, cases={arm: list(rows.values()) for arm, rows in results.items()})
    stem = "mask-pair" if mask_only else "all-slots"
    save(study/f"{stem}-score.json", report)
    with (study/f"{stem}-summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)
    return report


def main(argv):
    if argv[:2] == ["llada21", "4"]:
        return protected(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("queue")
    p = sub.add_parser("score")
    p.add_argument("--study", type=Path, default=STUDY)
    p.add_argument("--samples", type=Path, default=DATA/"samples.json")
    p.add_argument("--mask-only", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "queue":
        queue()
    else:
        report = score(args.study, args.samples, mask_only=args.mask_only)
        for row in report["summary"]:
            if row["cohort"] == "full90":
                print(row)


if __name__ == "__main__":
    main(sys.argv[1:])
