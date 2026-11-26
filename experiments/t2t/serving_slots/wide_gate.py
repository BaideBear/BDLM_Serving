"""Frozen90 two-phase offer replay at multiple low widths, on either physical block size."""
import json
import os
from pathlib import Path
from time import perf_counter_ns

from bdlm_serving.evaluation.frozen import sha
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.methods.t2t.offer_gate import choose_native
from bdlm_serving.methods.t2t.online import load_backend
from bdlm_serving.methods.t2t.protocol import Arm, verify_work
from bdlm_serving.methods.t2t.slot_trace import RoundInfo
from experiments.t2t.serving_slots.switch_report import audit, gemma_stages
from tools.run_t2t import check_flat, durable

POLICIES = tuple("wide_" + p for p in (
    "fixed_low", "fixed_high", "all_high", "tail_high", "all_low", "tail_low"))
REFERENCES = {
    "llada21": {8:"llada-q8", 16:"llada21-16-matched", 24:"llada21-expanded-fixed-24-12-6"},
    "gemma": {32:"gemma-q32", 64:"gemma-grid-k1", 192:"gemma-expanded-fixed-192-96-48"},
}


def choice(policy, low, info):
    if policy not in POLICIES:
        raise ValueError("Unregistered phase or rule")
    _, rule, phase = policy.split("_")
    width = info.block_width
    high = phase == "high" if rule == "fixed" else (info.tick % 4 < 2) == (phase == "high")
    external = width if high else low
    offer = choose_native(info, external, low, width//4, "all" if rule == "fixed" else rule)
    return external, offer


def validate(row, decisions, low):
    ll = row["model"] == "llada21"
    width = 32 if ll else 256
    pattern = []
    prev = None
    for e in row["events"]:
        if "slot_tick" in e:
            t = len(pattern)
            if prev is None or prev["block"] != e["block"] or e["slot_tick"] != t or t >= len(decisions):
                raise ValueError("Budget consumed outside the physical block or in wrong order")
            pos = e["block"] * width if ll else e["block"]
            masks = prev["masked_after"] if ll else width-prev["resolved"]
            mandatory = e["mandatory_fresh"] if ll else e["available_fresh"]
            if ll and e["masked_before"] != masks:
                raise ValueError("MASK count changed between events")
            external, offer = choice(row["policy"], low, RoundInfo(t, pos, width, masks, mandatory))
            if ([t,pos,masks,mandatory,external,offer.q] != decisions[t] or offer.q != e["slot_offer"]
                    or offer.native_full != (e["stage"] == "native_offer")):
                raise ValueError("Decision does not reproduce external and executed offer")
            pattern.append(f"{width}n" if offer.native_full else low)
        prev = e
    if len(pattern) != len(decisions):
        raise ValueError("Unused budget decision")
    verify_work(row)
    counts = audit(row, pattern or [low])
    if not ll:
        gemma_stages(row)
    return counts


def fixture(data, model, low, digest, width):
    folder = data / REFERENCES[model][low]
    p = f"t2t_q{low}_f0_m{low}_hard_prefix_zero"
    meta = json.loads((folder/"run.json").read_text())
    if ((meta["model"], meta["suite"], meta["samples_sha256"]) != (model,"frozen90",digest)
            or p not in meta["policies"]
            or json.loads((folder/"qualified.json").read_text()) != dict(model=model,full_width=width,matched=90)
            or json.loads((folder/f"complete-{p}.json").read_text()) != dict(policy=p,requests=90)):
        raise ValueError("Unqualified control")
    return folder, p


def run(args, prepare=None):
    import torch
    low = int(os.environ["WIDE_LOW"])
    model = args.model
    width = {"llada21":32, "gemma":256}[model]
    digest = sha(args.samples)
    tasks = load_tasks(args.samples)
    folder, ref_policy = fixture(Path(args.controls), model, low, digest, width)
    out = args.output
    out.mkdir(exist_ok=False)
    durable(out/"run.json", dict(model=model, suite="frozen90", samples_sha256=digest, batch_size=1,
                                 block=width, low=low, threshold=width//4, policies=POLICIES,
                                 decision_columns=["tick","block_start","masks","mandatory","external","effective"]))
    backend = load_backend(args, prepare)
    for p in POLICIES:
        initial = width if p.endswith("high") else low
        arm = Arm(initial, 0, initial, "hard_prefix_zero")
        times = []

        def execute(task):
            decisions = []

            def source(info):
                external, offer = choice(p, low, info)
                decisions.append([info.tick,info.block_start,info.masks,info.mandatory,external,offer.q])
                return offer
            return backend(task, arm, next_budget=source), decisions

        warm, warm_log = execute(tasks[0])
        for task in tasks:
            i = task["index"]
            torch.cuda.synchronize()
            begin = perf_counter_ns()
            row, decisions = execute(task)
            torch.cuda.synchronize()
            elapsed = perf_counter_ns()-begin
            if i == 0 and (warm != row or warm_log != decisions):
                raise ValueError("Nondeterministic first request")
            row.update({k:task[k] for k in ("index","task_id","dataset","seed")}, model=model, policy=p)
            validate(row, decisions, low)
            if p in POLICIES[:2]:
                peer = ref_policy if p == "wide_fixed_low" else "fullwidth"
                check_flat(row, json.loads((folder/f"{i:02d}-{peer}.json").read_text()))
            row["gate_decisions"] = decisions
            durable(out/f"{i:02d}-{p}.json", row)
            times.append(dict(index=i,wall_ns=elapsed))
            print(f"{p} {i:02d} F={row['generation_forwards']} Q={row['generation_query_rows']}",flush=True)
        durable(out/f"timing-{p}.json",dict(policy=p,requests=times))
        durable(out/f"complete-{p}.json",dict(policy=p,requests=90))
    durable(out/"replay-qualified.json",dict(model=model,low=low,matched_per_endpoint=90,samples=digest))
