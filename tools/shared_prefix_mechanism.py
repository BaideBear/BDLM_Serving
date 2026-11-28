"""GPU-only attribution probe: does a reserved seat need to actually project a prefix row?"""
import argparse
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import sys

from bdlm_serving.methods.t2t import unified_rows as rows
from experiments.t2t.serving_slots.shared_fixed_shard import (
    DEVICES, MANIFEST, WORKERS, check_board, check_sources, identity, load_tasks, shard_cases,
)


ARMS = ("all_mask", "reserve_idle", "prefix")
OFFERS = {"llada21": (4, 8), "gemma": (32, 64)}
DOSE = rows.Params(Fraction(1, 8), Fraction(1, 4))


def idle(full, nxt, opened):
    paid = tuple(sorted((*full.dirty, *full.mask)))
    expected_open = opened or bool(nxt and paid and paid[-1] >= nxt.start)
    if (full.open_next != expected_open or set(paid) != set(full.rows) - set(full.prefix)
            or (full.rows and not paid)):
        raise RuntimeError("a reserved prefix seat changed the block gate, displaced debt, or emptied the call")
    return replace(full, rows=paid, sampled=full.mask, prefix=())


@contextmanager
def instrument(arm):
    """One process, one request. Before first prefix exposure all three plans must match."""
    if arm not in ARMS:
        raise ValueError("unknown treatment arm")
    base = rows.plan_rows
    tape = sha256()
    info = dict(calls=0, exposed_calls=0, reserved_seats=0, avoided_mask_rows=0, first=None)

    def choose(q, b, head, next_block=None, *, opened=False, params=DOSE):
        if params != DOSE:
            raise ValueError("unexpected dose")
        kw = dict(opened=opened)
        full = base(q, b, head, next_block, params=DOSE, **kw)
        control = base(q, b, head, next_block, params=replace(DOSE, prefix_share=0), **kw)
        unfilled = idle(full, next_block, opened)
        if control.dirty != full.dirty or not set(full.mask) <= set(control.mask):
            raise RuntimeError("the hypothetical prefix change altered debt or added new MASK rows")
        info["calls"] += 1
        if full.prefix:
            info["exposed_calls"] += 1
            info["reserved_seats"] += len(full.prefix)
            info["avoided_mask_rows"] += len(control.mask) - len(full.mask)
            if info["first"] is None:
                info["first"] = dict(tick=info["calls"]-1, prior=tape.hexdigest(), prefix=list(full.prefix),
                                     dirty=list(full.dirty), mask=list(full.mask), no_prefix_mask=list(control.mask),
                                     opened=opened, origin=head.start)
        elif full != control or full != unfilled:
            raise RuntimeError("unexposed treatments diverge on the same state")
        plan = {"prefix": full, "reserve_idle": unfilled, "all_mask": control}[arm]
        if info["first"] is None:
            tape.update(json.dumps([plan.rows, plan.sampled, plan.dirty, plan.cold, plan.open_next],
                                   separators=(",", ":")).encode()+b"\n")
        return plan

    rows.plan_rows = choose
    try:
        yield info
    finally:
        rows.plan_rows = base
        info["unexposed_tape"] = tape.hexdigest()


def registration(base):
    if (base["stage"], base["cap"]) != ("formal90", 2048):
        raise ValueError("unregistered worker source")
    return dict(format="shared-prefix-mechanism-v1", source=identity(base), stage="mechanism18",
                wrapper=sha256(Path(__file__).read_bytes()).hexdigest(), arms=ARMS, offers=OFFERS,
                prefix="1/8", open="1/4", cases="screen18", cap=2048)


@contextmanager
def observe_writes(model, worker):
    """Read already measured native edits, not logits or a counterfactual edit prediction."""
    stats = dict(prefix_Q=0, prefix_t2t=0, prefix_revocations=0, edit_total=0, sample_mask_to_token=0,
                 drain_F=0)

    def account(e):
        p, edit, revoked = (set(e[k]) for k in ("prefix", "edited", "revoked"))
        if not (edit | revoked) <= set(e["sampled"]) or edit & revoked:
            raise ValueError("edit or revocation outside the actual sampler positions")
        stats["prefix_Q"] += len(p)
        stats["prefix_t2t"] += len(p & edit)
        stats["prefix_revocations"] += len(p & revoked)
        stats["edit_total"] += len(edit)
        stats["sample_mask_to_token"] += len(e["admitted"])

    key = "evaluate_fixed" if model == "llada21" else "_account"
    previous = getattr(worker, key)
    if model == "llada21":
        def tap(*args, **kwargs):
            result = previous(*args, **kwargs)
            for e in result["trace"]:
                account(e)
                stats["drain_F"] += int(e["drain_only"])
            stats["output_ids_sha256"] = sha256(json.dumps(result["output_ids"], separators=(",",":")).encode()).hexdigest()
            return result
    else:
        def tap(roles, event, offer):
            retval = previous(roles, event, offer)
            if event["route"] == "ordinary":
                for e in event["parts"]:
                    account(e)
                stats["drain_F"] += int(event["shared_plan"]["drain_only"])
            return retval
    setattr(worker, key, tap)
    try:
        yield stats
    finally:
        setattr(worker, key, previous)


def check_triplet(records):
    if len(records) != 3 or {r["arm"] for r in records} != set(ARMS):
        raise ValueError("unpaired mechanism test")
    for k in ("model", "offer", "index", "version", "board", "seed", "prompt_sha256"):
        if len({r[k] for r in records}) != 1:
            raise ValueError(f"triplet mismatch: {k}")
    first = records[0]["probe"]["first"]
    if any(r["probe"]["first"] != first for r in records):
        raise ValueError("trajectory differed before treatment")
    for r in records:
        probe, roles = r["probe"], r["roles"]
        billed_prefix = roles.get("prefix_Q", 0) if r["model"] == "gemma" else roles["clean_resolved"]
        expected = probe["reserved_seats"] if r["arm"] == "prefix" else 0
        if billed_prefix != expected:
            raise ValueError("measured physical prefix bill disagrees with treatment")
        writes = r["writes"]
        if (writes["prefix_Q"] != expected or
                writes["prefix_t2t"] + writes["prefix_revocations"] > writes["prefix_Q"]):
            raise ValueError("invalid prefix edit attribution")
        if r["status"] == "complete" and not r.get("output_ids_sha256"):
            raise ValueError("missing complete output fingerprint")
    if first is None:
        keys = ("status", "correct", "N", "F", "Q", "max_decoder_Q", "roles", "writes", "output_ids_sha256")
        if (any(any(r.get(k) != records[0].get(k) for k in keys) for r in records)
                or any(r["probe"]["unexposed_tape"] != records[0]["probe"]["unexposed_tape"] for r in records)):
            raise ValueError("unexposed controls disagree")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=("register", "run", "preflight"))
    p.add_argument("--base", required=True, type=Path)
    for arg in ("checkpoint", "samples"):
        p.add_argument("--"+arg, type=Path)
    p.add_argument("--board", choices=DEVICES)
    p.add_argument("--model", choices=OFFERS)
    p.add_argument("--version")
    a = p.parse_args()
    reg = json.loads(a.base.read_bytes())
    plan = registration(reg)
    version = identity(plan)
    if a.cmd == "register":
        print(json.dumps(plan, indent=2, sort_keys=True))
        print(version, file=sys.stderr)
        return
    if not (a.version == version and a.model and a.board and a.samples and a.checkpoint):
        p.error("mismatched registration or missing run arguments")
    check_sources(reg)
    tasks = load_tasks(a.samples)
    shard = shard_cases(json.loads(MANIFEST.read_bytes())["samples"], a.board, "screen18")
    if a.cmd == "preflight":
        print(f"Validated mechanism test: {a.board}/{a.model}, 3 x 2 x 3, {version}", file=sys.stderr)
        return
    check_board(a.board)
    with redirect_stdout(sys.stderr):
        worker = import_module("experiments.t2t.serving_slots." + WORKERS[a.model])
        handle = worker.load(a.checkpoint)
    for j, task in enumerate(shard):
        for t, q in enumerate(OFFERS[a.model]):
            pack = []
            for k in range(3):
                arm = ARMS[(j+t+k) % 3]
                with redirect_stdout(sys.stderr), instrument(arm) as probe, observe_writes(a.model, worker) as writes:
                    res = worker.evaluate(handle, tasks[task["index"]], q, DOSE)
                record = {k: res[k] for k in ("status", "correct", "N", "F", "Q", "decoder_F", "encoder_F",
                                              "decoder_Q", "encoder_Q", "max_decoder_Q", "roles")}
                record.update({k: task[k] for k in ("index", "seed", "task_id", "dataset", "prompt_sha256")})
                digest = writes.pop("output_ids_sha256", res.get("output_ids_sha256"))
                record.update(model=a.model, board=a.board, offer=q, version=version, stage=plan["stage"],
                              arm=arm, probe=probe, writes=writes, output_ids_sha256=digest)
                pack.append(record)
            check_triplet(pack)
            for r in pack:
                print(json.dumps(r, separators=(",", ":"), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
