"""Fixed tight offers: compare cold-first with earliest unresolved MASK per physical block."""
import argparse
from collections import Counter
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
from experiments.t2t.serving_slots.shared_point_metrics import _receipt
from tools.shared_anchor_report import costs

ARMS = ("cold_first", "left_first")
OFFERS = {"llada21": (4,8), "gemma": (32,64)}


def alternative(plan, head, nxt):
    """Replace only MASK identities within existing quotas; no changes to debt, prefix or block admission."""
    selected = set(plan.mask)
    if not selected:
        return plan
    chosen, cold = [], []
    for block in (head, nxt):
        if block is None:
            continue
        n = len(selected & block.masks)
        if n:
            m = sorted(block.masks-block.dirty)[:n]
            chosen.extend(m)
            cold.extend(i for i in m if i not in block.resident)
    sampled = tuple(sorted((*chosen, *plan.prefix)))
    paid = tuple(sorted((*sampled, *plan.dirty)))
    if len(set(paid)) != len(plan.rows) or len(chosen) != len(plan.mask):
        raise ValueError("MASK ordering changed the physical bill at the same state")
    return replace(plan, mask=tuple(chosen), cold=tuple(cold), sampled=sampled, rows=paid)


@contextmanager
def instrument(arm):
    if arm not in ARMS:
        raise ValueError("invalid MASK policy")
    original = rows.plan_rows
    stats = Counter()
    def select(q,b,head,next_block=None,*,opened=False,params):
        plan = original(q,b,head,next_block,opened=opened,params=params)
        other = alternative(plan,head,next_block)
        out = plan if arm == ARMS[0] else other
        displaced = len(set(plan.mask)-set(other.mask))
        stats.update(calls=1, differing_calls=int(displaced>0), different_mask_seats=displaced,
                     mask_Q=len(out.mask), cold_Q=len(out.cold), prefix_Q=len(out.prefix), dirty_Q=len(out.dirty))
        return out
    rows.plan_rows = select
    try:
        yield stats
    finally:
        rows.plan_rows = original


def register(base):
    if (base["stage"],base["cap"]) != ("formal90",2048):
        raise ValueError("expected the registered formal decoder")
    return dict(format="shared-mask-order-v1",stage="mask_order18",source=identity(base),cases="screen18",
                wrapper=sha256(Path(__file__).read_bytes()).hexdigest(),arms=ARMS,offers=OFFERS,
                cap=2048,params=base["params"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=("register","preflight","run"))
    p.add_argument("--base", required=True,type=Path)
    for k in ("samples","checkpoint"):
        p.add_argument("--"+k,type=Path)
    p.add_argument("--board",choices=DEVICES)
    p.add_argument("--model",choices=OFFERS)
    p.add_argument("--version")
    a = p.parse_args()
    base = json.loads(a.base.read_bytes())
    panel = register(base)
    version = identity(panel)
    if a.cmd == "register":
        print(json.dumps(panel,indent=2,sort_keys=True)); print(version,file=sys.stderr)
        return
    if not all((a.board,a.model,a.samples,a.checkpoint)) or a.version != version:
        p.error("missing arguments or incorrect registration")
    check_sources(base)
    tasks = load_tasks(a.samples)
    shard = shard_cases(json.loads(MANIFEST.read_bytes())["samples"],a.board,"screen18")
    if a.cmd == "preflight":
        print(f"qualified inputs: {a.board}/{a.model}, 3 cases x 2 offers x 2 arms",file=sys.stderr)
        return
    check_board(a.board)
    params = rows.Params(**{k:Fraction(v) for k,v in base["params"][a.model].items()})
    with redirect_stdout(sys.stderr):
        worker = import_module("experiments.t2t.serving_slots."+WORKERS[a.model])
        handle = worker.load(a.checkpoint)
    for j,t in enumerate(shard):
        for z,q in enumerate(OFFERS[a.model]):
            for k in range(len(ARMS)):
                arm = ARMS[(k+j+z)%len(ARMS)]
                with redirect_stdout(sys.stderr), instrument(arm) as stats:
                    result = worker.evaluate(handle,tasks[t["index"]],q,params)
                r = {k:result[k] for k in ("status","correct","N","F","Q","decoder_F","encoder_F",
                                           "decoder_Q","encoder_Q","max_decoder_Q","roles")}
                r.update({k:t[k] for k in ("index","seed","dataset","task_id","prompt_sha256")})
                r.update(stage=panel["stage"],board=a.board,model=a.model,version=version,
                         source=panel["source"],offer=q,arm=arm,plan_counts=dict(stats))
                _receipt(r,a.model,q)
                bills = costs(r)
                if ((stats["cold_Q"],stats["mask_Q"]-stats["cold_Q"],stats["dirty_Q"],stats["prefix_Q"])
                        != tuple(bills[k] for k in ("cold_mask","warm_mask","dirty","prefix"))):
                    raise ValueError("selected roles differ from the measured GPU bill")
                print(json.dumps(r,separators=(",",":"),allow_nan=False),flush=True)


if __name__ == "__main__":
    main()
