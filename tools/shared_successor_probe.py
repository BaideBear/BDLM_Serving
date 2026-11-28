"""A paired, fixed-2B test of admitting the successor, with no other policy changes."""
import argparse
from collections import Counter
from contextlib import contextmanager, redirect_stdout
from fractions import Fraction
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import sys

from bdlm_serving.methods.t2t import unified_rows
from experiments.t2t.serving_slots.shared_fixed_shard import (
    DEVICES, MANIFEST, OFFERS, WORKERS, check_board, check_sources, identity, load_tasks, shard_cases,
)
from experiments.t2t.serving_slots.shared_point_metrics import _receipt
from tools.shared_anchor_report import costs

ARMS=("head_only","allow_successor")


@contextmanager
def instrument(arm):
    if arm not in ARMS:
        raise ValueError("unregistered successor arm")
    base=unified_rows.plan_rows
    info=Counter()
    def select(q,b,head,next_block=None,*,opened=False,params):
        if q!=2*b:
            raise ValueError("only registered 2B offers are allowed")
        if arm=="head_only" and (opened or (next_block and (next_block.dirty or next_block.resident))):
            raise ValueError("cannot suppress an established successor or its debt")
        nxt=next_block if arm=="allow_successor" else None
        out=base(q,b,head,nxt,opened=opened,params=params)
        boundary=head.start+head.size
        head_q=sum(i<boundary for i in out.rows)
        tail_q=len(out.rows)-head_q
        if (arm=="head_only" and (tail_q or out.open_next)) or not set(out.dirty)<=set(out.rows):
            raise ValueError("illegal successor projection or unprojected dirty row")
        info.update(ordinary_F=1,ordinary_Q=len(out.rows), head_Q=head_q, tail_Q=tail_q,
                    dual_F=int(bool(tail_q and head_q)), tail_sampled_Q=sum(i>=boundary for i in out.sampled),
                    tail_dirty_Q=sum(i>=boundary for i in out.dirty), tail_open_F=int(out.open_next and not opened))
        return out
    unified_rows.plan_rows=select
    try:
        yield info
    finally:
        unified_rows.plan_rows=base


def registration(base):
    if base["stage"]!="formal90" or base["cap"]!=2048:
        raise ValueError("unregistered base")
    return dict(source=identity(base), wrapper=sha256(Path(__file__).read_bytes()).hexdigest(),
                format="shared-successor-v1",stage="successor18",cases="screen18",
                arms=ARMS,offers={m:v[-1] for m,v in OFFERS.items()},params=base["params"],cap=2048)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd",choices=("register","preflight","run"))
    p.add_argument("--base",type=Path,required=True)
    p.add_argument("--version")
    p.add_argument("--samples",type=Path)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--board",choices=DEVICES)
    p.add_argument("--model",choices=OFFERS)
    a=p.parse_args()
    base=json.loads(a.base.read_bytes())
    reg=registration(base)
    version=identity(reg)
    if a.cmd=="register":
        print(json.dumps(reg,indent=2,sort_keys=True)); print(version,file=sys.stderr)
        return
    if version!=a.version or not all((a.samples,a.checkpoint,a.model,a.board)):
        p.error("incorrect registration or missing worker arguments")
    check_sources(base)
    tasks=load_tasks(a.samples)
    shard=shard_cases(json.loads(MANIFEST.read_bytes())["samples"],a.board,"screen18")
    q=OFFERS[a.model][-1]
    params=unified_rows.Params(**{k:Fraction(v) for k,v in base["params"][a.model].items()})
    if a.cmd=="preflight":
        print(f"validated {a.board}/{a.model} Q{q} x three cases x two arms: {version}",file=sys.stderr)
        return
    check_board(a.board)
    with redirect_stdout(sys.stderr):
        worker=import_module("experiments.t2t.serving_slots."+WORKERS[a.model])
        handle=worker.load(a.checkpoint)
    for j,task in enumerate(shard):
        pair=[]
        for k in range(2):
            arm=ARMS[(j+k)%2]
            with redirect_stdout(sys.stderr), instrument(arm) as info:
                r=worker.evaluate(handle,tasks[task["index"]],q,params)
            keys=("status","correct","N","F","Q","decoder_Q","encoder_Q","decoder_F","encoder_F","max_decoder_Q","roles")
            rec={name:r[name] for name in keys}
            rec.update({name:task[name] for name in ("index","seed","task_id","dataset","prompt_sha256")})
            rec.update(stage=reg["stage"], model=a.model, board=a.board, version=version, source=identity(base),
                       offer=q, arm=arm, params=base["params"][a.model], ordinary=dict(info))
            _receipt(rec,a.model,q)
            role=costs(rec)
            if (info["ordinary_Q"]!=r["Q"]-role["terminal_verification"]-role["rebuild"]
                    or info["head_Q"]+info["tail_Q"]!=info["ordinary_Q"]
                    or (arm=="head_only" and r["max_decoder_Q"]>q//2)):
                raise ValueError("physical bill disagrees with the 2B intervention")
            pair.append(rec)
        for rec in pair:
            print(json.dumps(rec,separators=(",",":"),allow_nan=False),flush=True)


if __name__=="__main__":
    main()
