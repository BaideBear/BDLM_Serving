"""Seal paired MASK-order GPU comparisons; verify every cold-first replay against formal90."""
import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.t2t.serving_slots.shared_fixed_shard import identity, shard_cases
from experiments.t2t.serving_slots.shared_point_metrics import _cohort, _receipt
from tools.frozen90_six_shards import BOARDS, MANIFEST
from tools.shared_anchor_report import costs, read_receipts, source_paths
from tools.shared_mask_order import ARMS, OFFERS, register
from tools.shared_successor_report import REPLAY


def check(r, ref, model, offer):
    _receipt(r,model,offer)
    charge = costs(r)
    s = r["plan_counts"]
    if (any(type(n) is not int or n < 0 for n in s.values())
            or s["differing_calls"] > s["calls"] or s["differing_calls"] > s["different_mask_seats"]
            or s["different_mask_seats"] > s["mask_Q"]
            or (s["cold_Q"],s["mask_Q"]-s["cold_Q"],s["dirty_Q"],s["prefix_Q"])
                != tuple(charge[k] for k in ("cold_mask","warm_mask","dirty","prefix"))):
        raise ValueError("mask intervention bill does not match projected rows")
    if r["arm"] == ARMS[0] and any(r[k] != ref[k] for k in REPLAY):
        raise ValueError("cold-first arm does not replay the reference")


def audit(root, formal_root, base, board=None):
    samples = json.loads(MANIFEST.read_bytes())["samples"]
    anchor_id, formal, _ = read_receipts(formal_root,samples,base)
    version = identity(register(base))
    data, ids, deferred = {}, [], []
    for b in (board,) if board else BOARDS:
        tasks = {t["index"]: t for t in shard_cases(samples,b,"screen18")}
        ids.extend(tasks)
        for model, offers in OFFERS.items():
            paths = source_paths(root,b,model)
            if len(paths)>1:
                raise ValueError("ambiguous ledgers")
            for path in paths:
                with path.open("rb") as stream:
                    for line in stream:
                        if not line.endswith(b"\n"):
                            deferred.append([b,model])
                            break
                        r = json.loads(line)
                        i,q,arm = r["index"],r["offer"],r["arm"]
                        key = model,q,i,arm
                        if (key in data or type(i) is not int or i not in tasks or type(q) is not int or q not in offers
                                or arm not in ARMS or (r["version"],r["source"],r["stage"],r["board"],r["model"])
                                != (version,anchor_id,"mask_order18",b,model)
                                or any(r[k] != tasks[i][k] for k in ("seed","dataset","task_id","prompt_sha256"))):
                            raise ValueError("invalid or duplicate MASK-order record")
                        check(r,formal[model,q,i],model,q)
                        data[key] = r
    points = []
    for model, offers in OFFERS.items():
        for q in offers:
            if not all((model,q,i,a) in data for i in ids for a in ARMS):
                continue
            rows = {arm:[data[model,q,i,arm] for i in ids] for arm in ARMS}
            p = dict(model=model,offer=q,arms={})
            for arm, entries in rows.items():
                counts = Counter()
                for r in entries:
                    counts.update(r["plan_counts"])
                p["arms"][arm] = dict(**_cohort(entries), plan_counts=dict(counts))
            pairs = zip(rows[ARMS[0]],rows[ARMS[1]],strict=True)
            changes = Counter()
            for a,b in pairs:
                if not a["plan_counts"]["differing_calls"] and any(a[k]!=b[k] for k in REPLAY):
                    raise ValueError("no treatment on reference path, but paired outcome changed")
                changes["rescued"] += int(b["correct"] and not a["correct"])
                changes["lost"] += int(a["correct"] and not b["correct"])
            points.append(dict(**p,**changes))
    return dict(version=version,source=anchor_id,scope=board or "screen18",seen=len(data),
                target=sum(map(len,OFFERS.values()))*len(ids)*len(ARMS),
                replay_matches=sum(k[-1]==ARMS[0] for k in data),deferred=deferred,points=points)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root","anchor","registration"):
        p.add_argument(key,type=Path)
    p.add_argument("--board",choices=BOARDS)
    p.add_argument("--require-complete",action="store_true")
    a = p.parse_args()
    report = audit(a.root,a.anchor,json.loads(a.registration.read_bytes()),a.board)
    if a.require_complete and (report["seen"] != report["target"] or report["deferred"]):
        p.error("incomplete paired panel")
    print(json.dumps(report,indent=2,allow_nan=False))
