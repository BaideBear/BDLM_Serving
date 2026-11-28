"""Verify fixed-2B paired receipts against the unchanged formal GPU execution."""
import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.t2t.serving_slots.shared_fixed_shard import identity, shard_cases
from experiments.t2t.serving_slots.shared_point_metrics import _cohort, _receipt
from tools.frozen90_six_shards import BOARDS, MANIFEST, OFFERS
from tools.shared_anchor_report import costs, read_receipts, source_paths
from tools.shared_successor_probe import ARMS, registration

REPLAY = ("status", "correct", "N", "F", "Q", "decoder_Q", "encoder_Q",
          "decoder_F", "encoder_F", "max_decoder_Q", "roles")


def check(r, formal, base, version, board, model, task):
    q = OFFERS[model][-1]
    if ((r["stage"], r["version"], r["source"], r["board"], r["model"], r["offer"], r["params"])
            != ("successor18", version, identity(base), board, model, q, base["params"][model])
            or r["arm"] not in ARMS
            or any(r[k] != task[k] for k in ("index", "dataset", "task_id", "seed", "prompt_sha256"))):
        raise ValueError("unexpected assignment")
    _receipt(r, model, q)
    c, m = costs(r), r["ordinary"]
    if (any(type(v) is not int or v < 0 for v in m.values())
            or m["head_Q"]+m["tail_Q"] != m["ordinary_Q"]
            or m["tail_sampled_Q"]+m["tail_dirty_Q"] != m["tail_Q"]
            or m["dual_F"] > m["ordinary_F"] or m["ordinary_F"] > r["decoder_F"]
            or m["ordinary_Q"] != r["Q"]-c["rebuild"]-c["terminal_verification"]):
        raise ValueError("inconsistent successor bill")
    if r["arm"] == "head_only":
        if m["tail_Q"] or m["tail_open_F"] or r["max_decoder_Q"] > q//2:
            raise ValueError("head-only arm used a successor")
    elif any(r[k] != formal[k] for k in REPLAY):
        raise ValueError("unchanged arm did not reproduce the formal receipt")


def audit(root, anchor, base, board=None):
    samples = json.loads(MANIFEST.read_bytes())["samples"]
    _, formal, _ = read_receipts(anchor, samples, base)
    version = identity(registration(base))
    boards = (board,) if board else BOARDS
    data, deferred, ids = {}, [], []
    for b in boards:
        tasks = {t["index"]:t for t in shard_cases(samples, b, "screen18")}
        ids.extend(tasks)
        for model in OFFERS:
            paths = source_paths(root, b, model)
            if len(paths) > 1:
                raise ValueError(f"ambiguous ledger: {b}/{model}")
            for path in paths:
                with path.open("rb") as stream:
                    for line in stream:
                        if not line.endswith(b"\n"):
                            deferred.append((b, model))
                            break
                        r = json.loads(line)
                        idx = r["index"]
                        key = (model, idx, r["arm"])
                        if key in data or idx not in tasks:
                            raise ValueError(f"duplicate or invalid key {key}")
                        check(r, formal[model, OFFERS[model][-1], idx], base, version, b, model, tasks[idx])
                        data[key] = r
    groups = {}
    for model in OFFERS:
        if not all((model, i, arm) in data for i in ids for arm in ARMS):
            continue
        rows = {arm:[data[model,i,arm] for i in ids] for arm in ARMS}
        groups[model] = g = {}
        for arm, group in rows.items():
            totals = Counter()
            for r in group:
                totals.update(r["ordinary"])
            g[arm] = dict(**_cohort(group), ordinary=dict(totals))
        pairs = list(zip(rows[ARMS[0]], rows[ARMS[1]], strict=True))
        g["rescued"] = sum(not a["correct"] and b["correct"] for a,b in pairs)
        g["lost"] = sum(a["correct"] and not b["correct"] for a,b in pairs)
    return dict(version=version, seen=len(data), target=len(ids)*len(OFFERS)*len(ARMS),
                replay_matches=sum(k[2]=="allow_successor" for k in data), deferred=deferred, results=groups)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "anchor", "base"):
        p.add_argument(key, type=Path)
    p.add_argument("--board", choices=BOARDS)
    p.add_argument("--require-complete", action="store_true")
    a = p.parse_args()
    result = audit(a.root, a.anchor, json.loads(a.base.read_bytes()), a.board)
    if a.require_complete and (result["seen"] != result["target"] or result["deferred"]):
        p.error("incomplete pairs")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
