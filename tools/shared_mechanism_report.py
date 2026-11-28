"""Read audited prefix-seat triplets without storing traces or outputs in the published result."""
import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.t2t.serving_slots.shared_fixed_shard import identity, shard_cases
from experiments.t2t.serving_slots.shared_point_metrics import BLOCK, _cohort, _receipt
from tools.frozen90_six_shards import BOARDS, MANIFEST
from tools.shared_anchor_report import costs, source_paths
from tools.shared_prefix_mechanism import ARMS, OFFERS, check_triplet, registration


def summarize(data, models, indices):
    results = {}
    for model in models:
        table = {}
        for q in OFFERS[model]:
            points = {}
            for arm in ARMS:
                rows = [data[model, q, i, arm] for i in indices]
                writes, probe = Counter(), Counter()
                for r in rows:
                    writes.update(r["writes"])
                    probe.update({k:r["probe"][k] for k in ("reserved_seats","avoided_mask_rows","exposed_calls")})
                points[arm] = dict(**_cohort(rows), observed_writes=dict(writes), probe=dict(probe),
                                   encoder_F=sum(r["encoder_F"] for r in rows))
            contrasts = {}
            for left, right in zip(ARMS, ARMS[1:]):
                pairs = [(data[model,q,i,left],data[model,q,i,right]) for i in indices]
                contrasts[f"{right}_minus_{left}"] = dict(
                    gained_correct=sum(b["correct"] and not a["correct"] for a,b in pairs),
                    lost_correct=sum(a["correct"] and not b["correct"] for a,b in pairs),
                    same_final_ids_on_complete_pairs=sum(a["output_ids_sha256"]==b["output_ids_sha256"]
                         for a,b in pairs if a["status"]==b["status"]=="complete"),
                    both_complete=sum(a["status"]==b["status"]=="complete" for a,b in pairs),
                    macro_TPF=points[right]["macro_TPF"]-points[left]["macro_TPF"],
                    TE_percentage_points=100*(points[right]["pooled_TE"]-points[left]["pooled_TE"]))
            table[q] = dict(arms=points, contrasts=contrasts)
        results[model] = table
    return results


def audit(roots, base, focus=None):
    version = identity(registration(base))
    samples = json.loads(MANIFEST.read_bytes())["samples"]
    boards = (focus,) if focus else BOARDS
    data, deferred, counts = {}, [], Counter()
    ids = []
    for board in boards:
        shard = {t["index"]:t for t in shard_cases(samples, board, "screen18")}
        ids.extend(shard)
        for model, offers in OFFERS.items():
            for root in roots:
                files = source_paths(root, board, model)
                if len(files)>1:
                    raise ValueError(f"ambiguous ledger {root}/{board}/{model}")
                for path in files:
                    with path.open("rb") as f:
                        for line_no, line in enumerate(f,1):
                            if not line.endswith(b"\n"):
                                deferred.append((board,model,line_no))
                                break
                            try:
                                r=json.loads(line)
                                i, q, arm = r["index"], r["offer"], r["arm"]
                                key = model, q, i, arm
                                if (type(i) is not int or i not in shard or type(q) is not int or q not in offers
                                        or arm not in ARMS or key in data
                                        or (r["version"],r["stage"],r["board"],r["model"])!=(version,"mechanism18",board,model)
                                        or any(r[k]!=shard[i][k] for k in ("task_id","seed","dataset","prompt_sha256"))):
                                    raise ValueError("unregistered or duplicate mechanism record")
                                _receipt(r,model,q)
                                costs(r)
                                if (type(r["encoder_Q"]) is not int or r["encoder_Q"]!=BLOCK[model]*r["encoder_F"]
                                        or not 1<=r["max_decoder_Q"]<=q):
                                    raise ValueError("invalid physical bill")
                                if any(type(n) is not int or n<0 for n in r["writes"].values()):
                                    raise ValueError("invalid observed write counters")
                                if model=="llada21" and q==4 and r["probe"]["first"] is not None:
                                    raise ValueError("Q4 floor cannot expose a clean prefix")
                                data[key]=r
                                counts[model]+=1
                            except (ValueError,KeyError,TypeError) as e:
                                raise ValueError(f"{path}:{line_no}: {e}") from e
    for model,q,i in {k[:3] for k in data}:
        keys=[(model,q,i,a) for a in ARMS]
        if all(k in data for k in keys):
            check_triplet([data[k] for k in keys])
    complete=[m for m,qs in OFFERS.items() if counts[m]==len(ids)*len(qs)*len(ARMS)
              and not any(d[1]==m for d in deferred)]
    return dict(version=version, scope=focus or "screen18", distinct_cases=len(ids), seen=len(data),
                target=len(ids)*sum(map(len,OFFERS.values()))*len(ARMS), by_model=dict(counts),
                deferred=deferred, results=summarize(data,complete,sorted(ids)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("base",type=Path)
    p.add_argument("roots",nargs="+",type=Path,help="parent(s) with H0,H4,A0,A1,A2,A3 subdirectories")
    p.add_argument("--board",choices=BOARDS)
    p.add_argument("--require-complete",action="store_true")
    a=p.parse_args()
    out=audit(a.roots,json.loads(a.base.read_bytes()),a.board)
    if a.require_complete and (out["deferred"] or out["seen"]!=out["target"]):
        p.error("missing verified triplets")
    print(json.dumps(out,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
