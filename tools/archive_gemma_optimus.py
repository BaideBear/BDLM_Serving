"""Publish only numeric observations from two scored Gemma sweep directories."""
import argparse
import csv
import json
from pathlib import Path

from result_catalog import ROOT, boolean, read


def export(ibs, obs, dest):
    tasks = json.loads((ROOT/"benchmarks/frozen30/manifest.json").read_text())["samples"]
    rows = []
    for mode, src in (("ibs",ibs),("obs",obs)):
        meta = json.loads((src/"run.json").read_text())
        wanted = {f"{mode}_{c}" for c in (4,8,16,32,64,128,256)}
        scores = read(src/"scores.csv")
        index = {(s["policy"],int(s["index"])):s for s in scores}
        if set(meta["policies"]) != wanted or len(scores) != 210 or len(index) != 210:
            raise ValueError(f"incomplete {mode} matrix")
        for policy in sorted(wanted,key=lambda p:int(p.split("_")[1])):
            c = int(policy.split("_")[1])
            for task in tasks:
                i = task["index"]
                r = json.loads((src/f"{i:02d}-{policy}.json").read_text())
                s = index[policy,i]
                if (any(r[k]!=task[k] for k in ("task_id","dataset","index","seed")) or
                    any(r[k]!=s[k] for k in ("task_id","dataset","policy","status"))):
                    raise ValueError(f"identity mismatch: {policy}, {i}")
                trace, events = r["trace"], r["events"]
                enc = [q for tag,q in events[1:] if tag=="encoder"]
                dec = [q for tag,q in events[1:] if tag=="decoder"]
                assert events[0][0]=="encoder" and all(q==256 for q in enc)
                assert dec==[t["q"] for t in trace] and len(enc)==(len(r["output_ids"])-1)//256
                assert sum(dec)+sum(enc)==r["generation_query_rows"] and len(enc)+len(dec)==r["generation_forwards"]
                assert all(int(s[k])==r[k] for k in ("output_tokens","generation_forwards","generation_query_rows",
                                                    "generation_physical_padded_rows","max_q"))
                cross = sum(t["cross"] for t in trace)
                assert mode=="obs" or cross==0
                row = dict(model="gemma",policy=policy,**{k:r[k] for k in ("index","task_id","dataset","seed",
                    "status","output_tokens","generation_forwards","generation_query_rows",
                    "generation_physical_padded_rows","truncated","eos","max_q","max_decoder_q")},
                    correct=boolean(s["correct"]),chunk=c,prefill_rows=events[0][1],decoder_forwards=len(dec),
                    decoder_query_rows=sum(dec),reencode_forwards=len(enc),reencode_query_rows=sum(enc),
                    decoder_forwards_above_chunk=sum(v>c for v in dec),cross_boundary_rows=cross)
                rows.append(row)
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    with tmp.open("w",newline="") as f:
        w = csv.DictWriter(f,rows[0],lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(dest)
    print(f"Verified {len(rows)} observations -> {dest}")


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ibs",required=True,type=Path)
    p.add_argument("--obs",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args = p.parse_args()
    export(args.ibs,args.obs,args.out)
