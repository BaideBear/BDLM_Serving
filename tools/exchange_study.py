"""Calibrate before opening the evaluation paths, then audit exchanged physical seats."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from bdlm_serving.evaluation.frozen import MANIFEST, sha
from bdlm_serving.methods.t2t.exchange import SCHEMA, MODES, choose, decode, position_keys
from tools.t2t_results import table


def states(folder, sample):
    with gzip.open(folder / f"{sample['index']:02d}.trace.jsonl.gz", "rt") as stream:
        rows = map(json.loads, stream)
        meta, prompt = next(rows), next(rows)
        assert all(meta[k] == sample[k] for k in ("index","dataset","task_id","seed"))
        last = prior = None
        mask = prompt["mask_id"]
        for row in rows:
            block, before = row["block"], row["before"]
            new = block != last
            if new:
                valid = list(range(max(0,prompt["prompt_length"]-block),32))
                fresh, strict, writes, clock = set(), set(), [-1]*32, 0
            else:
                assert before == prior
            clock += 1
            resolved = [x != mask for x in before]
            after = row["after"]
            assert len(after) == 32 and row["accepted"] == [x != mask for x in after]
            changed = {i for i in valid if before[i] != after[i]}
            admitted = {i for i in changed if not resolved[i] and after[i] != mask}
            edited = {i for i in changed if resolved[i] and after[i] != mask}
            if not new and any(not resolved[i] for i in valid):
                yield (valid,resolved,fresh,strict,writes,clock), admitted, edited
            for i in changed:
                writes[i] = clock
            fresh, strict, prior, last = admitted | edited, edited, after, block


def fit(calibration, manifest):
    src = calibration/"weights.json"
    train = json.loads(src.read_text())["samples"]
    unique = lambda data,k: {(r["dataset"],r[k]) for r in data}
    assert Counter(s["dataset"] for s in train) == {k:8 for k in ("gsm8k","math500","humaneval")}
    assert len(unique(train,"source_index")) == len({s["prompt_sha256"] for s in train}) == 24
    assert unique(train,"source_index").isdisjoint(unique(manifest,"source_index"))
    assert {s["prompt_sha256"] for s in train}.isdisjoint(s["prompt_sha256"] for s in manifest)
    counts, hits, n, positives = Counter(),Counter(),Counter(),Counter()
    for s in train:
        for (v,r,f,t,w,c),admit,edit in states(calibration,s):
            for i,key in position_keys(v,r,f,t,w,c).items():
                group = key[0]
                event = i in (edit if group == "old" else admit)
                counts[key] += 1
                hits[key] += event
                n[group] += 1
                positives[group] += event
    prior = {k:positives[k]/n[k] for k in n}
    assert positives["old"] == json.loads(src.read_text())["calibration_old_positives"]
    return dict(schema=SCHEMA, calibration_manifest_sha256=sha(src),
        samples=24, prior_strength=16, counts=n, positives=positives, fallback=prior,
        rates=[dict(key=k,n=counts[k],hits=hits[k],probability=(hits[k]+16*prior[k[0]])/(counts[k]+16))
               for k in sorted(counts)])


def replay(folder, manifest, rates):
    assert json.loads((folder/"verified.json").read_text()) == dict(model="llada21",matched_requests=30)
    records = defaultdict(Counter)
    for task in manifest:
        for s,admit,edit in states(folder,task):
            v,r,f,t,w,c = s
            for q,cap in ((4,1),(8,1),(16,2)):
                base,_,_ = choose(*s, q, 0, rates, 1.)
                for mode,value in MODES.items():
                    chosen,reserve,offers = choose(*s,q,cap,rates,value)
                    assert set(reserve) == set(chosen)-set(base) and len(chosen)==len(base)
                    for victim_type in ("mask","old"):
                        key = q,cap,mode,victim_type
                        totals = records[key]
                        totals["states"] += 1
                        for candidate,victim,swap in offers:
                            if ("old" if r[victim] else "mask") != victim_type:
                                continue
                            totals["offers"] += 1
                            totals["swaps"] += swap
                            h = int(candidate in edit)
                            cost = int(victim in (edit if r[victim] else admit))
                            totals["candidate_event_swapped"] += swap*h
                            totals["victim_event_swapped"] += swap*cost
                            totals["both_events_swapped"] += swap*h*cost
                            totals["candidate_event_declined"] += (not swap)*h
                            totals["victim_event_declined"] += (not swap)*cost
    cols = ("states","offers","swaps","candidate_event_swapped","victim_event_swapped",
            "both_events_swapped","candidate_event_declined","victim_event_declined")
    return [dict(q=q,cap=e,mode=m,victim=v,**{k:z[k] for k in cols})
            for (q,e,m,v),z in sorted(records.items())]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("fit","replay"))
    for name in ("dir","rates"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--out",type=Path)
    a = p.parse_args()
    frozen = json.loads(MANIFEST.read_text())["samples"]
    if a.action == "fit":
        a.rates.write_text(json.dumps(fit(a.dir,frozen),indent=2)+"\n")
    else:
        if a.out is None:
            p.error("replay requires --out")
        table(a.out,replay(a.dir,frozen,decode(json.loads(a.rates.read_text()))))
