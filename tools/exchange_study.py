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
    records, coverage = defaultdict(Counter), defaultdict(Counter)
    for task in manifest:
        for s,admit,edit in states(folder,task):
            v,r,f,t,w,c = s
            for q,cap in ((4,1),(8,1),(16,2)):
                base,_,_ = choose(*s, q, 0, rates, 1.)
                for mode,value in MODES.items():
                    chosen,reserve,offers = choose(*s,q,cap,rates,value)
                    assert set(reserve) == set(chosen)-set(base) and len(chosen)==len(base)
                    target = edit - f
                    cover = coverage[q,cap,mode]
                    cover.update(states=1, old_targets=len(target), base_old_hits=len(target.intersection(base)),
                                 selected_old_hits=len(target.intersection(chosen)), newly_covered=len(target.intersection(reserve)),
                                 lost_old_hits=len(target.intersection(set(base)-set(chosen))),
                                 mask_admissions=len(admit), base_admits=len(admit.intersection(base)),
                                 selected_admits=len(admit.intersection(chosen)), selected_Q=len(chosen),
                                 same_state_base_Q=len(base))
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
    for z in coverage.values():
        assert z["old_targets"] == 191 and z["selected_Q"] == z["same_state_base_Q"]
        assert z["newly_covered"]-z["lost_old_hits"] == z["selected_old_hits"]-z["base_old_hits"]
    return ([dict(q=q,cap=e,mode=m,victim=v,**{k:z[k] for k in cols})
             for (q,e,m,v),z in sorted(records.items())],
            [dict(q=q,cap=e,mode=m,**z) for (q,e,m),z in sorted(coverage.items())])


def audit_pair(arm, base):
    for k in ("model","index","seed","task_id","dataset","prefill_rows"):
        if arm[k] != base[k]:
            raise ValueError(f"Pair metadata mismatch: {k}")
    if arm["status"] != "success" or base["status"] != "success":
        raise ValueError("Failed request in paired audit")
    events, controls = arm["events"], base["events"]
    for tick,e in enumerate(events):
        reserved = e.get("reserved_extra_edit_positions", [])
        if tick >= len(controls):
            raise ValueError("Missing control round before first swap")
        b = controls[tick]
        if not reserved:
            if e != b:
                raise ValueError(f"Pre-swap event diverged: {arm['index']} at {tick}")
            continue
        if b["stage"] != "selected" or e["stage"] != "selected":
            raise ValueError("Swap outside sparse selection")
        for field in ("block","context_version","step_in_block","hard_clock","mandatory_fresh",
                      "masked_before","actual_q","physical_q","layer_physical_q","F","prefix_kv_length"):
            if e[field] != b[field]:
                raise ValueError(f"First-intervention mismatch: {field}")
        picked,neutral = set(e["absolute_positions"]),set(b["absolute_positions"])
        if set(reserved) != picked-neutral or len(neutral-picked) != len(reserved):
            raise ValueError("Reserved seats are not exactly the new physical rows")
        displaced = b["selected_masked_before"]-e["selected_masked_before"]
        if displaced not in (0,len(reserved)):
            raise ValueError("Unexpected mixed MASK and old victims")
        return dict(prefix_events=tick, anchor_swaps=len(reserved), victim="mask" if displaced else "old",
                    anchor_mask_lost=displaced, anchor_added_strict=len(e["reserved_strict_edit_positions"]),
                    anchor_net_accept=e["accepted"]-b["accepted"], anchor_net_strict=e["hard_strict_edits"]-b["hard_strict_edits"])
    for k in ("output_ids","output_tokens","generation_forwards","generation_query_rows","events"):
        if arm[k] != base[k]:
            raise ValueError(f"No-swap request diverged: {k}")
    return dict(prefix_events=len(events),anchor_swaps=0,victim="none",anchor_mask_lost=0,
                anchor_added_strict=0,anchor_net_accept=0,anchor_net_strict=0)


def anchors(runs,control,manifest,rates_path):
    records = []
    for run in runs:
        config = json.loads((run/"run.json").read_text())
        assert config["exchange_rates_sha256"] == sha(rates_path)
        assert json.loads((run/"qualified.json").read_text()) == dict(model="llada21",full_width=32,matched=30)
        for arm,name in zip(config["arms"],config["policies"],strict=True):
            q = arm["budget"]
            cname = f"t2t_q{q}_f0_m{q}_hard_noedit"
            assert arm["selection"] in MODES and json.loads((run/f"complete-{name}.json").read_text()) == dict(policy=name,requests=30)
            for sample in manifest:
                i = sample["index"]
                new = json.loads((run/f"{i:02d}-{name}.json").read_text())
                base = json.loads((control/f"{i:02d}-{cname}.json").read_text())
                assert all(new[k] == sample[k] for k in ("index","seed","task_id","dataset"))
                records.append(dict(q=q,policy=name,index=i,dataset=sample["dataset"],task_id=sample["task_id"],
                                    **audit_pair(new,base)))
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("fit","replay","anchors"))
    for name in ("dir","rates"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--out",type=Path)
    p.add_argument("--coverage",type=Path)
    p.add_argument("--run",type=Path,action="append")
    a = p.parse_args()
    frozen = json.loads(MANIFEST.read_text())["samples"]
    if a.action == "fit":
        a.rates.write_text(json.dumps(fit(a.dir,frozen),indent=2)+"\n")
    elif a.action == "replay":
        if a.out is None:
            p.error("replay requires --out")
        offers,counts = replay(a.dir,frozen,decode(json.loads(a.rates.read_text())))
        table(a.out,offers)
        if a.coverage:
            table(a.coverage,counts)
    else:
        if not a.run or not a.out:
            p.error("anchors requires --run and --out; --dir is the same-budget zero-reserve run")
        table(a.out,anchors(a.run,a.dir,frozen,a.rates))
