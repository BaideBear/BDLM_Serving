"""Audit a completed frozen90 external-slot experiment and score both phases."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from itertools import groupby
from random import Random

from bdlm_serving.evaluation.frozen import sha
from bdlm_serving.methods.t2t.slot_trace import parse_offer
from experiments.t2t.prefix_window.score_run import snapshot
from tools.prefix_cohorts import score_files


def gemma_stages(row):
    """Account for terminal authority separately from the width of a decoder event."""
    events=row["events"]
    phases=("bootstrap","selected","native_offer","verify","reencode")
    c=Counter({f"{p}_{field}":0 for p in phases for field in ("F","Q")})
    c.update(blocks=0,native_closes=0,verify_closes=0,bootstrap_closes=0,
             M2T=0,strict_T2T=0,revoked=0,selected_stop_candidates=0)
    block_width=256
    previous=None
    for start,group in groupby(events,key=lambda e:e["block"]):
        group=list(group)
        if previous is not None and start!=previous+block_width:
            raise ValueError("Unordered Gemma canvases")
        previous=start
        c["blocks"]+=1
        if group[0]["stage"]!="bootstrap":
            raise ValueError("Missing full bootstrap")
        resolved=0
        for index,e in enumerate(group):
            p=e["stage"]
            if p not in phases[:-1] or (e["finished"] or e["terminal_clock"]) != (index==len(group)-1):
                raise ValueError("Unaccounted or premature block completion")
            if (e["edited"]!=e["hard_strict_edits"] or
                    resolved+e["admitted"]-e["revoked"]!=e["resolved"]):
                raise ValueError("Bad admission/edit/revocation counters")
            resolved=e["resolved"]
            c.update(M2T=e["admitted"],strict_T2T=e["edited"],revoked=e["revoked"])
            c[p+"_F"]+=1
            c[p+"_Q"]+=e["q"]
            if p=="selected" and e["stop_candidate"]:
                c["selected_stop_candidates"]+=1
        last=group[-1]["stage"]
        if last not in ("bootstrap","verify","native_offer"):
            raise ValueError("A sparse selection finalized a block")
        c[("native" if last=="native_offer" else last)+"_closes"]+=1
    enc=[q for tag,q in row["physical_events"][1:] if tag=="encoder"]
    if len(enc)!=c["blocks"]-1 or any(q!=block_width for q in enc):
        raise ValueError("Missing inter-block encoder forward")
    c.update(reencode_F=len(enc),reencode_Q=sum(enc))
    if (any((c[p+"_Q"],c[p+"_F"])!=(row["q_by_phase"].get(p,0),row["forwards_by_phase"].get(p,0)) for p in phases)
            or c["strict_T2T"]!=row["t2t_edits"]):
        raise ValueError("Phase work does not reproduce the physical ledger")
    return c


def audit(row, pattern):
    prev = None
    c = Counter(requests=1, switched_requests=0, up=0, down=0, selections=0,
                mandatory=0, masks=0, backfill=0, prefix=0, charged_other=0,
                overflow_rows=0, underfill_rows=0, offered_rows=0)
    ll = row["model"] == "llada21"
    if ll:
        work = [(e["stage"], e["physical_q"]) for e in row["events"] if e["phase"] != "prefill"]
    else:
        raw = row["physical_events"]
        if tuple(raw[0]) != ("encoder", row["prefill_rows"]):
            raise ValueError("Missing observed Gemma prompt prefill")
        if [q for tag,q in raw[1:] if tag == "decoder"] != [e["q"] for e in row["events"]]:
            raise ValueError("Gemma decoder work disagrees with observer")
        work = raw[1:]
    width = 32 if ll else 256
    slots = [parse_offer(str(field),width) for field in pattern]
    full = 0
    for e in row["events"]:
        if e["stage"] not in ("selected","native_offer"):
            if "slot_tick" in e or "slot_offer" in e:
                raise ValueError("Consumed a budget on a non-selected event")
            continue
        s = slots[c["selections"] % len(pattern)]
        q,k,mode = s.plan(width)
        if e["slot_tick"] != c["selections"] or e["slot_offer"] != q:
            raise ValueError("Slot clock, phase or offer changed")
        up, down = int(prev is not None and q > prev), int(prev is not None and q < prev)
        prev = q
        if e["stage"] == "native_offer":
            if ll:
                start=e["block"]*width
                valid=(e["physical_q"]==width and e["absolute_positions"]==list(range(start,start+width))
                       and e["layer_physical_q"] and all(v==width for v in e["layer_physical_q"])
                       and not e["approximate_same_block_kv"] and 0<=e["mandatory_fresh"]<=width)
            else:
                valid=(e["q"]==width and e["positions"]==list(range(width))
                       and e["mandatory_fresh"]==e["fresh_selected"]==e["available_fresh"])
            if (not valid or not s.native_full or e["overflow"] or "prefix_plan" in e):
                raise ValueError("Explicit native budget did not execute a complete canvas")
            full += width
            c.update(selections=1,up=up,down=down,offered_rows=q,native_offers=1,native_rows=width)
            continue
        p = e["prefix_plan"]
        if (s.native_full or p["quota"] != k or p["budget"] != q
                or e.get("slot_prefix_mode") != (mode if k else None)):
            raise ValueError("Wrong allocation for this slot")
        obs = ([pos-p["block_start"] for pos in e["absolute_positions"]] if ll else e["positions"])
        sets = [set(p[k]) for k in ("fresh_rows", "zero_mask_rows", "zero_backfill_rows", "added_rows")]
        actual = len(obs)
        if (not 0 <= p["prefix_end"] <= width or any(not 0 <= i < width for i in obs)
                or len(sets[3]) > k or not sets[3] <= set(p["eligible_prefix_rows"])
                or any(not 0 <= i < p["prefix_end"] for i in sets[3])
                or p["evicted_rows"] or sum(map(len, sets)) != actual or set.union(*sets) != set(obs)
                or actual != p["actual_width"] or obs != p["selected_rows"]
                or len(sets[0]) != e["mandatory_fresh"] or
                len(sets[1]) != e["selected_masked_before" if ll else "selected_mask"]
                or (ll and (not e["layer_physical_q"] or any(v != actual for v in e["layer_physical_q"])))
                or e["physical_q" if ll else "q"] != actual
                or max(0, len(sets[0])-q) != p["overflow"] or max(0, actual-q) != p["overflow"]):
            raise ValueError("Selected physical rows or protected quota disagree")
        for key, s in zip(("mandatory", "masks", "backfill", "prefix"), sets):
            c[key] += len(s)
        c.update(selections=1, up=up, down=down, overflow_rows=max(0, actual-q),
                 underfill_rows=max(0, q-actual), offered_rows=q)
    c["switched_requests"] = int(c["up"] + c["down"] > 0)
    selected = sum(c[k] for k in ("mandatory","masks","backfill","prefix"))
    f, total = len(work), sum(v for _, v in work)
    if (selected != row["q_by_phase"].get("selected", 0) or
            f != row["generation_forwards"] or total != row["generation_physical_padded_rows"]
            or total != row["generation_query_rows"] or
            full != row["q_by_phase"].get("native_offer",0) or
            selected + full != c["offered_rows"] + c["overflow_rows"] - c["underfill_rows"]):
        raise ValueError("Physical work mismatch")
    c.update(charged_other=total-selected-full, N=row["output_tokens"], F=f, physical_Q=total)
    return c


def first_changed_offer(reference, row):
    """Compare full observed histories until a physical selection changes policy."""
    if any(reference[k] != row[k] for k in ("model","index","task_id","dataset","seed","prefill_rows")):
        raise ValueError("Different requests in offer ablation")
    for i,(before,after) in enumerate(zip(reference["events"],row["events"])):
        if before.get("slot_offer")!=after.get("slot_offer"):
            fields=("block","clock","slot_tick","available_fresh")
            if (before.get("slot_tick") is None or after.get("slot_tick") is None
                    or any(before[k]!=after[k] for k in fields)):
                raise ValueError("Different pre-step state at first changed offer")
            return i
        if before!=after:
            raise ValueError("Divergence before first changed offer")
    keys = ("events","physical_events","output_ids","status","eos","truncated",
            "generation_forwards","generation_physical_padded_rows","t2t_edits")
    if any(reference[k] != row[k] for k in keys):
        raise ValueError("Unchanged offers did not reproduce observed termination and work")
    return None


def prefix_ablation(reference,row):
    """Verify common observed history and exact block-local random additions."""
    if any(reference[k]!=row[k] for k in ("model","index","task_id","dataset","seed","prefill_rows")):
        raise ValueError("Different requests in prefix ablation")
    seen=False
    c=Counter(prefix_exposed_requests=0,prefix_exposed_rounds=0,prefix_strict_edits=0,prefix_revokes=0)
    for i,e in enumerate(row["events"]):
        p=e.get("prefix_plan")
        add=[]
        if p:
            # This experiment has just one sparse plan; H is native full-canvas.
            if p["budget"]!=128 or p["quota"]!=32 or e.get("slot_prefix_mode")!="hard_prefix_slack_random":
                raise ValueError("Unregistered sparse prefix arm")
            pool=list(p["eligible_prefix_rows"])
            count=min(32,len(pool),max(0,128-len(p["zero_rows"]))) if p["zero_rows"] else 0
            Random(f"physical-prefix:{row['seed']}:{e['block']}:{e['clock']}").shuffle(pool)
            add=sorted(pool[:count])
            if (e["block"]!=p["block_start"] or e["clock"]!=p["clock"] or
                    add!=p["added_rows"] or p["evicted_rows"] or set(add)&set(p["zero_rows"]) or
                    sorted(p["zero_rows"]+add)!=e["positions"]):
                raise ValueError("Prefix dose, seed, block, or donor mismatch")
            actions=e["prefix_actions"]
            if actions["locked"] or not (set(actions["proposed_strict"])|set(actions["proposed_revoked"]))<=set(add):
                raise ValueError("Unexpected prefix action")
            c.update(prefix_exposed_rounds=int(bool(add)),prefix_strict_edits=len(actions["proposed_strict"]),
                     prefix_revokes=len(actions["proposed_revoked"]))
        if seen:
            continue
        if i>=len(reference["events"]):
            raise ValueError("Unexplained additional event")
        old=reference["events"][i]
        if add:
            if (any(e.get(k)!=old.get(k) for k in ("stage","block","clock","available_fresh","slot_tick","slot_offer"))
                    or old["positions"]!=p["zero_rows"]
                    or any(old["prefix_plan"][k]!=p[k] for k in
                        ("fresh_rows","zero_mask_rows","zero_backfill_rows","eligible_prefix_rows","prefix_end"))):
                raise ValueError("First exposure does not extend the same zero plan")
            seen=True
        else:
            # Discard only added diagnostics, not actions, positions or realized work.
            def canonical(event):
                d={k:v for k,v in event.items() if k not in
                   ("slot_prefix_mode","edit_slots","prefix_actions","prefix_canvas_sha256","prefix_draft_sha256")}
                for k in ("prefix_plan","prefix_empty_verify_plan"):
                    if k in d:
                        d[k]={field:v for field,v in d[k].items() if field!="quota"}
                return d
            if canonical(old)!=canonical(e):
                raise ValueError("Divergence before first physical prefix injection")
    if not seen and (len(reference["events"])!=len(row["events"]) or
                     any(reference[k]!=row[k] for k in ("physical_events","output_ids","eos","truncated","status"))):
        raise ValueError("No-op prefix request changed output or work")
    c["prefix_exposed_requests"]=int(seen)
    return c


def report(run, cases, out, reference_switch_run=None, thinning_run=None):
    meta = json.loads((run/"run.json").read_text())
    model = meta["model"]
    a,b = {"gemma": (32,64), "llada21": (8,16)}[model]
    base = ((a,), (b,), (a,a,b,b), (b,b,a,a))
    def labels_for(phases):
        return ["slottrace_" + "_".join(map(str,t)) for t in phases]
    mid,high=(128,"256n") if model=="gemma" else (16,"32n")
    native = ((mid,),(high,),(mid,mid,high,high),(high,high,mid,mid))
    sparse_high = ((128,),(256,),(128,128,256,256),(256,256,128,128))
    thin = ((128,),("256n",),("256n",128,128,128),(128,"256n",128,128))
    u,h="128u32","256n"
    prefix_high=((u,),(h,),(h,h,u,u),(u,u,h,h),(h,u,u,u),(u,h,u,u))
    dose = ((32,), (64,), ("32r8",), ("32r8","32r8",64,64), (64,64,"32r8","32r8"),
            (32,)*8+(64,)*8, (64,)*8+(32,)*8)
    options = [base,native] + ([dose,sparse_high,thin,prefix_high] if model == "gemma" else [])
    phases = next((t for t in options if labels_for(t) == meta["policies"]),None)
    if (phases is None or (phases in (dose,sparse_high,thin,prefix_high)) != bool(reference_switch_run)
            or (phases==prefix_high)!=bool(thinning_run)):
        raise ValueError("Unregistered slot experiment")
    labels = labels_for(phases)
    sources = [run] * len(labels)
    pairs = [(0,1),(0,2),(1,2),(0,3),(1,3),(2,3)]
    prefix_peers={}
    if phases == dose:
        ref = json.loads((reference_switch_run/"run.json").read_text())
        if (ref["model"],ref["policies"]) != (model,labels_for(base)):
            raise ValueError("Expected both original K0 switch phases as references")
        phases += base[2:]
        labels += labels_for(base[2:])
        sources += [reference_switch_run]*2
        pairs = [(0,2),(7,3),(8,4),(7,5),(8,6),(3,4),(5,6)]
        pairs += [(endpoint,mixed) for endpoint in (0,1) for mixed in (3,4,5,6)]
    elif phases==prefix_high:
        peers=(native[0],native[3],native[2],thin[2],thin[3])
        if any(json.loads((r/"run.json").read_text())["policies"]!=labels_for(s)
               for r,s in ((reference_switch_run,native),(thinning_run,thin))):
            raise ValueError("Incomplete zero-prefix sources")
        peer_runs=[reference_switch_run]*3+[thinning_run]*2
        prefix_peers={labels[i]:(r,label) for i,r,label in
                      zip((0,2,3,4,5),peer_runs,labels_for(peers),strict=True)}
        phases+=peers
        labels+=labels_for(peers)
        sources+=peer_runs
        pairs=[(6,0),(7,2),(8,3),(9,4),(10,5),(1,2)]
    elif phases in (sparse_high,thin):
        ref = json.loads((reference_switch_run/"run.json").read_text())
        if (ref["model"],ref["policies"]) != (model,labels_for(native)):
            raise ValueError("Expected qualified 128/native256 reference phases")
        if phases == thin:
            phases += native[-1:]
            labels += labels_for(native[-1:])
            sources += [reference_switch_run]
            pairs += [(4,0),(4,2),(4,3)]
        else:
            phases += native[1:]
            labels += labels_for(native[1:])
            sources += [reference_switch_run]*3
            pairs += [(4,1),(5,2),(6,3)]
    out.mkdir(exist_ok=True, parents=True)
    cache = out / ".inputs"
    dirs, summary, mechanism = [], [], []
    for label, pattern, source in zip(labels, phases, sources, strict=True):
        folder = snapshot(source, label, sha(cases), cache)
        dirs.append(folder)
        peer=prefix_peers.get(label)
        peer_path=snapshot(peer[0],peer[1],sha(cases),cache) if peer else None
        counts,stage_counts = [],[]
        for i in range(90):
            raw = json.loads((folder / f"{i:02d}-{label}.json").read_text())
            if (raw["index"], raw["policy"], raw["model"]) != (i, label, model):
                raise ValueError("Request identity mismatch")
            counts.append(audit(raw, pattern))
            if peer:
                zero=json.loads((peer_path/f"{i:02d}-{peer[1]}.json").read_text())
                counts[-1].update(prefix_ablation(zero,raw))
            if pattern in thin[2:] and not prefix_peers:
                ref_label=labels_for(native[-1:])[0]
                ref=json.loads((reference_switch_run/f"{i:02d}-{ref_label}.json").read_text())
                first=first_changed_offer(ref,raw)
                counts[-1].update(changed_requests=int(first is not None))
            if model=="gemma":
                stage_counts.append(gemma_stages(raw))
        for cohort, idx in (("original30", slice(30)), ("added60", slice(30,None)), ("all90", slice(None))):
            sums = Counter()
            for r in counts[idx]:
                sums.update(r)
            summary.append(dict(model=model, policy=label, cohort=cohort, **sums))
            if stage_counts:
                total=Counter()
                for part in stage_counts[idx]:
                    total.update(part)
                mechanism.append(dict(model=model,policy=label,cohort=cohort,**total))
    for ctl, arm in pairs:
        path = out/f"{model}-{labels[ctl]}-vs-{labels[arm]}.csv"
        score_files(dirs[ctl], dirs[arm], cases, path, baseline_policy=labels[ctl], arm_policy=labels[arm])
    with (out/f"{model}-slot-work.csv").open("w", newline="") as f:
        fields = dict.fromkeys(k for r in summary for k in r)
        w = csv.DictWriter(f, fieldnames=list(fields), lineterminator="\n",restval=0)
        w.writeheader()
        w.writerows(summary)
    if mechanism:
        with (out/f"{model}-slot-phases.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(mechanism[0]),lineterminator="\n")
            w.writeheader()
            w.writerows(mechanism)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run","cases","out"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--reference-switch-run",type=Path)
    parser.add_argument("--thinning-run",type=Path)
    report(**vars(parser.parse_args()))
