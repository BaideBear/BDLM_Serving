"""Compare Gemma with real external runs: a common candidate or matched offered slots."""
import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path

from bdlm_serving.evaluation.frozen import sha
from bdlm_serving.runtime.optimus_limits import validate_trace
from experiments.t2t.prefix_window.score_run import link_requests, snapshot
from experiments.t2t.serving_slots.switch_report import audit
from tools.prefix_cohorts import load_cases, load_request_set, select_window_cases, _window_row


def read(path):
    return json.loads(path.read_text())


def measured_decoder(row, policy):
    """One request: physical projections and token admissions are separate ledgers."""
    cap = "_cap_" in policy
    if cap:
        validate_trace(row)
    events = row["events"]
    if events[0][0] != "encoder" or any(tag not in ("encoder", "decoder")
                                         for tag, _ in events[1:]):
        raise ValueError("Invalid Gemma stage record")
    observed = [q for tag, q in events[1:] if tag == "decoder"]
    reencode = [q for tag, q in events[1:] if tag == "encoder"]
    if (not observed or any(q != 256 for q in reencode)
            or observed != [s["q"] for s in row["trace"]]
            or sum(observed)+sum(reencode) != row["generation_physical_padded_rows"]
            or len(events)-1 != row["generation_forwards"]):
        raise ValueError("Observed decoder rows differ from reported work")
    work = []
    for s in row["trace"]:
        q, admitted = s["q"], s["accepted"]
        if not 0 <= admitted <= q:
            raise ValueError("Admissions outside projected rows")
        offer = s["offer"] if cap else int(policy.rsplit("_", 1)[1])
        proposed = s["proposed_q"] if cap else q
        work.append((q, admitted, max(0, q-offer), max(0, admitted-offer), proposed-q))
    return work, sum(reencode)


HIGH_FIRST = (256,256,128,128)


def check_first_cut(original, limited, slot):
    """Until the first binding forward the two fixed policies are identical."""
    old, new = original["trace"], limited["trace"]
    first = next((i for i,s in enumerate(old) if s["q"]>slot),None)
    clipped = next((i for i,s in enumerate(new) if s["proposed_q"]>s["offer"]),None)
    if first!=clipped:
        raise ValueError("Spurious or missed first binding forward")
    until = len(old) if first is None else first
    if len(new)<until or any({k:n[k] for k in o}!=o for o,n in zip(old[:until],new[:until])):
        raise ValueError("Diverged before the first cap")
    if first is not None:
        if new[first]["proposed_q"]!=old[first]["q"]:
            raise ValueError("Changed candidate size at first cap")
    elif (len(new)!=len(old) or any(original[k]!=limited[k] for k in
            ("output_ids","events","generation_forwards","generation_physical_padded_rows"))):
        raise ValueError("Nonbinding fixed control did not reproduce the original")


def offered_plan(policy):
    if policy=="flow_w1":
        return (256,)
    if policy=="flow_w2":
        return None  # No qualified own 512 schedule; retain W2 in the candidate comparison.
    mode, *slots = policy.split("_")
    if mode not in ("ibs","obs"):
        raise ValueError(f"Unknown external policy {policy}")
    if slots and slots[0]=="cap":
        slots = slots[1:]
    plan = tuple(map(int,slots))
    if plan not in ((128,), (256,), HIGH_FIRST, tuple(reversed(HIGH_FIRST))):
        raise ValueError(f"No corresponding own schedule for {policy}")
    return plan


def nearest_plan(target, totals):
    """Use all-90 physical work only. Resolve ties consistently in input order."""
    if target <= 0 or not totals or min(totals.values()) <= 0:
        raise ValueError("Expected positive complete-run work")
    return min(totals, key=lambda plan: abs(totals[plan]-target))


def compare(data, out, capped=None, matched=False, cap_policy=None, nearest=False,
            candidate_run=None, candidate=None):
    if nearest and (matched or not capped or cap_policy):
        raise ValueError("Nearest-Q needs all capped controls and its own matching mode")
    if bool(candidate_run)!=bool(candidate) or candidate and (matched or nearest or not capped or cap_policy):
        raise ValueError("A single measured candidate needs its run and all external controls")
    samples = data/"samples.json"
    cases = load_cases(samples)
    digest = sha(samples)
    cache = out/".inputs"
    @lru_cache(maxsize=None)
    def by_index(folder, p):
        rows, model, policy, _ = load_request_set(folder,cases,"Gemma",p)
        if (model,policy) != ("gemma",p):
            raise ValueError("Wrong model or arm")
        return {r["index"]:r for r in rows}
    @lru_cache(maxsize=4)
    def own_for(plan):
        sequence = tuple("256n" if q==256 else q for q in plan)
        label = "slottrace_"+"_".join(map(str,sequence))
        directory = snapshot(data/"gemma-slots-native256-128",label,digest,cache)
        own = by_index(directory,label)
        low = {}
        for i in own:
            row=read(directory/f"{i:02d}-{label}.json")
            audit(row,sequence)
            low[i]=[e["q"] for e in row["events"] if e.get("slot_offer")==128]
        return label,own,low
    if candidate:
        if candidate.startswith("gate_"):
            from .gated import PLANS, original, validate_gate
            from .switch_report import first_changed_offer
            _,gate_mode,phase=candidate.split("_")
            meta=read(candidate_run/"run.json")
            if (phase not in PLANS or gate_mode not in ("all","early","tail") or
                    candidate not in meta["policies"] or meta["samples_sha256"]!=digest or
                    meta["model"]!="gemma" or meta["threshold"]!=64 or meta["suite"]!="frozen90" or
                    tuple(meta["offers"][phase])!=PLANS[phase] or
                    any(read(candidate_run/f"complete-{p}.json")!=dict(policy=p,requests=90)
                        for p in dict.fromkeys((candidate,"gate_all_high","gate_all_low")))):
                raise ValueError("Incomplete or unqualified mask gate")
            path=link_requests(candidate_run,candidate,cache)
            refname=original(phase)
            refdir=snapshot(data/"gemma-slots-native256-128",refname,digest,cache)
        else:
            if candidate not in ("slottrace_256n_128_128_128","slottrace_128_256n_128_128"):
                raise ValueError("Unregistered position ablation")
            path=snapshot(candidate_run,candidate,digest,cache)
            pattern=candidate.removeprefix("slottrace_").split("_")
        own=by_index(path,candidate)
        low={}
        for i in own:
            row=read(path/f"{i:02d}-{candidate}.json")
            if candidate.startswith("gate_"):
                validate_gate(row,row["gate_decisions"],phase,gate_mode)
                if first_changed_offer(read(refdir/f"{i:02d}-{refname}.json"),row)!=row["first_changed_event"]:
                    raise ValueError("Changed first intervention")
            else:
                audit(row,pattern)
            low[i]=[e["q"] for e in row["events"] if e.get("slot_offer")==128]
        measured_candidate=candidate,own,low
    dirs = [(data/"gemma-optimus-reference90",["ibs_128","obs_128"]),
            (data/"gemma-optimus-primary90",["ibs_256","obs_256"]),
            (data/"gemma-wave-reference90",["flow_w1","flow_w2"])]
    if cap_policy and not capped:
        raise ValueError("A named cap policy requires its registered run")
    if capped:
        meta = read(capped/"run.json")
        gate = read(capped/"qualified-logits.json")
        if (not meta["hard_q_cap"] or meta["cap_clock"]!="all_decoder_forwards_request_local"
                or meta["cap_excludes"]!="encoder" or meta["actual_token_edits"]!="M2T only"
                or gate!=dict(modes=["ibs","obs"],chunk=256,first_step_logits="bit_identical",index=0)):
            raise ValueError("Wrong capped-mode registration")
        if cap_policy and cap_policy not in meta["policies"]:
            raise ValueError("Unknown cap policy in this run")
        dirs.append((capped,[cap_policy] if cap_policy else meta["policies"]))
    plans = ((128,), (256,), HIGH_FIRST, tuple(reversed(HIGH_FIRST)))
    totals = {plan:sum(r["physical_Q"] for r in own_for(plan)[1].values())
              for plan in plans} if nearest else {}
    lines, fixed_effects = [], []
    for folder,policies in dirs:
        meta = read(folder/"run.json")
        if meta["samples_sha256"]!=digest or meta["suite"]!="frozen90":
            raise ValueError(f"Wrong suite for {folder}")
        for p in policies:
            if (nearest or candidate) and folder!=capped and not p.startswith("flow_"):
                continue
            if p not in meta["policies"] or read(folder/f"complete-{p}.json")!=dict(policy=p,requests=90):
                raise ValueError(f"Missing or incomplete {p}")
            plan=offered_plan(p) if matched else HIGH_FIRST
            if plan is None:
                continue
            rows = by_index(link_requests(folder,p,cache),p)
            if nearest:
                target = sum(r["physical_Q"] for r in rows.values())
                plan = nearest_plan(target, totals)
            mixed,own,low=measured_candidate if candidate else own_for(plan)
            is_optimus = p.startswith(("ibs_","obs_"))
            if is_optimus:
                caps = "_cap_" in p
                ledgers = {}
                slots = offered_plan(p)
                original = None
                if caps and len(slots)==1:
                    orig_name=p.replace("_cap_", "_")
                    orig_run=next(path for path,labels in dirs if path!=capped and orig_name in labels)
                    original=by_index(link_requests(orig_run,orig_name,cache),orig_name)
                for i in rows:
                    req=read(folder/f"{i:02d}-{p}.json")
                    ledgers[i]=measured_decoder(req,p)
                    if original is not None:
                        check_first_cut(read(orig_run/f"{i:02d}-{orig_name}.json"),req,slots[0])
            for w in (30,60,90):
                subset=select_window_cases(cases,w)
                s=_window_row(subset,rows,own,w,"all","gemma",p,mixed,None)
                if nearest:
                    s.update(match_on="all90_absolute_physical_Q_gap",
                             all90_relative_Q_gap=(totals[plan]-target)/target,
                             slice_relative_Q_gap=s["arm_physical_Q"]/s["baseline_physical_Q"]-1)
                limits=[q for r in subset for q in low[r["index"]]]
                deltas=[own[r["index"]]["physical_Q"]-rows[r["index"]]["physical_Q"] for r in subset]
                if sum(deltas)!=s["arm_physical_Q"]-s["baseline_physical_Q"]:
                    raise ValueError("Paired physical work does not match the aggregate")
                s.update(mixed_low_rounds=len(limits),mixed_low_over=sum(q>128 for q in limits),
                         mixed_low_max=max(limits,default=""),mixed_low_excess=sum(max(0,q-128) for q in limits),
                         arm_lower_Q_requests=sum(d<0 for d in deltas),arm_equal_Q_requests=deltas.count(0),
                         arm_higher_Q_requests=sum(d>0 for d in deltas))
                if is_optimus:
                    work = [ledgers[r["index"]] for r in subset]
                    qs, edits, over, edits_over, cuts = zip(*(e for steps, _ in work for e in steps))
                    s.update(baseline_decoder_steps=len(qs),baseline_max_decoder_q=max(qs),
                             baseline_decoder_over=sum(x>0 for x in over),baseline_decoder_excess=sum(over),
                             baseline_actual_rows_cut=sum(cuts) if caps else "",
                             baseline_requests_over=sum(any(e[2] for e in steps) for steps, _ in work),
                             baseline_requests_clipped=sum(any(e[4] for e in steps) for steps, _ in work) if caps else "",
                             baseline_decoder_Q=sum(qs),baseline_reencode_Q=sum(q for _, q in work),
                             baseline_max_admitted=max(edits),baseline_total_admitted=sum(edits),
                             baseline_admission_over_steps=sum(x>0 for x in edits_over))
                    for offer in dict.fromkeys(slots):
                        by_offer = [e for steps,_ in work for tick,e in enumerate(steps)
                                    if slots[tick%len(slots)]==offer]
                        s[f"baseline_offer{offer}_rounds"]=len(by_offer)
                        s[f"baseline_offer{offer}_peak_q"]=max((e[0] for e in by_offer),default="")
                        s[f"baseline_offer{offer}_Q"]=sum(e[0] for e in by_offer)
                        s[f"baseline_offer{offer}_rows_clipped"]=sum(e[4] for e in by_offer) if caps else ""
                    if original is not None:
                        row=_window_row(subset,original,rows,w,"all","gemma",orig_name,p,slots[0])
                        row.update(requests_clipped=s["baseline_requests_clipped"],rows_clipped=sum(cuts))
                        fixed_effects.append(row)
                lines.append(s)
    stem = ("gemma-external-"+candidate if candidate else "gemma-nearest-physical-Q" if nearest else
            "gemma-matched-offers-vs-external" if matched else "gemma-mixed-vs-external")
    if cap_policy:
        stem += "-only-"+cap_policy
    write(out/(stem+".csv"),lines)
    if capped and not matched and not cap_policy and not nearest and not candidate:
        if len(fixed_effects)!=12:
            raise ValueError("Missing one or more fixed cap controls")
        write(out/"gemma-original-vs-capped-fixed.csv",fixed_effects)


def write(path, lines):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in lines for k in r)),lineterminator="\n")
        writer.writeheader()
        writer.writerows(lines)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("data","out","capped"):
        parser.add_argument("--"+key,type=Path,required=key!="capped")
    parser.add_argument("--matched",action="store_true",help="Use identical nominal offer order, not equal physical cost")
    parser.add_argument("--nearest",action="store_true",help="Closest of four own plans by all90 physical Q to each capped Optimus and FlowBlock baseline, not equal limits or per-request work")
    parser.add_argument("--cap-policy",help="Include only this fully completed cap arm; distinctly named output")
    parser.add_argument("--candidate-run",type=Path,help="Complete position ablation or gated run")
    parser.add_argument("--candidate",help="One registered position ablation or mask gate policy")
    compare(**vars(parser.parse_args()))
