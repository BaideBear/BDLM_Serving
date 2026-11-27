"""Report the three fixed low-budget arms only on identical, sealed CPU requests."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from statistics import fmean
import sys

from .llada_all_slots90 import DATA, read, sha, require
from .fixed_cold_exposure import exposure
from .llada_low_start_pilot import INDICES, ARMS


def phase_counts(row, *, optimus=False):
    """Calls have exclusive purposes; row counts describe overlapping work within calls."""
    c=Counter(dict(build_F=0,prompt_tail_F=0,mask_sample_F=0,edit_only_F=0,refresh_only_F=0,
                   prompt_Q=0,first_mask_Q=0,dirty_Q=0,clean_mask_Q=0,clean_decoded_Q=0,
                   mask_proposals=0,m2t=0))
    decoded=set()
    if optimus:
        from .ibs_prefix_pair import audit
        ref=audit(row,0)
        origin=len(row["prompt_ids"])
        tape=[e for e in row["events"] if e["phase"]!="prefill"]
    else:
        ref=row
        origin=row["prefill_Q"]
        tape=row["trace"]
    seen=set(range(row["prefill_rows"] if optimus else origin))
    for e in tape:
        if optimus:
            rows=e["absolute_positions"]
            sample=[32*e["block"]+p for p in e["selected_local"]]
            require(sample==rows,"Only ordinary, zero-prefix IBS is comparable")
            masks={p for p,is_mask in zip(sample,e["selected_masked"],strict=True) if is_mask}
            proposals=masks
            editable={p for p in sample if p>=origin and p not in masks}
            writes=set(e["written_absolute"])
            admissions=writes&masks
        else:
            rows=e["rows"]
            masks={p for p in rows if p>=origin and p not in decoded}
            proposals=masks.intersection(e["sampled"])
            editable=set(e["sampled"])-proposals
            admissions=set(e["admitted"])
            require(admissions<=proposals and editable<=decoded,"Invalid mask sample or edit")
            decoded.update(admissions)
            decoded.difference_update(e["revoked"])
        require(rows and len(rows)<=8 and len(rows)==len(set(rows)),"Invalid Q8 tape")
        dirty=set(e["dirty_before"])&set(rows)
        category=("build_F" if e["phase"]=="build" else "prompt_tail_F" if max(rows)<origin
                  else "mask_sample_F" if proposals else "edit_only_F" if editable else "refresh_only_F")
        c[category]+=1
        for p in rows:
            key=("prompt_Q" if p<origin else "dirty_Q" if p in dirty else
                 "first_mask_Q" if p not in seen else "clean_mask_Q" if p in masks else "clean_decoded_Q")
            c[key]+=1
        c["mask_proposals"]+=len(proposals)
        c["m2t"]+=len(admissions)
        seen.update(rows)
    require(sum(v for k,v in c.items() if k.endswith("_F"))==ref["F"] and
            sum(v for k,v in c.items() if k.endswith("_Q"))==ref["Q"],"Non-exhaustive phase ledger")
    return c


def work(case=None):
    # Reuse completion/identity checks. Each Q8 arm has to be available on the exact same inputs.
    verified=table(case)
    require(sum(r["q"]==8 for r in verified)==3,"Q8 three-arm comparison is incomplete")
    indices=INDICES if case is None else (case,)
    low=DATA/"llada21-low-start-cpu3-v1"
    strict=DATA/"llada21-strict-q8-m2t-prefix-cpu3-v1"
    basemeta=read(strict/"run.json")
    require(basemeta["samples_sha256"]==sha(DATA/"samples.json") and basemeta["cap"]==2048
            and read(strict/"complete.json")["run_sha256"]==sha(strict/"run.json"),"Unsealed baseline")
    groups={}
    for i in indices:
        ref=read(strict/f"{i:02}-strict_ibs_q8_m90_prefix0.json")
        dig=hashlib.sha256(json.dumps(ref["prompt_ids"],separators=(",",":")).encode()).hexdigest()
        key=tuple(ref[k] for k in ("seed","task_id","dataset"))
        for a in ("strict_ibs","eager","resident","credit"):
            x=ref if a=="strict_ibs" else read(low/f"{i:02}-q8-{a}.json")
            require(tuple(x[k] for k in ("seed","task_id","dataset"))==key and
                    (a=="strict_ibs" or x["input_sha256"]==dig),"Unpaired input")
            counts=phase_counts(x,optimus=a=="strict_ibs")
            n,f,q=(ref[k] for k in ("output_tokens","generation_forwards","generation_query_rows")) if a=="strict_ibs" else (
                x[k] for k in ("N","F","Q"))
            groups.setdefault(a,[]).append((counts,n,f,q,bool(x["score"]["correct"])))
    records=[]
    for a,group in groups.items():
        totals={k:sum(c[k] for c,*_ in group) for k in group[0][0]}
        n,f,q=(sum(x[j] for x in group) for j in (1,2,3))
        records.append(dict(indices=",".join(map(str,indices)),q=8,arm=a,correct=sum(x[4] for x in group),
                            requests=len(group),macro_TPF=fmean(x[1]/x[2] for x in group),efficiency=n/q,
                            N=n,F=f,paid_Q=q,**totals))
    return records


def fixed_mask_rank():
    """Hold IBS's actual seat count and token state fixed; compare clean MASK order only."""
    from .ibs_prefix_pair import audit
    strict=DATA/"llada21-strict-q8-m2t-prefix-cpu3-v1"
    require(read(strict/"complete.json")["run_sha256"]==sha(strict/"run.json"),"Unsealed control")
    output=[]
    for i in INDICES:
        x=read(strict/f"{i:02}-strict_ibs_q8_m90_prefix0.json")
        audit(x,0)
        origin=len(x["prompt_ids"])
        decoded=set(range(origin))
        sums=Counter()
        for e in x["events"]:
            if e["phase"]=="prefill":
                continue
            rows=e["absolute_positions"]
            start=32*e["block"]
            dirty=set(e["dirty_before"])
            actual=[p for p in rows if p not in decoded and p not in dirty]
            require([p not in decoded for p in rows]==e["selected_masked"],"Wrong pre-forward token state")
            pool=[p for p in range(start,start+32) if p not in dirty and p not in decoded]
            shadow=pool[:len(actual)]
            if actual:
                sums["mask_calls"]+=1
                sums["same_rank_calls"]+=actual==shadow
                sums["different_rows"]+=len(set(actual)-set(shadow))
                sums["selected_clean_masks"]+=len(actual)
            w=set(e["written_absolute"])
            edits=set(e["edited_positions"])
            require((w|edits)<=set(rows),"Write outside IBS rows")
            decoded.update(w)
            decoded.difference_update(edits-w)  # a sampled edit back to MASK
        output.append(dict(index=i,q=8,mask_calls=sums["mask_calls"],same_rank_calls=sums["same_rank_calls"],
                           different_rows=sums["different_rows"],selected_clean_masks=sums["selected_clean_masks"]))
    return output


def table(case=None):
    root = DATA/"llada21-low-start-cpu3-v1"
    register = root/"run.json"
    meta = read(register)
    require(meta["samples"] == sha(DATA/"samples.json") and meta["indices"] == list(INDICES)
            and meta["budgets"] == [4,8] and meta["arms"] == list(ARMS), "Invalid protocol")
    seal = None if case is not None else read(root/"complete.json")
    if seal is not None:
        require(seal["registration"] == sha(register) and len(seal["files"]) == 18, "Incomplete run")
    records=[]
    for q in (4,8):
        idxs = INDICES if case is None else (case,)
        paths={a:[root/f"{i:02}-q{q}-{a}.json" for i in idxs] for a in ARMS}
        if case is not None and not all(p.is_file() for group in paths.values() for p in group):
            continue
        pairs={}
        for arm in ("eager", "resident", "credit"):
            group=[]
            for i, p in zip(idxs, paths[arm], strict=True):
                if seal is not None:
                    require(seal["files"].get(p.name) == sha(p), "Changed sealed row")
                row=read(p)
                require(row["registration_sha256"] == sha(register) and row["index"] == i and
                        row["arm"] == arm and row["q"] == q and
                        row["exposure"] == exposure(row), "Wrong row or exposure")
                calls=row["trace"]
                require(len(calls)==row["F"] and sum(c["q"] for c in calls)==row["Q"] and
                        all(c["layer_physical_q"]==[c["q"]]*20 and c["q"]<=q for c in calls),
                        "Invalid F/Q")
                key=tuple(row[k] for k in ("input_sha256","seed","dataset","task_id"))
                require(pairs.setdefault(i,key)==key,"Unpaired input")
                group.append(row)
            n,f,paid=(sum(r[k] for r in group) for k in ("N","F","Q"))
            visits=lambda k:sum(r["row_visits"].get(k,0) for r in group)
            records.append(dict(indices=",".join(map(str,idxs)),q=q,arm=arm,correct=sum(
                r["correct"] for r in group),requests=len(group),macro_TPF=fmean(r["N"]/r["F"] for r in group),
                efficiency=n/paid,N=n,F=f,paid_Q=paid,build_F=sum(r["phase_F"].get("build",0) for r in group),
                build_Q=sum(c["q"] for r in group for c in r["trace"] if c["phase"]=="build"),
                prefix_Q=visits("prefix"),prefix_edits=visits("prefix_edits"),
                first_before_full=sum(r["exposure"]["first_before_full"] for r in group),
                failures=sum(bool(r["failure"]) for r in group)))
    require(records,"No complete matched comparison for this item")
    return records


def gate_table():
    from .llada_low_start_pilot import compare_gate
    root=DATA/"llada21-low-start-gate-cpu3-v1"
    base=DATA/"llada21-low-start-cpu3-v1"
    table()  # Verify all 18 original records and their fixed inputs before pairing.
    reg=read(root/"run.json")
    seal=read(root/"complete.json")
    require(reg["eager_seal"]==sha(base/"complete.json") and reg["samples"]==sha(DATA/"samples.json")
            and reg["arms"]==["confident"] and reg["indices"]==list(INDICES) and reg["budgets"]==[4,8]
            and seal["registration"]==sha(root/"run.json") and len(seal["files"])==6,
            "Missing six registered interventions")
    result=[]
    for q in reg["budgets"]:
        for i in INDICES:
            p=root/f"{i:02}-q{q}-confident.json"
            require(sha(p)==seal["files"][p.name],"Changed treated request")
            x, ref=read(p),read(base/f"{i:02}-q{q}-eager.json")
            require(x["arm"]=="confident" and x["registration_sha256"]==seal["registration"]
                    and (x["q"],x["index"])==(q,i) and x["correct"]==x["score"]["correct"]
                    and all(x[k]==ref[k] for k in ("seed","input_sha256","task_id","dataset","law"))
                    and compare_gate(x,ref)==x["treatment"],"Mismatched treatment or prompt")
            widths=[e["q"] for e in x["trace"]]
            require(len(widths)==x["F"] and sum(widths)==x["Q"] and
                    all(e["offer"]==q and 0<e["q"]==len(set(e["rows"]))<=q
                        and e["layer_physical_q"]==[e["q"]]*20 for e in x["trace"]),
                    "Unaudited treated Q")
            result.append(dict(index=i,q=q,eager_correct=int(ref["correct"]),correct=int(x["correct"]),
                        eager_N=ref["N"],eager_F=ref["F"],eager_Q=ref["Q"],eager_TPF=ref["N"]/ref["F"],
                        N=x["N"],F=x["F"],paid_Q=x["Q"],TPF=x["N"]/x["F"],efficiency=x["N"]/x["Q"],
                        **x["treatment"]))
    return result


def expanded_table():
    from . import llada_fixed_family_eval as ev
    from .llada_low_start_pilot import compare_gate
    from bdlm_serving.evaluation.frozen90 import load_tasks
    root=DATA/"llada21-low-start-gate-added60-v1"
    reg,seal=read(root/"run.json"),read(root/"complete.json")
    tasks=load_tasks(DATA/"samples.json")
    indices=[30+20*d+t for t in range(20) for d in range(3)]
    require(reg["indices"]==indices and reg["budgets"]==[4,8] and
            reg["arms"]==["eager","confident"] and reg["law"]==[1,8] and
            reg["device"]=="cpu" and reg["scoring"]==ev.SCORER_SHA and
            reg["samples"]==sha(DATA/"samples.json")==ev.reg.SAMPLES_SHA and
            seal["registration"]==sha(root/"run.json") and len(seal["files"])==240,
            "Unsealed or unregistered expanded cohort")
    out=[]
    for q in reg["budgets"]:
        groups={}
        for i in indices:
            pair=[]
            for arm in reg["arms"]:
                p=root/f"{i:02}-q{q}-{arm}.json"
                require(seal["files"].get(p.name)==sha(p),"Changed output")
                x=read(p)
                require(x["registration_sha256"]==seal["registration"] and x["q"]==q and x["law"]==[1,8]
                        and x["arm"]==arm and all(x[k]==tasks[i][k] for k in ev.reg.KEYS)
                        and x["correct"] is bool(x["score"]["correct"]) and len(x["trace"])==x["F"]
                        and sum(e["q"] for e in x["trace"])==x["Q"] and all(e["offer"]==q and
                        0<e["q"]==len(set(e["rows"]))<=q and e["layer_physical_q"]==[e["q"]]*20 for e in x["trace"]),
                        "Invalid sample, score or physical bill")
                pair.append(x)
            a,b=pair
            require(a["input_sha256"]==b["input_sha256"] and b["treatment"]==compare_gate(b,a),
                    "Unpaired or incorrect treatment")
            for group in ("added60",a["dataset"]):
                # Never retain the large traces across requests.
                groups.setdefault(group,[]).append(tuple(tuple(x[k] for k in ("correct","N","F","Q","failure"))
                                                            for x in pair))
        for group,g in groups.items():
            require(len(g)==(60 if group=="added60" else 20),"Unbalanced dataset")
            outrow=dict(q=q,cohort=group,requests=len(g),
                        rescued=sum(not a[0] and b[0] for a,b in g),lost=sum(a[0] and not b[0] for a,b in g))
            for j,arm in enumerate(reg["arms"]):
                side=[p[j] for p in g]
                n,f,paid=(sum(x[k] for x in side) for k in (1,2,3))
                outrow.update({arm+"_"+k:v for k,v in dict(correct=sum(x[0] for x in side),N=n,F=f,paid_Q=paid,
                    macro_TPF=fmean(x[1]/x[2] for x in side),efficiency=n/paid,
                    failures=sum(bool(x[4]) for x in side)).items()})
            out.append(outrow)
    return out


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case",type=int,choices=INDICES)
    ap.add_argument("--work",action="store_true",help="Attribute Q8 work including same-CPU strict Optimus IBS")
    ap.add_argument("--gate",action="store_true",help="Audit the six sealed confidence-gated requests")
    ap.add_argument("--added60",action="store_true",help="Only report the completed 240-request added60 experiment")
    ap.add_argument("--frontier",action="store_true",help="Check clean MASK rank on the same strict IBS states")
    args=ap.parse_args()
    if args.frontier and (args.added60 or args.gate or args.work or args.case is not None):
        ap.error("--frontier is a separate same-state diagnostic")
    if (args.gate or args.added60) and (args.work or args.case is not None or args.gate and args.added60):
        ap.error("Select one complete paired cohort")
    rows=(fixed_mask_rank() if args.frontier else expanded_table() if args.added60 else gate_table()
          if args.gate else (work if args.work else table)(args.case))
    out=csv.DictWriter(sys.stdout,fieldnames=rows[0],lineterminator="\n")
    out.writeheader()
    out.writerows(rows)
