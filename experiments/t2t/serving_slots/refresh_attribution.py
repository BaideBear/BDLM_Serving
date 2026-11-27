"""Count paid query-row visits and same-step edits from sealed single-request traces."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from experiments.t2t.serving_slots.llada_all_slots90 import DATA, SAMPLES_SHA, require, sha


def read(path):
    return json.loads(path.read_bytes())


def full90(data, folder, arm):
    group = data/folder
    manifest = read(data/"samples.json")
    for i, task in enumerate(manifest if isinstance(manifest,list) else manifest["samples"]):
        r = read(group/f"{i:02d}-{arm}.json")
        require(r["index"] == i and r["policy"] == arm and r["task_id"] == task["task_id"]
                and r["seed"] == task["seed"], "Wrong request or arm")
        yield r


def tally(data, model, q, k, arm, family, group, missing=False):
    c = Counter()
    width = 0
    for r in full90(data, group, arm):
        if family == "unified":
            ev = r["trace"]
        else:
            ev = [e for e in r["events"] if e.get("phase") != "prefill"]
        paid = r["generation_physical_padded_rows"]
        decoder = 0
        for j,e in enumerate(ev):
            other, other_edits = 0, 0
            if family == "unified":
                w,m,p,et = (e[key] for key in ("q","mandatory_q","prefix","strict_edits"))
                mt,pt = 0,0
                require(p == 0, "Compact record cannot attribute prefix edits")
            elif family == "capped-ibs":
                physical = set(e["absolute_positions"])
                dirty = set(e["dirty_before"]) & physical
                edit = set(e["edited_positions"])
                require(edit <= physical and e["q_tokens"] == len(physical) <= q,"Bad projection")
                w,m,p,et,mt,pt = len(physical),len(dirty),0,len(edit),len(dirty & edit),0
            else:
                w = e["physical_q"] if model == "llada21" else e["q"]
                if family == "progress80":
                    m = e.get("mandatory_q",0)
                    require(m == len(set(e.get("mandatory_fresh_positions",())) &
                                     set(e["absolute_positions"])), "Unpaid mandatory row")
                else:
                    m = e["mandatory_fresh"] if model == "llada21" else e["fresh_selected"]
                pt, p = map(set,(e["reserved_strict_edit_positions"], e["reserved_extra_edit_positions"]))
                require(pt <= p and p <= set(e.get("absolute_positions",e.get("positions",[]))),
                        "Unprojected prefix edit")
                plan=e.get("prefix_plan")
                if plan:
                    off = e["block"]*32 if model == "llada21" else 0
                    require(p == {off+x for x in plan["added_rows"]},"Non-prefix seat")
                pt,p = len(pt),len(p)
                if family == "progress80":
                    # This runner uses hard_frontier, NOT a physical-prefix quota.
                    require(not plan, "Unexpected optional prefix plan for Q64")
                    other, other_edits, p, pt = p, pt, 0, 0
                et,mt = e["hard_strict_edits"],e.get("edited_in_fresh",0)
            require(0<=pt<=p and 0<=mt<=m and 0<=other_edits<=other
                    and m+p+other<=w and pt+mt+other_edits<=et,"Invalid row counts")
            c.update(required=m, prefix=p, required_t2t=mt, prefix_t2t=pt, total_t2t=et,
                     other_reserved=other,other_reserved_t2t=other_edits)
            if e.get("stage") == "selected" and family in ("slack-right","fixed-zero"):
                masks = e["selected_masked_before"] if model=="llada21" else e["selected_mask"]
                require(w>=m+p+masks, "Invalid backfill size")
                c.update(selected_Q=w,selected_required=m,selected_mask=masks,
                         selected_old_backfill=w-m-p-masks)
                if model=="llada21":
                    fresh = set(plan["fresh_rows"])
                    require(len(fresh)==m, "Bad selected refresh count")
                    unassigned, backfill = et-pt, w-m-p-masks
                    lo, hi = max(0,unassigned-backfill), min(m,unassigned)
                    require(lo<=hi, "Inconsistent T2T counts")
                    nxt = ev[j+1] if j+1<len(ev) else {}
                    if nxt.get("block")==e["block"] and nxt.get("stage")=="selected":
                        # Writes to an already-resolved fresh row must be T2T.
                        hits = len(fresh & set(nxt["prefix_plan"]["fresh_rows"]))
                        require(lo<=hits<=hi, "Inconsistent consecutive write sets")
                        lo=hi=hits
                    c.update(selected_required_t2t_min=lo,selected_required_t2t_max=hi)
            decoder += w
            width = max(width,w)
        encoder = (r["encoder_q"] if family=="unified" else
                   sum(w for phase,w in r["physical_events"][1:] if phase=="encoder")
                   if model=="gemma" else 0)
        require(decoder+encoder == paid,"Paid Q mismatch")
        c.update(paid_Q=paid,decoder_Q=decoder,encoder_Q=encoder)
    row = dict(model=model,offer=q,K=k,arm=arm,protocol=family,n=90,peak_decoder_Q=width,**c)
    if missing:
        row["required_t2t"]=None
    return row


def llada_strict_pilot(data,k,threshold):
    c = Counter()
    arm = f"strict_ibs_q8_m{threshold}_prefix{k}"
    folder = data/"llada21-strict-q8-m2t-prefix-cpu3-v1"
    for i in (0,10,20):
        r = read(folder/f"{i:02d}-{arm}.json")
        require(r["index"]==i and r["policy"]==arm, "Wrong strict Q8 pilot arm")
        visits = 0
        for e in r["events"]:
            if e["phase"]=="prefill":
                continue
            rows = set(e["absolute_positions"])
            req = rows & set(e["dirty_before"])
            pref = set(e.get("prefix_refresh_absolute", ()))
            edit = set(e["edited_positions"])
            require(len(rows)==e["q_tokens"]<=8 and pref<=rows and edit<=rows and
                    not pref & (req | edit), "Invalid KV-only trace")
            c.update(required=len(req),prefix=len(pref),required_t2t=len(req&edit),
                     prefix_t2t=0,total_t2t=len(edit))
            visits += len(rows)
        require(visits==r["generation_physical_padded_rows"],"Bad Q8 pilot denominator")
        c.update(paid_Q=visits,decoder_Q=visits,encoder_Q=0)
    return dict(model="llada21",offer=8,K=k,arm=arm,protocol="strict-cpu3-kvonly",
                n=3,peak_decoder_Q=8,**c)


def strict_pilot(data,k):
    c = Counter()
    for i in (0,10,20):
        g = data/"gemma-strict-eval-v1"
        stem = f"cpu-q128-k{k}-{i:02d}"
        r = read(g/"compact"/(stem+".json"))
        trace = read(g/"raw"/stem/"evidence.json")["trace"]
        require(r["run_sha256"] == sha(g/"raw"/stem/"run.json") and
                sum(e["q"] for e in trace) == r["decoder_Q"] and r["decoder_q_max"]==128,
                "Wrong pilot tape")
        for e in trace:
            req, pref, edit = (set(e[key]) for key in ("dirty_before","prefix","edited"))
            req &= set(e["rows"])
            require(not req & pref and req | pref <= set(e["rows"]), "Overlapping refresh types")
            c.update(required=len(req),prefix=len(pref),required_t2t=len(req&edit),
                     prefix_t2t=len(pref&edit),total_t2t=len(edit))
        c.update(paid_Q=r["Q"],decoder_Q=r["decoder_Q"],encoder_Q=r["encoder_Q"])
    return dict(model="gemma",offer=128,K=k,arm=f"strict-right-k{k}",protocol="strict-cpu3",
                n=3,peak_decoder_Q=128,**c)


def report(data):
    require(sha(data/"samples.json")==SAMPLES_SHA,"Wrong frozen suite")
    rows = []
    for q,k in ((4,2),(8,4),(16,4),(32,8)):
        name = {4:"llada21-4-matched",8:"llada-q8",16:"llada21-16-matched",32:"llada21-32-matched"}[q]
        rows.append(tally(data,"llada21",q,0,f"t2t_q{q}_f0_m{q}_hard_prefix_zero","fixed-zero",name,True))
        arm=f"t2t_q{q}_f0_m{q-k}_hard_prefix_slack_right"
        rows.append(tally(data,"llada21",q,k,arm,"slack-right",f"llada21-registered-q{q}-k{k}",True))
        cap = "low" if q==4 else "anchor" if q==32 else "endpoints"
        rows.append(tally(data,"llada21",q,0,f"optcap_ibs_{q}","capped-ibs","llada21-optimus-hardcap-"+cap))
    rows.append(tally(data,"llada21",64,0,"progress80-causal","progress80",
                      "llada21-q64-progress80-mask-gpu90-v1",True))
    for q in (32,64,128,256):
        k=q//4
        ugroup = f"gemma-unified90-q{q}" + ("-q128_slack_random" if q==128 else "")
        rows.append(tally(data,"gemma",q,0,f"q{q}","unified",ugroup,True))
        rows.append(tally(data,"gemma",q,k,f"t2t_q{q}_f0_m{k}_hard_prefix_slack_right",
                          "slack-right",f"gemma-ratio-q{q}-part2"))
    rows.append(tally(data,"gemma",512,0,"q512","unified","gemma-unified90-q512-q512_open",True))
    rows.extend(strict_pilot(data,k) for k in (0,32))
    rows.extend(llada_strict_pilot(data,k,t) for t in (90,50) for k in (0,2))
    return rows


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--data",type=Path,default=DATA)
    a=p.parse_args()
    rows=report(a.data)
    with a.output.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(key for r in rows for key in r)),
                              lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
