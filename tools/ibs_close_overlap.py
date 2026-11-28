"""Count actual idle seats at mask-free IBS closure. No forward/quality prediction."""
import argparse
from collections import Counter
from hashlib import sha256
from itertools import groupby
import json
from pathlib import Path

from ibs_frozen_eos_tail import require


def trace(r, model):
    if model == "llada21":
        return [(e["block"], e["absolute_positions"], e["written_absolute"], e["edited_positions"],
                 e["cached_prefix_after"], e["selected_masked"], e["dirty_before"])
                for e in r["events"] if e["phase"] != "prefill"]
    return [(e["block"], e["positions"], e["admitted"], [], e["ready"], None, None)
            for e in r["trace"]]


def count(r, model, q, verify_plan=None):
    b, p = (32,len(r["prompt_ids"])) if model == "llada21" else (256,0)
    cap = r.get("generation_cap", 2048)
    require(cap == 2048, "wrong cap")
    rows = trace(r, model)
    c = Counter(requests=1, F=r["generation_forwards"], decoder_F=len(rows))
    groups = [(g,list(v)) for g,v in groupby(rows, key=lambda e:e[0])]
    require([g for g,_ in groups] == list(range(p//b,p//b+len(groups))), "not in order")
    for j,(g,events) in enumerate(groups):
        lo, end = b*g,b*(g+1)
        active = set(range(lo,end))
        resolved = {i for i in active if i < p}
        resident, pending = set(), set()
        first = None
        next_observed = j+1 < len(groups)
        for t,(_,pos,writes,edits,ready,mask,dirty) in enumerate(events):
            projected, w, e = set(pos),set(writes),set(edits)
            require(0 < len(pos) == len(projected) <= q and projected <= active and
                    (w|e) <= projected and pending <= projected, "invalid selected rows")
            require(dirty is None or set(dirty)==pending, "unaccounted prior debt")
            require(mask is None or list(mask)==[x not in resolved for x in pos], "old mask mismatch")
            require(not (w-e) & resolved and e <= resolved, "unexpected write or edit")
            c["decoder_Q"] += len(pos)
            slack = q-len(pos)
            eligible = (first is None and active == resolved and end < p+cap and
                        active == resident|projected and slack > 0)
            if verify_plan is not None:
                proposal = verify_plan(offer=q,head_start=lo,width=b,generated_end=p+cap,
                                       selected_head=pos,dirty_head=pending,resident_head=resident,
                                       unresolved_head=active-resolved,generated_eos=False,
                                       already_tried=first is not None)
                expected = tuple(range(end,end+min(slack,p+cap-end))) if eligible else ()
                require(proposal == expected, "shared rule differs from independent tape replay")
            if eligible:
                # Historical IBS does not store the token ID of each edit.
                # Record the pre-step geometry; classify EOS separately below.
                first = t
                c["geometry_blocks"] += 1
                c["geometry_seats"] += min(slack, p+cap-end)
                c["first_geometry_per_request"] = 1
                if next_observed:
                    c["later_promoted"] += 1
                    c["head_edited_before_promotion"] += any(ev[3] for ev in events[t:])
                    if not any(ev[3] for ev in events[t:]):
                        c["promoted_no_later_edit"] += 1
                        c["promoted_no_later_edit_request"] = 1
                        c["promoted_no_later_edit_seats"] += min(slack,p+cap-end)
                        c["immediate_clean_promotion"] += t+1 == len(events)
                        nxt = groups[j+1][1][0][1]
                        ext = set(range(end,end+min(slack,p+cap-end)))
                        c["positions_also_used_by_next_first_call"] += len(ext.intersection(nxt))
                        c["entire_next_first_call_covered"] += set(nxt) <= ext
            resolved.difference_update(e-w)
            resolved.update(w)
            pending = w
            resident.update(projected)
            require(set(range(lo,lo+ready)) <= (resolved & resident)-pending, "invalid seal")
        if next_observed:
            require(not pending and resolved == resident == active and events[-1][4] == b,
                    "promoted incomplete head")
    require(c["decoder_F"] <= c["F"], "generation bill")
    return c


def audit(root, model, q, verify_plan=None):
    if model=="llada21":
        source = root/("llada21-optimus-hardcap-"+{4:"low",8:"endpoints",16:"endpoints",32:"anchor"}[q])
        name=f"optcap_ibs_{q}"
    else:
        source=(root/"gemma-fixed-caps-q32q64-v1" if q in (32,64) else
                root.parent/"private-optimus-cap90-v1/gemma-optimus-cap90-v1")
        name=f"ibs_cap_{q}"
    control=(root/"fixed-family-reference-case-ledger-v1.json").read_bytes()
    require(sha256(control).hexdigest()=="ff9caccbf776416f0237ddcc73af774fbe567c57929b359583f70fba128d64ce",
            "wrong ledger")
    refs=json.loads(control)["arms"][model+":"+name]["items"]
    totals={k:Counter() for k in ("all90","original30","new60","pilot6")}
    digest=sha256()
    for i in range(90):
        raw=(source/f"{i:02}-{name}.json").read_bytes()
        digest.update(sha256(raw).digest())
        r=json.loads(raw)
        require(r["model"]==model and r["policy"]==name and
                all(r[k]==refs[i][k] for k in ("index","seed","task_id","dataset")) and
                r["index"]==i and r["status"] in ("success","generation_failed"), "identity")
        c=count(r,model,q,verify_plan)
        events_q=(sum(x[1] for x in r["events"] if x[0]=="decoder") if model=="gemma" else
                  r["generation_query_rows"])
        require((c["F"],c["decoder_Q"],r["output_tokens"])==(refs[i]["F"],events_q,refs[i]["N"]),
                "bill mismatch")
        for k in ("all90","original30" if i<30 else "new60"):
            totals[k].update(c)
        if i in (7,13,25,33,55,71):
            totals["pilot6"].update(c)
    return dict(model=model,q=q,source_sha256=digest.hexdigest(),cohorts=totals)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--verify-rule",action="store_true",help="Import shared rule from PYTHONPATH")
    args=parser.parse_args()
    rule=None
    if args.verify_rule:
        from bdlm_serving.methods.t2t.close_overlap import spare_successor_rows as rule
    print(json.dumps([audit(args.root,model,q,rule) for model,tiers in
          (("llada21",(4,8,16,32)),("gemma",(32,64,128,256))) for q in tiers],indent=2,sort_keys=True))
