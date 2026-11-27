"""Attribute a fixed LLaDA request's real decoder work to the then-current head and tail."""
import argparse
from collections import Counter
import csv
import sys

from .llada_all_slots90 import DATA, read, require, sha
from .fixed_family_report import COHORTS, report


def ledger(case, b=32, cap=2048):
    origin=case["prefill_Q"]
    limit=origin+cap
    q=case["q"]
    output_end=origin+len(case["output_ids"])
    counts=Counter(dict(head_Q=0,tail_Q=0,both_F=0,tail_only_F=0,head_only_F=0,
                        tail_cold_F=0,warm_head_no_mask_sample_F=0,possible_head_mask_Q=0,
                        tail_first_m2t=0,tail_first_in_output=0,tail_first_later_edited=0,
                        beyond_output_Q=0,tail_beyond_output_Q=0))
    decoded,initialized,first,edits=set(),set(),{},set()
    head=origin//b
    for tick,e in enumerate(case["trace"]):
        require(e["block"]==head and e["tick"]==tick and e["offer"]==q,"Incorrect physical block or offer")
        rows=set(e["rows"])
        require(e["actual_q"]==e["physical_q"]==len(rows)==e["q"]<=q and
                e["layer_physical_q"]==[len(rows)]*20,"Incorrect projection")
        split=(head+1)*b
        h={i for i in rows if i<split}
        t=rows-h
        require(all(max(origin,head*b)<=p<min(limit,split+b) for p in rows),"Unexpected third block")
        counts["head_Q"]+=len(h)
        counts["tail_Q"]+=len(t)
        counts["both_F" if h and t else "head_only_F" if h else "tail_only_F"]+=1
        counts["beyond_output_Q"]+=sum(p>=output_end for p in rows)
        counts["tail_beyond_output_Q"]+=sum(p>=output_end for p in t)
        masks=set(range(max(origin,head*b),min(split,limit)))-decoded
        sampled=set(e["sampled"])
        d=set(e["dirty"])
        cold=set(e["cold"])
        if cold&t:
            counts["tail_cold_F"]+=1
            if masks and masks<=initialized and not masks&sampled:
                # A warm unresolved head was passed over. Count only missing masks
                # that could go in unused physical seats, keeping all original rows.
                counts["warm_head_no_mask_sample_F"]+=1
                counts["possible_head_mask_Q"]+=min(q-len(rows),len(masks-rows-d))
        a=set(e["admitted"])
        changed=set(e["edited"]+e["revoked"])
        require(a|changed<=sampled<=rows and not a&decoded and changed<=decoded and
                cold==rows-initialized,"Invalid admission, edit or first projection")
        for p in a:
            first.setdefault(p,p in t)
        edits.update(p for p in changed if first[p])
        decoded.update(a)
        decoded.difference_update(e["revoked"])
        initialized.update(rows)
        for retired in e["retired_blocks"]:
            require(retired==head,"Out-of-order retirement")
            head+=1
    counts["tail_first_m2t"]=sum(first.values())
    counts["tail_first_in_output"]=sum(t and p<output_end for p,t in first.items())
    counts["tail_first_later_edited"]=sum(p<output_end for p in edits)
    require(counts["head_Q"]+counts["tail_Q"]==case["Q"] and
            sum(counts[k] for k in ("both_F","tail_only_F","head_only_F"))==case["F"],
            "Work does not reconcile")
    return counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=("cpu","cuda"),required=True)
    parser.add_argument("--cohort",choices=("smoke3",*COHORTS),default="smoke3")
    args=parser.parse_args()
    idx=(0,10,20) if args.cohort=="smoke3" else sorted(COHORTS[args.cohort])
    snapshot=report(DATA,"llada21",args.device)
    verified=(snapshot["protocol_smoke3"] if args.cohort=="smoke3" else snapshot["cohorts"][args.cohort])
    require({32,64}<=verified.keys(),"Missing complete paired tiers")
    root=DATA/"llada21-fixed-family-v1"/args.device/"law-1-8"
    digest=sha(root/"run.json")
    records=[]
    for i in idx:
        key=None
        for q in (32,64):
            case=read(root/f"{i:02}-q{q}.json")
            ident=tuple(case[k] for k in ("index","seed","task_id","input_sha256"))
            require(case["registration_sha256"]==digest and case["q"]==q and ident[0]==i
                    and (key is None or ident==key),"Mismatched paired request")
            key=ident
            records.append(dict(device=args.device,index=i,q=q,correct=int(case["correct"]),
                       failure=int(bool(case["failure"])),N=case["N"],F=case["F"],paid_Q=case["Q"],
                       **ledger(case)))
    w=csv.DictWriter(sys.stdout,fieldnames=records[0],lineterminator="\n")
    w.writeheader()
    w.writerows(records)


if __name__=="__main__":
    main()
