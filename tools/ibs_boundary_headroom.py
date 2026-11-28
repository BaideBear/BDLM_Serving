"""Fixed-ID cost sensitivity of one successor launch per clean IBS boundary."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from ibs_close_overlap import count, trace
from ibs_frozen_eos_tail import require


def bill_work(rows, model, prompt):
    b = 32 if model == "llada21" else 256
    previous = None
    work = Counter()
    for block, positions, written, edited, _, masked, expected_due in rows:
        if block != previous:
            require(previous is None or not due, "carried debt across IBS blocks")
            lo = b*block
            resolved = {p for p in range(lo, lo+b) if p < prompt}
            resident, due = set(), set()
            previous = block
        pos = set(positions)
        require(len(positions) == len(pos) and due <= pos, "missing payment")
        if expected_due is not None:
            require(due == set(expected_due), "mismatched debt")
        masks = pos-resolved
        require(masked is None or list(masked) == [p in masks for p in positions], "mask mismatch")
        if masks:
            label = "mixed_mask_debt" if due else "mask_no_debt"
        else:
            label = "no_mask_with_debt" if due else "no_mask_no_debt"
        work[label+"_F"] += 1
        work["debt_Q"] += len(due)
        work["unresolved_first_Q"] += len(masks-resident)
        work["unresolved_revisit_Q"] += len(masks & resident)
        work["other_resolved_Q"] += len(pos-due-masks)
        new = set(written)
        work["first_view_admits"] += len(new & (masks-resident))
        work["revisit_admits"] += len(new & masks & resident)
        work["edited_row_visits"] += len(edited)
        require(due <= resolved and new <= pos, "invalid projected state")
        resolved.difference_update(set(edited)-new)
        resolved.update(new)
        due = new
        resident.update(pos)
    return work


def bound(n, f, launches):
    """Optimistic what-if, not a bound on trajectories that change their IDs."""
    require(0 <= launches < f and n >= 0, "invalid fixed-trajectory bill")
    return n/(f-launches)


def paths(root, model, q):
    if model == "llada21":
        return root/("llada21-optimus-hardcap-"+{4:"low",8:"endpoints",16:"endpoints",32:"anchor"}[q]), f"optcap_ibs_{q}"
    return (root/"gemma-fixed-caps-q32q64-v1" if q <= 64 else
            root.parent/"private-optimus-cap90-v1/gemma-optimus-cap90-v1"), f"ibs_cap_{q}"


def summarize(records):
    c = Counter()
    for r in records:
        c.update(r["work"])
    return dict(n=len(records), N=sum(r["N"] for r in records), F=sum(r["F"] for r in records),
                Q=sum(r["Q"] for r in records), actual_tpf=mean(r["actual"] for r in records),
                one_free_successor_call_tpf=mean(r["ideal"] for r in records),
                cases_with_clean_promotion=sum(r["launches"]>0 for r in records),
                clean_launches=sum(r["launches"] for r in records), work=c)


def audit(root, model, q, ledger):
    source, policy = paths(root, model, q)
    refs = ledger["arms"][model+":"+policy]["items"]
    require(len(refs)==90, "wrong cohort")
    digest = sha256()
    records = []
    for i, ref in enumerate(refs):
        raw = (source/f"{i:02}-{policy}.json").read_bytes()
        digest.update(sha256(raw).digest())
        r = json.loads(raw)
        require(r["index"]==i and r["model"]==model and r["policy"]==policy and
                all(ref[k]==r[k] for k in ("task_id","seed","dataset","index")), "identity")
        s = count(r, model, q)
        f, n = s["F"], r["output_tokens"]
        require((f,n,r["generation_query_rows"])==(ref["F"],ref["N"],ref["Q"]), "wrong reference bill")
        launches = s["promoted_no_later_edit"]
        work = bill_work(trace(r,model), model, len(r["prompt_ids"]) if model=="llada21" else 0)
        require(sum(v for k,v in work.items() if k.endswith("_F"))==s["decoder_F"] and
                sum(v for k,v in work.items() if k.endswith("_Q"))==s["decoder_Q"], "work partition mismatch")
        records.append(dict(N=n,F=f,Q=s["decoder_Q"],launches=launches,
                            work=work,actual=n/f,ideal=bound(n,f,launches)))
    return dict(model=model,q=q,source_sha256=digest.hexdigest(),all90=summarize(records),
                original30=summarize(records[:30]),new60=summarize(records[30:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    raw = (args.root/"fixed-family-reference-case-ledger-v1.json").read_bytes()
    require(sha256(raw).hexdigest()=="ff9caccbf776416f0237ddcc73af774fbe567c57929b359583f70fba128d64ce",
            "unexpected frozen ledger")
    ledger = json.loads(raw)
    print(json.dumps([audit(args.root,m,q,ledger) for m,qs in
                      (("llada21",(4,8,16,32)),("gemma",(32,64,128,256))) for q in qs],
                     sort_keys=True,indent=2))
