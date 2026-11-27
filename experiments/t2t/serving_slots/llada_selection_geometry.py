"""Compare the capped IBS row rule on recorded states. No model or logits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

B=32


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def policy(lo, end, masked, dirty, q):
    span=set(range(lo,end))
    decoded=span-masked
    reserve=max(1,q-len(decoded))
    candidate=decoded | set(sorted(span & masked)[:reserve])
    check(dirty<=candidate,"prefix hid a mandatory write")
    rows=sorted(dirty)+sorted(candidate-dirty)
    return tuple(sorted(candidate)),tuple(sorted(rows[:q]))


def verify_ibs(r,q):
    origin=len(r["prompt_ids"])
    resolved=set(range(origin))
    c=Counter()
    for e in r["events"]:
        if e["phase"]=="prefill":
            continue
        base=B*e["block"]
        masked=set(range(base,base+B))-resolved
        before=set(e["dirty_before"])
        candidate,picked=policy(base+e["cached_prefix_before"],base+B,masked,before,q)
        check(e["mode"]=="ibs" and e["physical_cap"]==q and
              candidate==tuple(base+i for i in e["candidate_local"]) and
              picked==tuple(e["absolute_positions"]) and
              [p in masked for p in picked]==e["selected_masked"],"IBS native replay mismatch")
        wrote,edits=set(e["written_absolute"]),set(e["edited_positions"])
        # An edit back to MASK is in edited_positions but is not a new non-MASK write.
        resolved.difference_update(edits-wrote)
        resolved.update(wrote)
        c.update(F=1,Q=len(picked))
    check(c["F"]==r["generation_forwards"] and c["Q"]==r["generation_query_rows"],"IBS cost mismatch")
    return c


def apply_to_request(r,q):
    origin=r["prefill_Q"]
    resolved=set(range(origin))
    c=Counter()
    for e in r["trace"]:
        base=B*e["block"]
        end=min(base+B,origin+2048)
        span=set(range(max(base,origin),end))
        before=set(e["dirty_before"])
        rows=set(e["rows"])
        mask=span-resolved
        check(before<=span and rows<=span and len(rows)==e["q"]==e["actual_q"]
              and e["visible_kv"]>=max(rows)+1,"bad Request bill")
        c.update(F=1,Q=len(rows))
        if base>=origin and end==base+B and mask:
            # This is a counterfactual single step on OUR state. Its prefix is
            # reconstructed, not the state of an independently evolving IBS request.
            lo=min(mask|before)
            _,alt=policy(lo,end,mask,before,q)
            selected=set(alt)
            c.update(compared_F=1,different_F=int(selected!=rows),
                     changed_mask_set_F=int(mask&selected!=mask&rows),
                     request_only_rows=len(rows-selected),ibs_rule_only_rows=len(selected-rows),
                     theirs_settled_interior=len((selected-rows)&resolved-before),
                     theirs_mask=len((selected-rows)&mask-before),
                     ours_settled_interior=len({p for p in rows-selected if p>=lo and p in resolved-before}),
                     ours_optional_prefix=len({p for p in rows-selected if p<lo}),
                     ours_mask=len((rows-selected)&mask-before),
                     smaller_visible_right_on_actual_Q_F=int(e["visible_kv"]>max(rows)+1),
                     smaller_visible_right_on_shadow_Q_F=int(e["visible_kv"]>max(selected)+1))
        admitted,edited,revoked=map(set,(e["admitted"],e["edited"],e["revoked"]))
        check(admitted<=mask and edited|revoked<=resolved,"token state desynchronized")
        resolved.update(admitted)
        resolved.difference_update(revoked)
    check(c["F"]==r["F"] and c["Q"]==r["Q"],"Request cost mismatch")
    check(c["theirs_settled_interior"]+c["theirs_mask"]==c["ibs_rule_only_rows"],
          "unclassified shadow seat")
    check(c["ours_settled_interior"]+c["ours_optional_prefix"]+c["ours_mask"]==c["request_only_rows"],
          "unclassified Request seat")
    return c


def run(ibs,req,manifest,tiers):
    result={}
    for q in tiers:
        by_i=[]
        digest=hashlib.sha256()
        for i,task in enumerate(manifest["samples"]):
            a=(ibs/f"{i:02d}-optcap_ibs_{q}.json").read_bytes()
            b=(req/f"c{q}-noborrow-{i:02d}.json").read_bytes()
            digest.update(hashlib.sha256(a).digest()+hashlib.sha256(b).digest())
            x,y=json.loads(a),json.loads(b)
            check(i==task["index"] and all(x[k]==y[k]==task[k] for k in ("index","task_id","dataset","seed"))
                  and x["generation_cap"]==2048 and x["policy"]==f"optcap_ibs_{q}"
                  and y["q"]==q and y["cross_open"]=="head" and y["borrow"] is False and
                  y["samples_sha256"]==manifest["samples_sha256"] and y["input_sha256"]==
                  hashlib.sha256(json.dumps(x["prompt_ids"],separators=(",",":")).encode()).hexdigest(),
                  "unpaired or unregistered case")
            by_i.append((verify_ibs(x,q),apply_to_request(y,q)))
        check(len(by_i)==90,"missing cases")
        cohorts={}
        for name,sub in (("original30",by_i[:30]),("new60",by_i[30:]),("frozen90",by_i)):
            cohorts[name]=dict(pairs=len(sub),ibs=dict(sum((a for a,b in sub),Counter())),
                               request=dict(sum((b for a,b in sub),Counter())))
        result[q]=dict(cohorts=cohorts,source_sha256=digest.hexdigest())
    return dict(protocol="llada-capped-ibs-geometry-v1",samples_sha256=manifest["samples_sha256"],offers=result)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ("ibs","request","manifest"):
        ap.add_argument("--"+key,type=Path,required=True)
    a=ap.parse_args()
    print(json.dumps(run(a.ibs,a.request,json.loads(a.manifest.read_bytes()),(8,16)),indent=2,sort_keys=True))
