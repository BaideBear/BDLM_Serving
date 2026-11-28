"""Independent per-step admission reconstruction from measured sampler output."""
def audit(q, head, warm, event, mask, previous):
    rows=event["sampled"]
    proposals=event.get("proposals",[])
    if len(rows)!=len(proposals) or not set(rows)<=set(event["rows"]):
        raise ValueError("not the paid sample")
    native=set()
    per_block={}
    for p,(old,draft,conf,low,high) in zip(rows,proposals):
        if old==mask:
            per_block.setdefault(p//32,[]).append((p,conf))
            if high: native.add(p)
    for block,candidates in per_block.items():
        if not any(p in native for p,_ in candidates):
            native.add(max(candidates,key=lambda x:x[1])[0])
    votes={}
    writes=set()
    extras=set()
    for p,(old,draft,conf,low,high) in zip(rows,proposals):
        if old!=mask or draft==mask:
            continue
        if p in native:
            writes.add(p)
        elif q==64 and warm and head<=p<head+32 and low:
            votes[p]=(head,draft)
            if previous.get(p)==votes[p]:
                extras.add(p)
    if set(event["admitted"])!=writes|extras:
        raise ValueError("wrong admission after a change of head, offer or evidence")
    return votes,extras
