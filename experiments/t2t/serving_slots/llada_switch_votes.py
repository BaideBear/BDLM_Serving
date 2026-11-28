"""Independent per-step admission reconstruction from measured sampler output."""
def audit(q, head, warm, event, mask, previous, *, frontier_left=None, frontier_added=None):
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
    run=set()
    if frontier_left is not None:
        if q!=32 or not warm or not head<=frontier_left<head+32:
            raise ValueError("invalid Q32 frontier")
        paid=dict(zip(rows,proposals))
        for p in range(frontier_left,head+32):
            if p not in paid: break
            old,draft,confidence,low,high=paid[p]
            if old!=mask or draft==mask or not low: break
            run.add(p)
    if frontier_added is not None:
        frontier_added.update(run-writes-extras)
    if set(event["admitted"])!=writes|extras|run:
        raise ValueError("wrong admission after a change of head, offer or evidence")
    return votes,extras
