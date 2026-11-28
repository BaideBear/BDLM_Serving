"""Run on n164 only. Reconstruct the upstream slot lifecycle without a model."""
import argparse
from collections import Counter
import json
from pathlib import Path

B, MASK, EOS, CAP = 32, 156895, 156892, 2048


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def verify(r):
    prompt = r["prompt_ids"]
    length = ((len(prompt)+CAP+B-1)//B)*B
    first = (len(prompt)//B)*B
    require(MASK not in prompt and r["cap"]==CAP,"prompt or cap")
    canvas = prompt+[MASK]*(length-len(prompt))
    counts = Counter(partial_prompt_requests=int(len(prompt)%B!=0))
    c, right, blocks = 0, 1, (length-first)//B
    steps, done, frontier, post = Counter(), set(), Counter(), Counter()
    stopped = False
    for tick in r["ticks"]:
        require(not stopped and (tick["head"],tick["commit"],tick["right"])==
                (first+c*B,c,right), "unexpected head, admission or call after EOS")
        slots = tick["slots"]
        require(len(slots)==right-c==tick["open"], "not all open slots attended")
        for g, slot in enumerate(slots,c):
            lo = first+B*g
            span = slice(lo,lo+B)
            immutable = [p<len(prompt) for p in range(lo,lo+B)]
            require(slot["ids"]==canvas[span] and slot["prompt"]==immutable and
                    slot["pe0"]==post[g] and slot["fin0"]==(g in done), "slot input")
            old,new = slot["ids"],slot["new"]
            require(len(new)==B and all(not flag or x==y for flag,x,y in zip(immutable,old,new)),
                    "immutable prompt changed")
            if g in done:
                require(new==old and slot["fin"],"locally finished slot changed")
                counts["finished_open_slot_projections"]+=1
            counts["actual_m2t_writes"]+=sum(x==MASK and y!=MASK for x,y in zip(old,new))
            counts["actual_t2t_writes"]+=sum(x!=MASK and y!=x for x,y in zip(old,new))
            canvas[span] = new
            steps[g]+=1
            frontier[g],post[g] = slot["frontier"],slot["pe"]
            if slot["fin"] or steps[g]>=B+16+1:
                done.add(g)
        while c in done:
            lo = first+c*B
            c+=1
            if EOS in canvas[lo:lo+B]:
                counts["eos_commits"]+=1
                canvas[lo+B:] = [EOS]*(len(canvas)-lo-B)
                stopped=True
                break
            if c==blocks:
                stopped=True
                break
        while not stopped and right-c<2 and right<blocks:
            if right>c and frontier[right-1]<.8:
                break
            right+=1
    require(stopped, "terminal reason not accounted for")
    out=canvas[len(prompt):len(prompt)+CAP]
    seq=out[:out.index(EOS)+1] if EOS in out else out
    require(seq==r["output_ids"] and r["eos"]==(EOS in out) and MASK not in seq,"terminal output")
    return counts


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory",type=Path)
    args=ap.parse_args()
    rows = [json.loads(l) for l in (args.directory/"compact.ndjson").read_bytes().splitlines()]
    output = {}
    for arm in ("flow_w2","active_w2"):
        stats = Counter()
        for z in rows:
            item=json.loads((args.directory/f"{z['index']:02d}-{arm}.json").read_bytes())
            require(item["index"]==z["index"] and item["arm"]==arm,"raw identity")
            stats.update(verify(item))
        output[arm]=dict(requests=len(rows),**stats)
    print(json.dumps(output,sort_keys=True))


if __name__=="__main__":
    main()
