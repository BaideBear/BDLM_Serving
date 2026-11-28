"""Reconstruct decisions from paid token IDs and original IBS event receipts."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from rank import B, Q, EOS, MASK, Versions

D = Path(__file__).resolve().parent
ROOT = Path("/data1/project_data/gx-erqw-20261123")
DATA = ROOT/"runtime/t2t-native-1125/prefix90"
def read(path): return json.loads(path.read_bytes())
def filehash(path): return sha256(path.read_bytes()).hexdigest()
def digest(x): return sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def need(ok, msg):
    if not ok: raise ValueError(msg)


def audit(r, arm, *, live=False, tape=None):
    v = Versions()
    p, cap = len(r["prompt_ids"]), r["generation_cap"]
    inputs, events, calls = r["forward_inputs"], r["events"], r["calls"]
    state = {}
    pending = {}
    cert = None
    paid = False
    stats = Counter()
    plan_log = []
    first = None
    n = 0
    for z,(e,c,ids) in enumerate(zip(events,calls,inputs,strict=True)):
        rows = e["absolute_positions"]
        need(rows==sorted(set(rows)) and len(rows)==len(ids)==c["physical_rows"]==c["rows"]==e["q_tokens"]
             and c["batch"]==1 and e["physical_forwards"]==1, "unmatched physical input")
        if e["phase"]=="prefill":
            need(z==0 and rows==list(range(p//B*B)) and ids==r["prompt_ids"][:len(ids)],"prefill")
            continue
        if cert and not live:
            break
        if e["phase"]=="certificate_debt":
            need(cert and not paid and z==len(events)-1 and rows==sorted(pending) and rows and
                 not (e["sampled_absolute"] or e["written_absolute"] or e["edited_positions"]), "terminal pay")
            need(all(x!=pending[a] for a,x in zip(rows,ids)) and rows[-1]+1==c["kv_rows"],"unpaid token ID")
            if live:
                before_close={a:new for a,old,new in tape[n-1]["changes"]}
                need(ids==[before_close[a] for a in rows],"paid stale terminal token")
            paid=True
            stats.update(pay_F=1,pay_Q=len(rows))
            pending.clear()
            continue
        need(not cert, "ordinary call after certificate")
        start = e["block"]*B
        base = [start+i for i in e["selected_local"]]
        actual = sorted(set(rows)-set(base))
        need(e["phase"]=="optimus" and 0<len(base)<=len(rows)<=Q and set(rows)==set(base)|set(actual)
             and set(pending)==set(e["dirty_before"])<=set(base) and e["offer_tick"]==n and
             e["physical_cap"]==Q and c["kv_rows"]==e["visible_kv"]==max(base)+1, "mandatory plan")
        cand = [start+i for i in e["candidate_local"]]
        expected = sorted((sorted(pending)+[a for a in cand if a not in pending])[:Q]) if len(cand)>Q else cand
        need(expected==base, "historical IBS selection")
        lookup = dict(zip(rows,ids))
        for a,x in lookup.items():
            if a in pending:
                need(x!=pending[a],"declared write but token ID did not change")
            elif a in state:
                need(x==state[a],"unreported change")
            else:
                need(x==(r["prompt_ids"][a] if a<p else MASK),"unknown initial token")
        lo, hi = max(p,start),min(start+B,p+cap)
        pool = [i for i in range(lo,min(start+e["cached_prefix_before"],hi,max(base)))
                if i in v.paid and i not in pending and i not in base and state[i]!=MASK]
        k=min(2,Q-len(base),len(pool))
        plan=v.plan(pool,k,lo,hi)
        expected_extra=[] if arm=="none" else plan[arm]
        need(actual==expected_extra and len(plan["right"])==len(plan["change"])==k,"extra set")
        need([lookup[a]==MASK for a in base]==e["selected_masked"], "masked eligibility")
        if k:
            stats.update(opportunity_ticks=1,spare_Q=k)
        if plan["right"]!=plan["change"]:
            stats.update(different_sets=1)
            if first is None:
                first=dict(tick=n,block=e["block"],sample=base,dirty=sorted(pending),**plan)
        if live:
            need(tape[n]["plan"]==json.loads(json.dumps(plan)) and tape[n]["paid"]==rows and
                 tape[n]["tick"]==n and tape[n]["mask"]==e["selected_masked"],"live version/role tape")
        plan_log.append((base,plan["right"],plan["change"]))
        state.update(lookup)
        v.project(n,rows)
        changes=set(e["written_absolute"])|set(e["edited_positions"])
        need(changes<=set(base) and set(e["edited_positions"])<=set(e["written_absolute"]),
             "unsupported unbilled revocation or unauthorized write")
        if live:
            writes=tape[n]["changes"]
            need({a for a,old,new in writes}==changes and
                 all(lookup[a]==old!=new for a,old,new in writes),"live write values")
        pending={a:lookup[a] for a in changes}
        v.write(n,changes)
        hi_ready=min(start+e["cached_prefix_after"],hi)
        need(not (set(range(lo,hi_ready)) & set(pending)), "dirty frozen prefix")
        eos=next((a for a in range(lo,hi_ready) if state[a]==EOS),None)
        if eos is not None:
            cert=dict(event=z,eos=eos,due=sorted(pending))
            # Historical traces continue. The next *real* ordinary inputs witness the changes.
            if not live and pending:
                nxt=dict(zip(events[z+1]["absolute_positions"],inputs[z+1],strict=True))
                need(all(a in nxt and nxt[a]!=x for a,x in pending.items()),"unwitnessed close debt")
        stats.update(ordinary_F=1,ordinary_Q=len(rows),extra_Q=len(actual))
        n+=1
    if live:
        failed=r["status"]=="generation_failed"
        need((not pending or failed) and (not failed or (not cert and r["request_failure"] and not r["output_ids"]))
             and (not cert or bool(cert["due"])==paid) and n==len(tape)
             and r["verified_layers"]==20 and r["generation_query_rows"]==stats["ordinary_Q"]+stats["pay_Q"]
             and r["generation_forwards"]==n+int(paid),"final bill or unpaid debt")
        if failed:
            stats["failure_outstanding_Q"]=len(pending)
        if cert:
            need(r["output_ids"][-1]==EOS and len(r["output_ids"])==cert["eos"]-p+1,"EOS completion")
    return dict(counts=dict(stats),first=first,certificate=cert,plans_sha=digest(plan_log)),plan_log


def replay():
    reg=read(D/"registration.json")
    h=DATA/reg["history"]
    run,done,ret=map(lambda name:read(h/name),("run.json","complete.json","returned.json"))
    need(filehash(h/"run.json")==reg["historical_run_sha256"]==done["study"] and
         ret["restored"] and ret["complete"]==filehash(h/"complete.json") and
         filehash(DATA/"samples.json")==reg["panel_sha256"]==run["samples_sha256"], "history seals")
    panel=read(DATA/"samples.json")["samples"]
    rows=[]
    for i in reg["cases"]:
        for key,arm in (("0","none"),("2","right")):
            path=h/f"{i:02d}-{run['arms'][key]}.json"
            r=read(path)
            need(all(r[k]==panel[i][k] for k in ("index","seed","dataset","task_id")) and
                 r["verified_layers"]==20 and r["generation_cap"]==2048, "case identity")
            out,_=audit(r,arm)
            rows.append(dict(index=i,seed=r["seed"],arm=arm,raw_sha=filehash(path),**out))
    different=sum(s["counts"].get("different_sets",0) for s in rows)
    return dict(registration=filehash(D/"registration.json"),eligibility_only=True,records=rows,
                different_sets=different,gate=bool(different))


if __name__=="__main__":
    print(json.dumps(replay(),sort_keys=True,separators=(",",":")))
