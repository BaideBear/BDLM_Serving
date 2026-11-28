"""Independent bill and first-fork checks on real Q16 trajectories."""
from tools import llada_ibs_eos_q16_eval as base
from tools.llada_ibs_eos_stop import require


def bill(r,tape,arm,*,eos,mask,excluded):
    # Reuse the full original physical/paid-stop audit, not its apply-only vote comparator.
    out=base.audit(r,dict(tape,ticks=[]),"stock",eos=eos,mask=mask,excluded=excluded)
    if arm=="stock":
        require(not tape["ticks"],"stock got votes")
        return out
    ordinary=[e for e in r["events"] if e["phase"]=="optimus"]
    p=len(r["prompt_ids"])
    expected={e["offer_tick"]:[a for a,m in zip(e["absolute_positions"],e["selected_masked"],strict=True)
        if m and p<=a<p+r["generation_cap"]] for e in ordinary}
    require([t["tick"] for t in tape["ticks"]]==[n for n,rows in expected.items() if rows],"missing measured tick")
    for t,want in base.independent_eligible(tape["ticks"],excluded):
        n=t["tick"]
        e=ordinary[n]
        require(t["block"]==e["block"] and t["positions"]==expected[n],"unmeasured candidate")
        data={a:(v,prob,native) for a,v,prob,native in zip(t["positions"],t["proposals"],
            t["probabilities"],t["native"],strict=True)}
        allowed=[a for a,(v,s,b) in data.items() if not b and v not in excluded and s>.5]
        other=sorted(allowed,key=lambda a:(-data[a][1],a))[:len(want)]
        applied=set(t["applied"])
        require(len(applied)==len(want)==len(t["applied"]) and applied==set(want if arm=="repeat" else other),
                "incorrect dose or membership")
        writes=dict(tape["writes"][n])
        # Only ordinary T2T can write a previously unmasked row.
        require(all((a in writes)==(b or a in applied) and (a not in writes or writes[a]==v)
                    for a,(v,_,b) in data.items()) and
                set(writes)-set(data)==set(e["edited_positions"]), "native or added writes drifted")
        out["eligible"]+=len(want)
        out["extra"]+=len(applied)
    return out


def fork(rows,tapes):
    l,r=(rows[a] for a in ("repeat","current"))
    lt,rt=(tapes[a] for a in ("repeat","current"))
    lts={x["tick"]:x for x in lt["ticks"]}
    rts={x["tick"]:x for x in rt["ticks"]}
    require(l["prompt_ids"]==r["prompt_ids"],"prompt")
    for j,(e,f) in enumerate(zip(l["events"],r["events"])):
        require(l["calls"][j]==r["calls"][j] and l["forward_inputs"][j]==r["forward_inputs"][j],
                "earlier physical/input drift")
        if e["phase"]!="optimus":
            require(e==f,"earlier prefill/terminal drift")
            continue
        n=e["offer_tick"]
        a,b=lts.get(n),rts.get(n)
        require(bool(a)==bool(b),"missing observer")
        if a:
            require({k:v for k,v in a.items() if k!="applied"}=={k:v for k,v in b.items() if k!="applied"},
                    "earlier votes or native drift")
        x,y=(set(a["applied"]),set(b["applied"])) if a else (set(),set())
        lw,rw=dict(lt["writes"][n]),dict(rt["writes"][n])
        if x!=y:
            require(len(x)==len(y) and
                    {k:v for k,v in e.items() if k not in ("written_absolute","cached_prefix_after")}==
                    {k:v for k,v in f.items() if k not in ("written_absolute","cached_prefix_after")} and
                    {k:v for k,v in lw.items() if k not in x}=={k:v for k,v in rw.items() if k not in y} and
                    x<=lw.keys() and y<=rw.keys(),"not a pure same-dose fork")
            return dict(tick=n,block=a["block"],k=len(x),repeat=sorted(x),current=sorted(y),
                paid=e["absolute_positions"],dirty=e["dirty_before"],native_T2T=e["edited_positions"],
                measured=[[p,v,s,b] for p,v,s,b in zip(a["positions"],a["proposals"],
                    a["probabilities"],a["native"],strict=True)],common_through_calls=j)
        require(e==f and lw==rw,"earlier event/write drift")
    require(l==r and lt==rt,"unexplained endpoint divergence")
    return None
