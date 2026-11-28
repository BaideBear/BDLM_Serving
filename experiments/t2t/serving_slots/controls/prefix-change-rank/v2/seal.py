"""Read-only secondary audit: requires all 18 identities and a returned holder."""
from itertools import zip_longest
from statistics import mean

from preflight import stop_oracle
from replay import DATA, D, digest, filehash, json, need, read
from study import ARMS, CASES, RETAINED, RETAINED_SHA, verify_mask, control_ref


def raw_path(i,a):
    return RETAINED if (i,a)==(7,"none") else D/f"{i:02}-{a}.json"


def verify_live(raw):
    r=raw["row"]
    verify_mask(raw["masks"],r)
    pending={}
    it=iter(raw["tape"])
    for e,ids in zip(r["events"],r["forward_inputs"],strict=True):
        pv=dict(zip(e["absolute_positions"],ids,strict=True))
        if e["phase"]=="prefill": continue
        need(len(ids)<=8 and set(pending)==set(e["dirty_before"]) and
             all(pv[a]==v for a,v in pending.items()),"missing or stale write payment")
        if e["phase"]=="certificate_debt":
            need(set(pv)==set(pending) and not e["sampled_absolute"] and not e["written_absolute"],"close")
            pending={}
            continue
        t=next(it)
        base=[32*e["block"]+j for j in e["selected_local"]]
        plus=sorted(set(pv)-set(base))
        selected_mask=[pv[a]==156895 for a in base]
        need(selected_mask==t["mask"]==e["selected_masked"] and
             (not plus or (all(pv[a]!=156895 and len(r["prompt_ids"])<=a<len(r["prompt_ids"])+2048
                         for a in plus) and plus==e["prefix_refresh_absolute"])),"role contamination")
        pending={a:new for a,old,new in t["changes"]}
        need(set(pending)==set(e["written_absolute"])<=set(base) and
             all(old==pv[a]!=new for a,old,new in t["changes"]),"reported write IDs")
    need(not pending and next(it,None) is None and r["status"]=="success" and
         r["generation_physical_padded_rows"]==r["generation_query_rows"] and
         r["verified_layers"]==20 and r["generation_query_rows"]+r["prefill_rows"]==
         sum(c["physical_rows"] for c in r["calls"]),"incomplete strict trajectory")


def main():
    done=read(D/"complete.json")
    rows=[json.loads(l) for l in (D/"compact.ndjson").read_text().splitlines()]
    hand=[json.loads(l) for l in (D/"handoff.ndjson").read_text().splitlines()]
    need([h["phase"] for h in hand]==["before","returned"] and filehash(RETAINED)==RETAINED_SHA and
         done["requests"]==18 and done["new_requests"]==17 and done["sha"]==digest(rows) and
         [r["index"] for r in rows]==CASES and not done["extension"] and len(list(D.glob("??-*.json")))==17,
         "incomplete study")
    reg=read(D/"registration.json")
    h=DATA/reg["history"]
    labels=read(h/"run.json")["arms"]
    out=[]
    for r in rows:
        i=r["index"]
        live={}
        arms=r["arms"]
        for a in ARMS:
            path=raw_path(i,a)
            raw=read(path)
            x=live[a]=raw["row"]
            need(filehash(path)==arms[a]["raw_sha"] and raw["arm"]==a and
                 (x["index"],x["seed"],x["task_id"])==(i,r["seed"],r["task_id"]),"raw identity")
            verify_live(raw)
            if a!="change":
                hist=read(h/f"{i:02}-{labels['0' if a=='none' else '2']}.json")
                cert=stop_oracle(hist)
                need(cert==arms[a]["certificate"],"original paid stop differs")
                control_ref(x,hist,cert)
        def match(a,b):
            s,t=live[a]["output_ids"],live[b]["output_ids"]
            idx=next((k for k,(u,v) in enumerate(zip_longest(s,t)) if u!=v),None)
            return dict(ids_equal=idx is None,first_output_offset=idx,
                delta_N=arms[b]["N"]-arms[a]["N"],delta_F=arms[b]["F"]-arms[a]["F"],
                delta_Q=arms[b]["Q"]-arms[a]["Q"],delta_correct=int(arms[b]["correct"])-int(arms[a]["correct"]))
        out.append(dict(index=i,seed=r["seed"],dataset=r["dataset"],stats={
            a:{**{k:arms[a][k] for k in ("N","F","Q","correct","prefill_Q","certificate")},
               "extra_Q":arms[a]["roles"].get("extra_Q",0)} for a in ARMS},
            fork={k:r["first"][k] for k in ("tick","ordinary","right","change","physical_Q")} if r["first"] else None,
            right_minus_none=match("none","right"),change_minus_none=match("none","change"),
            change_minus_right=match("right","change")))
    def group(rs):
        z={}
        for a in ARMS:
            s=[r["stats"][a] for r in rs]
            totals={k:sum(r[k] for r in s) for k in ("N","F","Q","extra_Q","correct")}
            z[a]=dict(**totals,macro_TPF=mean(r["N"]/r["F"] for r in s),
                pooled_TE=totals["N"]/totals["Q"])
        return z
    return dict(cases=CASES,registration=done["registration"],compact_sha=filehash(D/"compact.ndjson"),
        new_requests=17,retained=RETAINED_SHA,returned=hand[-1],groups={
            "all6":group(out),"original3":group(out[:3]),"additional3":group(out[3:])},pairs=out)


if __name__=="__main__":
    print(json.dumps(main(),sort_keys=True,separators=(",",":")))
