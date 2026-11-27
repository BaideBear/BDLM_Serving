"""Confirm the sealed IBS8 spare-prefix pair and audit unregistered revocations."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean


def check(ok, why):
    if not ok: raise ValueError(why)


def aligned(*columns):
    check(len({len(col) for col in columns}) == 1, "Mismatched trace lengths")
    return zip(*columns)


SEATS = ("dirty_Q", "free_mask_Q", "free_decoded_Q", "prompt_Q", "prefix_Q", "idle")


def inspect(row, quota):
    counts=Counter(); debt=set(); revokes=set(); first=None; mask_id=None
    for e,call,ids in aligned(row["events"],row["calls"],row["forward_inputs"]):
        positions=e["absolute_positions"]
        lookup=dict(aligned(positions,ids))
        check(positions==sorted(lookup) and len(lookup)==e["q_tokens"]==call["physical_rows"]==call["rows"]
              and call["batch"]==1,"Wrong physical forward")
        if e["phase"]=="prefill":
            check(not counts["F"],"Prefill after generation")
            continue
        q=len(positions);tick=counts["F"]; start=e["block"]*32
        selected=[start+i for i in e["selected_local"]]
        sampled=set(selected)
        extra=set(positions)-sampled
        check(e["offer_tick"]==tick and e["physical_cap"]==8 and 0<q<=8
              and max(selected)+1==e["visible_kv"]==call["kv_rows"] and sampled<=lookup.keys()
              and len(extra)<=min(quota,8-len(selected)) and
              all(max(start,len(row["prompt_ids"]))<=v<start+e["cached_prefix_before"] for v in extra),
              "Changed window, sampling geometry or prefix bounds")
        if quota:
            check(extra==set(e["prefix_refresh_absolute"]),"Wrong extra rows")
        if extra and first is None: first=tick
        mask_rows={v for v,m in aligned(selected,e["selected_masked"]) if m}
        masked={lookup[v] for v in mask_rows}
        if masked:
            check(len(masked)==1 and (mask_id is None or masked=={mask_id}),"Ambiguous MASK ID")
            mask_id=next(iter(masked))
        check(e["dirty_before"]==sorted(debt) and debt<=sampled
              and all(lookup[p]==mask_id for p in revokes & lookup.keys()),"Wrong token debt or revocation")
        other=sampled-debt
        prompt={p for p in other if p<len(row["prompt_ids"])}
        check(not prompt & mask_rows and len(selected)==len(sampled),"Invalid prompt or repeated sampler seat")
        counts.update(dirty_Q=len(debt),free_mask_Q=len(other & mask_rows),
                      free_decoded_Q=len(other-prompt-mask_rows),prompt_Q=len(prompt),idle=8-q,
                      all_seats_dirty_F=int(len(debt)==8),at_cap_F=int(q==8))
        edits=set(e["edited_positions"]); writes=set(e["written_absolute"])
        counts["revoked_not_paid_next"]+=len(revokes-lookup.keys())
        revokes=edits-writes
        check(edits|writes<=sampled and not ((edits|writes)&extra),"Extra row mutated")
        debt=writes
        counts.update(F=1,Q=q,prefix_Q=len(extra),rounds=bool(extra),edited=len(edits),
                      recorded_writes=len(writes),revoked=len(revokes))
    check(not debt and not revokes and row["verified_layers"]==20 and row["generation_cap"]==2048
          and row["generation_forwards"]==counts["F"] and row["generation_query_rows"]==counts["Q"],
          "Unpaid terminal debt or missing layer proof")
    check(sum(counts[k] for k in SEATS)==8*counts["F"],"Missing or double-billed seats")
    return counts,dict(first=first,rounds=counts["rounds"],prefix_Q=counts["prefix_Q"],
                       N=row["output_tokens"],F=counts["F"],Q=counts["Q"])


def audit(root,manifest):
    def read(name):return json.loads((root/name).read_bytes())
    digest=lambda name:sha256((root/name).read_bytes()).hexdigest()
    m=json.loads(manifest.read_bytes());s=read("run.json");done=read("complete.json");ret=read("returned.json")
    score=read("aggregate.json")
    check(s["formal"] and s["q"]==8 and s["indices"]==done["indices"]==list(range(90))
          and s["samples_sha256"]==score["samples_sha256"]==m["samples_sha256"] and len(done["pairs"])==90
          and done["study"]==digest("run.json") and ret["restored"] and ret["complete"]==digest("complete.json"),
          "Not a sealed frozen90 pair")
    data={"control":[],"refresh":[]}; hashes=sha256()
    for task,pair in aligned(m["samples"],done["pairs"]):
        i=task["index"]; rows=[]
        for k,side in ((0,"control"),(2,"refresh")):
            label=s["arms"][str(k)]
            blob=(root/f"{i:02}-{label}.json").read_bytes();hashes.update(sha256(blob).digest())
            a=json.loads(blob)
            check(a["policy"]==label and a["status"]=="success"
                  and all(a[key]==task[key] for key in ("index","task_id","dataset","seed")),"Wrong input")
            count,bill=inspect(a,k)
            check(bill==pair[side],"Did not match sealed per-request bill")
            count["N"]=bill["N"]
            data[side].append(count)
            rows.append(a)
        check(pair["index"]==i and (rows[0]["output_ids"]==rows[1]["output_ids"])==pair["same_ids"]
              and rows[0]["prompt_ids"]==rows[1]["prompt_ids"],"Wrong pair")
    out={}
    for name,sl in (("full90",slice(None)),("original30",slice(0,30)),("new60",slice(30,None))):
        g,=[v for v in score["statistics"] if v["cohort"]==name and v["scope"]=="all"]
        totals={}
        for key,alias in (("control","control"),("refresh","trial")):
            cases=data[key][sl];total=sum(cases,Counter())
            stat=g[alias]
            check(all(total[k]==stat[k] for k in ("N","F","Q")) and total["prefix_Q"]==stat["injected_prefix_Q"]
                  and len(cases)==g["requests"] and abs(mean(v["N"]/v["F"] for v in cases)-stat["macro_TPF"])<1e-8,
                  "Mismatched aggregate and sealed results")
            totals[key]={k:total[k] for k in ("revoked","revoked_not_paid_next","edited","recorded_writes")}
        split={}
        for key in ("control","refresh"):
            t=sum(data[key][sl],Counter())
            split[key]={k:t[k] for k in (*SEATS,"F","at_cap_F","all_seats_dirty_F")}
        out[name]=dict(score=g,debt_audit=totals,offered_seats=split)
    return dict(raw_root=str(root),run_sha256=digest("run.json"),pairs_sha256=hashes.hexdigest(),
                samples_sha256=m["samples_sha256"],cohorts=out)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path);p.add_argument("manifest",type=Path)
    print(json.dumps(audit(**vars(p.parse_args())),indent=2,sort_keys=True))
