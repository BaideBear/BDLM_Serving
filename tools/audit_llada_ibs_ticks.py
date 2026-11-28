"""Where does strict fixed-window IBS actually admit and edit on frozen90?"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

TIERS=(4,8,16,32)
RUN_DIRECTORIES={4:"llada21-optimus-hardcap-low",8:"llada21-optimus-hardcap-endpoints",
                 16:"llada21-optimus-hardcap-endpoints",32:"llada21-optimus-hardcap-anchor"}


def check(ok,why):
    if not ok:
        raise ValueError(why)


def inspect(r,q):
    prompt=len(r["prompt_ids"])
    resident=set()
    owed=set()
    cnt=Counter()
    check(len(r["events"])==len(r["calls"]),"missing physical forward")
    for e,f in zip(r["events"],r["calls"]):
        pos=e["absolute_positions"]
        rows=set(pos)
        check(pos==sorted(rows) and f["batch"]==1 and len(pos)==e["q_tokens"]==f["rows"]==f["physical_rows"],
              "wrong physical forward")
        if e["phase"]=="prefill":
            check(not cnt and not resident and pos==list(range(prompt//32*32)),"bad prefill")
            resident.update(pos)
            continue
        check(0<len(pos)<=q==e["physical_cap"] and e["offer_tick"]==cnt["F"] and
              e["mode"]=="ibs" and sorted(rows)==[32*e["block"]+p for p in e["selected_local"]]
              and max(rows)+1==e["visible_kv"]==f["kv_rows"] and len(rows)==len(e["selected_masked"]),
              "wrong sampling width or cap")
        dirty=set(e["dirty_before"])
        check(dirty==owed and dirty<=rows,"missing/forgotten write")
        masked={p for p,flag in zip(pos,e["selected_masked"]) if flag}
        cold=masked-resident
        warm=masked&resident
        writes=set(e["written_absolute"])
        edited=set(e["edited_positions"])
        revoked=edited-writes
        admit=writes-edited
        check(admit<=masked and edited<=rows-masked and all(p>=prompt for p in edited|writes)
              and len(edited)==e["edited"] and len(admit)<=e["accepted"],"unknown write type")
        # Historical recorder excludes token -> MASK from dirty. Report these
        # separately; never use its ledger to justify dropping a real debt.
        owed=(owed-rows)|writes
        resident.update(rows)
        cnt.update(F=1,Q=len(pos),N_admitted=len(admit),cold_Q=len(cold),warm_Q=len(warm),
                   cold_admits=len(admit&cold),warm_admits=len(admit&warm),dirty_Q=len(dirty),
                   edits=len(edited),edits_dirty=len(edited&dirty),edits_clean=len(edited-dirty),
                   revokes=len(revoked),prompt_Q=sum(p<prompt for p in rows),
                   all_dirty_F=dirty==rows,cap_full_debt_F=len(dirty)==q,
                   all_dirty_edits=len(edited) if dirty==rows else 0,
                   cap_full_debt_edits=len(edited) if len(dirty)==q else 0,
                   prompt_only_F=all(p<prompt for p in rows),
                   mixed_mask_F=bool(cold and warm),cold_with_decoded_F=bool(cold and rows-masked),
                   mixed_mask_warm_Q=len(warm) if cold else 0,
                   mixed_mask_warm_admits=len(admit&warm) if cold else 0,
                   edits_with_cold=len(edited) if cold else 0,
                   edits_with_warm_only=len(edited) if warm and not cold else 0,
                   edits_without_masks=len(edited) if not masked else 0)
    failed=r.get("status","success")=="generation_failed"
    check((not owed or failed) and (cnt["F"],cnt["Q"])==(r["generation_forwards"],r["generation_query_rows"])
          and cnt["cold_admits"]+cnt["warm_admits"]==cnt["N_admitted"]
          and cnt["edits_with_cold"]+cnt["edits_with_warm_only"]+cnt["edits_without_masks"]==cnt["edits"],
          "incomplete or unreconciled trace")
    cnt.update(requests=1,generation_failed=int(failed),outstanding_on_failure=len(owed))
    return cnt


def audit(root,manifest,tiers=TIERS):
    tasks=manifest["samples"]
    check([t["index"] for t in tasks]==list(range(90)),"incomplete manifest")
    blob=(root/"fixed-family-reference-case-ledger-v1.json").read_bytes()
    seal=sha256(blob).hexdigest()
    check(seal=="ff9caccbf776416f0237ddcc73af774fbe567c57929b359583f70fba128d64ce","unexpected reference ledger")
    ledger=json.loads(blob)
    check(ledger["samples"]==manifest["samples_sha256"],"wrong reference tasks")
    out={}
    for q in tiers:
        by_task=[]
        dig=sha256()
        for i,t in enumerate(tasks):
            blob=(root/RUN_DIRECTORIES[q]/f"{i:02d}-optcap_ibs_{q}.json").read_bytes()
            dig.update(sha256(blob).digest())
            r=json.loads(blob)
            check(r["policy"]==f"optcap_ibs_{q}" and r["generation_cap"]==2048 and
                  r["status"] in ("success","generation_failed") and
                  r["generation_physical_padded_rows"]==r["generation_query_rows"] and
                  all(r[k]==t[k] for k in ("dataset","seed","task_id","index")),f"wrong control or task {q}/{i}")
            c=inspect(r,q)
            ref=ledger["arms"][f"llada21:optcap_ibs_{q}"]["items"][i]
            check((r["output_tokens"],c["F"],c["Q"])==(ref["N"],ref["F"],ref["Q"]) and
                  all(ref[k]==t[k] for k in ("index","task_id","seed","dataset")) and
                  (not c["generation_failed"] or not ref["correct"]),"reference bill or failure mismatch")
            by_task.append(c)
        out[q]=dict(rows_sha256=dig.hexdigest(),directory=RUN_DIRECTORIES[q],cohorts={name:{
                    k:sum(c[k] for c in group) for k in sorted(by_task[0])} for name,group in
                    (("original30",by_task[:30]),("new60",by_task[30:]),("frozen90",by_task))})
    return dict(protocol="llada-capped-ibs-paid-ticks-v1",samples=manifest["samples_sha256"],reference=seal,tiers=out)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root",type=Path)
    ap.add_argument("manifest",type=Path)
    a=ap.parse_args()
    print(json.dumps(audit(a.root,json.loads(a.manifest.read_bytes())),indent=2,sort_keys=True))
