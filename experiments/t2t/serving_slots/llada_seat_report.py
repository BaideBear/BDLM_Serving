"""Report only a complete registered frontier/interior cohort; no model required."""
import argparse
from hashlib import sha256
import json
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from statistics import mean


def paired(part,left,right):
    groups=defaultdict(list)
    for r in part:
        a,b=r[left],r[right]
        groups[r["dataset"]].append((b["correct"]-a["correct"],b["N"]/b["F"]-a["N"]/a["F"]))
    if len(groups)!=3 or min(map(len,groups.values()))<2:
        raise ValueError("Expected paired data from all three datasets")
    rng=Random(1128); quality=[]; tpf=[]
    for _ in range(4000):
        chosen=[x for g in groups.values() for x in rng.choices(g,k=len(g))]
        quality.append(sum(x[0] for x in chosen)/len(part))
        tpf.append(sum(x[1] for x in chosen)/len(part))
    def ci(v):
        v.sort()
        return [round(v[99],5),round(v[3899],5)]
    return dict(correct_rate=ci(quality),macro_TPF=ci(tpf))


def window_rows(arm):
    k=arm["row_visits"]
    s=Counter(dirty=k["dirty"],cold_mask=k["cold"],waiting_mask=k["clean_unresolved"],
              clean_prefix=k["prefix"],clean_interior=arm.get("window_interior",0))
    if min(s.values())<0 or sum(s.values())!=arm["Q"]:
        raise ValueError("Nonexclusive or unclassified head-only projection rows")
    return s


def read_ibs(path):
    blob=path.read_bytes()
    if sha256(blob).hexdigest()!="7c7d94d0a090c234b18322c339361b3b0d0d76a5ba83f03083d8659a6805ffde":
        raise ValueError("Strict IBS rows do not match the sealed compact export")
    return json.loads(blob)


def reference(root):
    """Export only identities, bills, scores and hashes. Never export model outputs."""
    ledger=(root/"fixed-family-reference-case-ledger-v1.json").read_bytes()
    digest=sha256(ledger).hexdigest()
    if digest!="ff9caccbf776416f0237ddcc73af774fbe567c57929b359583f70fba128d64ce":
        raise ValueError("Not the independently rescored strict reference ledger")
    data=json.loads(ledger)
    records={}
    for q,group in ((4,"low"),(8,"endpoints"),(16,"endpoints")):
        name=f"optcap_ibs_{q}"
        items=data["arms"]["llada21:"+name]["items"]
        if len(items)!=90: raise ValueError("Incomplete IBS reference")
        checked=[]
        for i,item in enumerate(items):
            p=root/("llada21-optimus-hardcap-"+group)/f"{i:02}-{name}.json"
            r=json.loads(p.read_bytes())
            if (r["policy"]!=name or item["index"]!=i or
                    any(r[k]!=item[k] for k in ("index","task_id","dataset","seed"))
                    or (r["output_tokens"],r["generation_forwards"],r["generation_query_rows"])!=
                       (item["N"],item["F"],item["Q"])):
                raise ValueError("Mismatched raw IBS bills or cases")
            prompt_digest=sha256(json.dumps(r["prompt_ids"],separators=(",",":")).encode()).hexdigest()
            checked.append(dict(item,input=prompt_digest))
        records[str(q)]=checked
    return dict(ledger=digest,samples=data["samples"],scorer=data["scorer"],arms=records)


def report(root,manifest,ibs=None):
    meta=json.loads((root/"run.json").read_bytes())
    arms={"llada-frontier-one-seat-v1":("old","exchange"),"llada-interior-one-seat-v1":("old","read","edit"),
          "llada-fixed-width-window-v1":("old","window"),"llada-warm-successor-q64-v1":("old","read","edit"),
          "llada-window-paid-mask-v1":("old","paid"),"llada-window-paid-interior-v1":("old","mixed_edit")}[meta["protocol"]]
    mixed="mixed_edit" in arms
    successor=meta["protocol"]=="llada-warm-successor-q64-v1"
    q,order=meta["q"],meta["order"]
    n=len(order)
    if (n not in (30,90) or order[:3]!=[0,10,20] or sorted(order)!=list(range(n))
            or q not in ((16,) if mixed else (64,) if successor else (4,8,16) if "window" in arms or "paid" in arms else (4,8))
            or meta["samples"]!=manifest["samples_sha256"] or not meta["device"].startswith("GPU-")):
        raise ValueError("Unregistered study")
    files=sorted((root/"rows").glob("*.json"))
    if {p.name for p in files}!={f"{i:02}.json" for i in order}:
        raise ValueError("Study is not complete")
    rows=[]
    sha=sha256()
    for i,p in enumerate(files):
        blob=p.read_bytes()
        sha.update(sha256(blob).digest())
        r=json.loads(blob)
        if (not all(r[k]==manifest["samples"][i][k] for k in ("index","seed","dataset","task_id"))
                or not mixed and "paid" not in arms and not r["first"] and i in order[:3]):
            raise ValueError(f"Unmatched input or unexercised anchor: {i}")
        if mixed:
            v=r["effects"]
            if (n!=90 or r.get("replayed_paid") is not True or bool(v["seats"])!=bool(r["first"])
                    or not 0<=v["T2T"]+v["revoked"]<=v["seats"] or not 0<=v["calls"]<=v["seats"]):
                raise ValueError("Unverified mixed-edit receipt")
        if r["first"] and not 0<r["first"]["width"]<=q:
            raise ValueError("Wrong first exchange width")
        if "paid" in arms and q==4 and not (n==90 and len(meta["paired_window_rows"])==64
                                           and r.get("replayed_window") is True):
            raise ValueError("Missing paired Q4 window replay")
        if "edit" in arms and bool(r["edit_first"])!=bool(r["first"]):
            raise ValueError("Unpaired read/edit opportunity")
        for k in arms:
            v=r[k]
            if (not 0<v["F"]<=v["Q"]<=q*v["F"] or not 0<v["max_decoder_Q"]<=q or v["N"]<0
                    or type(v["correct"]) is not bool or (v["failure"] and v["correct"])
                    or len(v["output_digest"])!=64 or (k=="read" and
                    (v.get("supp_T2T",0)+v.get("supp_revoke",0) if successor else v["interior_writes"]))):
                raise ValueError(f"Invalid physical receipt {i} {k}")
        rows.append(r)
    if "paid" in arms and not any(rows[i]["first"] for i in order[:3]):
        raise ValueError("Unexercised paid MASK pilot")
    if mixed and (meta.get("paired_paid_rows")!="16ad875482119522eb7670cc20e2e1b2c5eec4ab10687968475c335ef0a9bc40"
                  or not any(rows[i]["effects"]["T2T"]+rows[i]["effects"]["revoked"] for i in order[:3])):
        raise ValueError("Unsealed paid parent or no real new write at pilot")
    if ibs is not None:
        if not ("window" in arms and meta["samples"]==ibs["samples"] and meta["scorer"]==ibs["scorer"]):
            raise ValueError("Reference cannot be paired")
        for r in rows:
            v=ibs["arms"][str(q)][r["index"]]
            if any(r[k]!=v[k] for k in ("index","dataset","task_id","seed","input")):
                raise ValueError("Different inputs on the external reference")
            r["strict_ibs"]=v
    out={}
    for name,part in (("original30",rows[:30]),("new60",rows[30:]),("frozen90",rows if n==90 else [])):
        if not part:
            continue
        e=out[name]=dict(pairs=len(part),exposed=sum(bool(r["first"]) for r in part),arms={},changes={})
        for k in arms:
            a=[r[k] for r in part]
            counts=(("actions","swapped_admits") if "exchange" in arms else
                    () if "window" in arms or "paid" in arms or successor or mixed else ("interior_swaps","interior_writes"))
            s={col:sum(v[col] for v in a) for col in ("N","F","Q","correct",*counts)}
            if "window" in arms or "paid" in arms:
                s.update({col:sum(v.get(col,0) for v in a) for col in ("window_calls","window_interior","window_edits")})
                s["actual_rows"]=dict(sum((window_rows(v) for v in a),Counter()))
            if "paid" in arms:
                s.update({col:sum(v.get(col,0) for v in a) for col in
                          ("paid_waiting_calls","paid_waiting_rows","paid_waiting_admits")})
            if successor:
                s.update({col:sum(v.get(col,0) for v in a) for col in ("supp_calls","supp_Q","supp_T2T","supp_revoke")})
            s.update(failed=sum(bool(v["failure"]) for v in a),
                     macro_TPF=mean(v["N"]/v["F"] for v in a),TE=s["N"]/s["Q"])
            e["arms"][k]=s
        if mixed:
            e["effects"]={k:sum(r["effects"][k] for r in part) for k in ("calls","seats","T2T","revoked")}
        for left,right in zip(arms,arms[1:]):
            e["changes"][f"{left}_to_{right}"]=dict(
                rescues=sum(not r[left]["correct"] and r[right]["correct"] for r in part),
                losses=sum(r[left]["correct"] and not r[right]["correct"] for r in part),
                different_ids=sum(r[left]["output_digest"]!=r[right]["output_digest"] for r in part),
                paired_stratified_ci95=paired(part,left,right))
        if ibs is not None:
            a,b="strict_ibs","window"
            e["window_minus_external_capped_IBS"]=dict(
                rescues=sum(not r[a]["correct"] and r[b]["correct"] for r in part),
                losses=sum(r[a]["correct"] and not r[b]["correct"] for r in part),
                delta_macro_TPF=mean(r[b]["N"]/r[b]["F"]-r[a]["N"]/r[a]["F"] for r in part),
                delta_TE=e["arms"][b]["TE"]-sum(r[a]["N"] for r in part)/sum(r[a]["Q"] for r in part),
                paired_stratified_ci95=paired(part,a,b),ledger_sha256=ibs["ledger"])
    return dict(registration=meta,rows_sha256=sha.hexdigest(),cohorts=out)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root",type=Path)
    ap.add_argument("manifest",type=Path,nargs="?")
    ap.add_argument("--export-reference",action="store_true")
    ap.add_argument("--ibs",type=Path)
    a=ap.parse_args()
    if a.export_reference:
        result=reference(a.root)
    else:
        result=report(a.root,json.loads(a.manifest.read_bytes()),read_ibs(a.ibs) if a.ibs else None)
    print(json.dumps(result,indent=2,sort_keys=True))
