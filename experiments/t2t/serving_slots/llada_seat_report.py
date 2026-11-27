"""Report only a complete registered frontier/interior cohort; no model required."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean


def report(root,manifest):
    meta=json.loads((root/"run.json").read_bytes())
    arms={"llada-frontier-one-seat-v1":("old","exchange"),"llada-interior-one-seat-v1":("old","read","edit"),
          "llada-fixed-width-window-v1":("old","window")}[meta["protocol"]]
    q,order=meta["q"],meta["order"]
    n=len(order)
    if (n not in (30,90) or order[:3]!=[0,10,20] or sorted(order)!=list(range(n))
            or q not in ((8,16) if "window" in arms else (4,8))
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
                or not r["first"] and i in order[:3]):
            raise ValueError(f"Unmatched input or unexercised anchor: {i}")
        if r["first"] and not 0<r["first"]["width"]<=q:
            raise ValueError("Wrong first exchange width")
        if "edit" in arms and bool(r["edit_first"])!=bool(r["first"]):
            raise ValueError("Unpaired read/edit opportunity")
        for k in arms:
            v=r[k]
            if (not 0<v["F"]<=v["Q"]<=q*v["F"] or not 0<v["max_decoder_Q"]<=q or v["N"]<0
                    or type(v["correct"]) is not bool or (v["failure"] and v["correct"])
                    or len(v["output_digest"])!=64 or (k=="read" and v["interior_writes"])):
                raise ValueError(f"Invalid physical receipt {i} {k}")
        rows.append(r)
    out={}
    for name,part in (("original30",rows[:30]),("new60",rows[30:]),("frozen90",rows if n==90 else [])):
        if not part:
            continue
        e=out[name]=dict(pairs=len(part),exposed=sum(bool(r["first"]) for r in part),arms={},changes={})
        for k in arms:
            a=[r[k] for r in part]
            counts=(("actions","swapped_admits") if "exchange" in arms else
                    () if "window" in arms else ("interior_swaps","interior_writes"))
            s={col:sum(v[col] for v in a) for col in ("N","F","Q","correct",*counts)}
            if "window" in arms:
                s.update({col:sum(v.get(col,0) for v in a) for col in ("window_calls","window_interior","window_edits")})
            s.update(failed=sum(bool(v["failure"]) for v in a),
                     macro_TPF=mean(v["N"]/v["F"] for v in a),TE=s["N"]/s["Q"])
            e["arms"][k]=s
        for left,right in zip(arms,arms[1:]):
            e["changes"][f"{left}_to_{right}"]=dict(
                rescues=sum(not r[left]["correct"] and r[right]["correct"] for r in part),
                losses=sum(r[left]["correct"] and not r[right]["correct"] for r in part),
                different_ids=sum(r[left]["output_digest"]!=r[right]["output_digest"] for r in part))
    return dict(registration=meta,rows_sha256=sha.hexdigest(),cohorts=out)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root",type=Path)
    ap.add_argument("manifest",type=Path)
    a=ap.parse_args()
    print(json.dumps(report(a.root,json.loads(a.manifest.read_bytes())),indent=2,sort_keys=True))
