"""Read-only fixed-family report; never pool different devices, laws or item sets."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import fmean
import sys

from bdlm_serving.evaluation.frozen import sha
from bdlm_serving.evaluation.frozen90 import load_tasks
from .llada_all_slots90 import DATA, require

MODELS = {"llada21": ("llada21-fixed-family-v1",(4,8,16,32,64),20),
          "gemma": ("gemma-fixed-family-v1",(32,64,128,256,512),85)}
COHORTS = {"original30":set(range(30)),"added60":set(range(30,90)),"full90":set(range(90))}


def measure(rows):
    n, f, q, dec = (sum(x[k] for x in rows) for k in ("N","F","Q","decoder_Q"))
    df = sum(x["decoder_F"] for x in rows)
    visits = sum((Counter(x["row_visits"]) for x in rows),Counter())
    return dict(requests=len(rows), correct=sum(x["correct"] for x in rows),
                quality=fmean(x["correct"] for x in rows),macro_TPF=fmean(x["N"]/x["F"] for x in rows),
                efficiency=n/q,N=n,F=f,Q=q,decoder_F=df,decoder_Q=dec,encoder_Q=q-dec,
                mean_decoder_Q=dec/df,max_decoder_Q=max(x["max_decoder_Q"] for x in rows),
                dirty_fraction=visits["dirty"]/dec,prefix_fraction=visits["prefix"]/dec,
                prefix_edits=visits["prefix_edits"],prefix_visits=visits["prefix"],
                failures=sum(bool(x.get("failure")) for x in rows))


def report(root, model, device, num=1, *, include_items=False):
    name, menu, layers = MODELS[model]
    suite = root/"samples.json"
    tasks=load_tasks(suite)
    folder=root/name/device/f"law-{num}-8"
    folders=[folder]
    if model=="gemma" and device=="cuda":
        # The Q512 successor has a separate output root so it never modifies
        # the first 360-case GPU seal. CPU registrations share a root.
        folders.append(root/"gemma-fixed-dual-v1"/device/f"law-{num}-8")
    files=[folder/"run.json"] if model=="llada21" and (folder/"run.json").exists() else [
        p for base in folders for p in base.glob("shard-*-of-*-tiers-*.json")]
    regs, assigned = {}, set()
    for p in files:
        reg=json.loads(p.read_bytes())
        require(reg["samples"]==sha(suite) and reg["device"]==device and
                set(reg["indices"])<=set(range(90)) and set(reg["tiers"])<=set(menu),"Wrong registration")
        cells={(i,q) for i in reg["indices"] for q in reg["tiers"]}
        require(len(cells)==len(reg["indices"])*len(reg["tiers"]) and not cells&assigned,"Duplicate assignment")
        assigned.update(cells)
        regs[sha(p)]=cells
    cases={q:{} for q in menu}
    inputs={}
    for p in sorted(p for base in folders for p in base.glob("[0-9][0-9]-q*.json")):
        r=json.loads(p.read_bytes())
        i,q=r["index"],r["q"]
        require((i,q) in regs.get(r["registration_sha256"],set()) and i not in cases[q] and
                p.name==f"{i:02}-q{q}.json" and
                r["law"]==[num,8] and all(tasks[i][k]==r[k] for k in ("seed","dataset","task_id")) and
                inputs.setdefault(i,r["input_sha256"])==r["input_sha256"],"Wrong item, seed, law or input")
        widths=[e["q"] for e in r["trace"]]
        enc=sum(bool(e.get("encoded")) for e in r["trace"])
        coords = "positions" if model=="gemma" and q==512 else "rows"
        require(widths and all(e["offer"]==q and 0<len(set(e[coords]))==e["q"]<=q
                              for e in r["trace"]) and len(widths)+enc==r["F"] and
                sum(widths)==r["decoder_Q"] and enc*256==r["encoder_Q"] and
                sum(widths)+enc*256==r["Q"] and max(widths)==r["max_decoder_Q"] and
                r["projection_modules"]==layers and type(r["correct"]) is bool and
                r["correct"] is bool(r["score"]["correct"]) and 0<=r["N"]<=len(r["output_ids"])<=2048
                and r["TPF"]==r["N"]/r["F"] and r["efficiency"]==r["N"]/r["Q"],"Unverified work")
        cases[q][i]={k:r[k] for k in ("index","seed","dataset","task_id","correct","N","F","Q",
                                     "decoder_Q","max_decoder_Q","row_visits","input_sha256")}
        cases[q][i].update(decoder_F=len(widths),failure=bool(r.get("failure")))
    coverage={q:dict(assigned=sum(t==q for _,t in assigned),sealed=len(cases[q])) for q in menu}
    tables={}
    for cohort,idx in COHORTS.items():
        tables[cohort]={q:measure([cases[q][i] for i in sorted(idx)]) for q in menu if idx<=cases[q].keys()}
    tiers=[q for q in menu if coverage[q]["assigned"]]
    common=set.intersection(*(set(cases[q]) for q in tiers)) if tiers else set()
    preview={}
    # A descriptive check on identical cases, never a replacement for any missing
    # formal cell. Empty if all assigned arms are complete.
    if common and len(common)<90:
        preview=dict(indices=sorted(common),metrics={q:measure([cases[q][i] for i in sorted(common)]) for q in tiers})
    pilots={0,10,20}
    smoke={q:measure([cases[q][i] for i in sorted(pilots)]) for q in tiers} if pilots<=common else {}
    return dict(model=model,device=device,law=[num,8],coverage=coverage,cohorts=tables,
                **({"items":cases} if include_items else {}),
                protocol_smoke3=smoke,
                incomplete_matched_preview=preview)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model",choices=MODELS)
    ap.add_argument("--device",choices=("cpu","cuda"),required=True)
    ap.add_argument("--num",type=int,choices=(0,1),default=1)
    output=ap.add_mutually_exclusive_group()
    output.add_argument("--smoke-csv",action="store_true",help="Only emit the complete, predeclared three-case check")
    output.add_argument("--case-index",type=int,choices=range(90),metavar="0..89",
                        help="Diagnostic only: one identical request at every target tier")
    a=ap.parse_args()
    outcome=report(DATA,a.model,a.device,a.num,include_items=a.case_index is not None)
    if not a.smoke_csv and a.case_index is None:
        print(json.dumps(outcome,indent=2,allow_nan=False))
    else:
        if a.case_index is None:
            rows=[dict(model=a.model,device=a.device,budget=q,law=f"{a.num}/8",indices="0,10,20",**v)
                  for q,v in outcome["protocol_smoke3"].items()]
            require(rows,"Not all registered tiers completed the three cases")
        else:
            items=outcome["items"]
            require(all(a.case_index in cells for cells in items.values()),"Missing requested item at some target tier")
            rows=[dict(model=a.model,device=a.device,budget=q,law=f"{a.num}/8",index=a.case_index,
                       **measure([cells[a.case_index]])) for q,cells in items.items()]
        w=csv.DictWriter(sys.stdout,fieldnames=list(rows[0]),lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
