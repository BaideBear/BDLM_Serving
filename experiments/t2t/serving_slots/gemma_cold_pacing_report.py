"""Compact per-question/width and per-block bills. No model, prompts or token IDs."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from .gemma_cold_pacing_audit import TIERS, require
from .gemma_cold_pacing_eval import checked_pair, receipt
from .gemma_rejected_noise_report import interval


def write_table(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w",newline="") as f:
        w = csv.DictWriter(f,rows[0],lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def gather(root, history, cohorts, out=None):
    paired, blocks, summary = [], [], []
    cache, registration, identities = {}, None, {}
    for name, ids in cohorts:
        for c in TIERS:
            pairs = []
            missing = []
            for i in ids:
                if not all(receipt(root,c,i,v).is_file() for v in (0,1)):
                    missing.append(i)
                    continue
                if (c,i) not in cache:
                    a,b = checked_pair(root,history,c,i)
                    sig = {k:a[k] for k in ("source","samples_sha256","scorer_sha256","gpu","layers",
                                             "projection_modules","checkpoint","torch","transformers")}
                    if registration is None:
                        registration = sig
                    require(registration == sig,"Mixed source registration")
                    identity = tuple(a[k] for k in ("task_id","dataset","seed","input_sha256"))
                    require(identities.setdefault(i,identity)==identity,"Changed frozen task across tiers")
                    cache[c,i] = a,b
                    row = dict(budget=c,index=i,dataset=a["dataset"],seed=a["seed"],
                               same_ids=a["output_digest"] == b["output_digest"],history_exact=1)
                    for arm,r in (("off",a),("on",b)):
                        row.update({arm+"_"+k:r[k] for k in ("correct","N","F","decoder_F","decoder_Q",
                                                               "encoder_F","encoder_Q","max_decoder_Q")})
                        row.update({arm+"_"+k:v for k,v in r["phase"].items()})
                        row[arm+"_TPF"] = round(r["N"]/r["F"],8)
                        row[arm+"_blocks"] = len(r["blocks"])
                        blocks.extend(dict(budget=c,index=i,arm=arm,**x) for x in r["blocks"])
                    paired.append(row)
                pairs.append(cache[c,i])
            require(not out or not missing, f"C{c}/{name} missing pairs: {missing}")
            s = dict(cohort=name,budget=c,expected=len(ids),matched=len(pairs),missing=len(missing))
            if pairs:
                for j,arm in enumerate(("off","on")):
                    rs = [p[j] for p in pairs]
                    s.update({arm+"_"+k:sum(r[k] for r in rs) for k in
                              ("correct","F","N","decoder_F","decoder_Q","encoder_F","encoder_Q")})
                    s.update({arm+"_"+k:sum(r["phase"][k] for r in rs) for k in rs[0]["phase"]})
                    s[arm+"_blocks"] = sum(len(r["blocks"]) for r in rs)
                    s[arm+"_macro_TPF"] = round(mean(r["N"]/r["F"] for r in rs),6)
                tpf = lambda r:r["N"]/r["F"]
                s["delta_macro_TPF"] = round(mean(tpf(b)-tpf(a) for a,b in pairs),6)
                for label,metric in (("TPF",tpf),("mean_F",lambda r:r["F"]),("quality",lambda r:int(r["correct"]))):
                    ci = interval(pairs,metric) or (None,None)
                    s[label+"_lo95"],s[label+"_hi95"] = ci
                s["rescues"] = sum(b["correct"] and not a["correct"] for a,b in pairs)
                s["losses"] = sum(a["correct"] and not b["correct"] for a,b in pairs)
                s["ids_changed"] = sum(a["output_digest"]!=b["output_digest"] for a,b in pairs)
                if out:
                    require(Counter(a["dataset"] for a,b in pairs) ==
                            dict.fromkeys(("gsm8k","math500","humaneval"),len(ids)//3),"Unbalanced cohort")
            summary.append(s)
    if out:
        require(len(cache)==270 and len(set(identities.values()))==90,"Not frozen90")
        qualification = json.loads((root/"qualification.json").read_text())
        require(qualification["source"]==registration["source"] and qualification["pairs"]==9,
                "Missing qualification")
        out.mkdir(parents=True,exist_ok=True)
        write_table(out/"pairs.csv",paired)
        write_table(out/"blocks.csv",blocks)
        write_table(out/"cohorts.csv",summary)
        tmp = out/"provenance.tmp"
        tmp.write_text(json.dumps(dict(**registration,root=str(root),history=str(history),pairs=len(cache),
                                      qualification=qualification["indices"]),indent=2,sort_keys=True)+"\n")
        tmp.replace(out/"provenance.json")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--history",type=Path,required=True)
    ap.add_argument("--out",type=Path)
    a = ap.parse_args()
    cohorts = [("frozen90",range(90))]
    if a.out:
        cohorts += [("original30",range(30)),("new60",range(30,90))]
    print(json.dumps(gather(a.root,a.history,cohorts,a.out)))


if __name__=="__main__":
    main()
