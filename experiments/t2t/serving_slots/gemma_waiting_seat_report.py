"""Only export scalar/hash evidence from a completed or predeclared no-go run."""
import argparse
import csv
import json
from hashlib import sha256
from pathlib import Path
from statistics import mean

from .gemma_joint_verify_eval import GPU, ORDER, check, digest
from .gemma_rejected_noise_report import interval
from .gemma_waiting_seat import alignment, gates

ROUTES=("sparse","drain","verify_paid","verify")
ROUTE_BILLS=tuple(f"{route}_{unit}" for route in ROUTES for unit in ("F","Q"))


def load(path):
    return json.loads(path.read_bytes())


def summarize(rows):
    out=dict(pairs=len(rows),exposed=sum(r["first_tick"] is not None for r in rows),
             same_ids=sum(r["same_ids"] for r in rows),
             lower_F=sum(r["on_F"]<r["off_F"] for r in rows),
             higher_F=sum(r["on_F"]>r["off_F"] for r in rows),
             rescues=sum(r["on_correct"]>r["off_correct"] for r in rows),
             losses=sum(r["off_correct"]>r["on_correct"] for r in rows))
    for a in ("off","on"):
        for k in ("correct","N","decoder_F","decoder_Q","encoder_F","encoder_Q","F","total_Q",
                  "warm","contested","opportunities",*ROUTE_BILLS):
            out[a+"_"+k]=sum(r[a+"_"+k] for r in rows)
        out[a+"_macro_TPF"]=round(mean(r[a+"_N"]/r[a+"_F"] for r in rows),6)
        out[a+"_pooled_TPF"]=round(out[a+"_N"]/out[a+"_F"],6)
    pairs=[tuple(dict(dataset=r["dataset"],tpf=r[a+"_N"]/r[a+"_F"],correct=r[a+"_correct"])
                 for a in ("off","on")) for r in rows]
    for k in ("tpf","correct"):
        ci=interval(pairs,lambda x:x[k]) or (None,None)
        out.update({f"delta_{k}_lo95":ci[0],f"delta_{k}_hi95":ci[1]})
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("root","history","manifest","out"):
        ap.add_argument("--"+k,type=Path,required=True)
    a=ap.parse_args()
    done,reg=load(a.root/"complete.json"),load(a.root/"registration.json")
    manifest=load(a.manifest)
    tasks={t["index"]:t for t in manifest["samples"]}
    expected=ORDER[:3] if done["outcome"]=="no-go" else ORDER
    files=sorted(a.root.glob("C64-pair-*.json"))
    check(reg["gpu"]==GPU and reg["samples"]==manifest["samples_sha256"] and
          done["outcome"] in ("no-go","frozen90") and len(expected)==len(files)==done["pairs"] and
          {p.name for p in files}=={f"C64-pair-{i:02}.json" for i in expected},"Missing or mixed run")
    anchors, rows, sums=[],[],[]
    for i in expected:
        p=a.root/f"C64-pair-{i:02}.json"
        blob=p.read_bytes()
        r=json.loads(blob)
        hblob=(a.history/p.name).read_bytes()
        h=json.loads(hblob)
        check(r["schema"]=="waiting-seat-pair-v1" and r["source"]==reg["source"] and r["gpu"]==GPU and
              r["reference_sha256"]==sha256(hblob).hexdigest() and h["source"]==reg["base"] and
              r["scorer_sha256"]==h["scorer_sha256"]==reg["scorer"] and r["off"]==h["on"] and
              all(r[k]==h[k]==tasks[i][k] for k in ("index","seed","task_id","dataset")),"Reference mismatch")
        first=alignment(r["off"],r["on"],r["seat_tapes"])
        check(first==r["first"],"First change mismatch")
        row={k:r[k] for k in ("index","dataset","task_id","seed")}
        row.update(first_tick=first and first["tick"],first_head=first and first["head"],
                   first_drop=first and first["swap"]["drop"],first_take=first and first["swap"]["take"],
                   first_width=first and first["swap"]["width"],
                   first_stamp_gap=first and first["swap"]["drop_stamp"]-first["swap"]["take_stamp"],
                   first_state=first and first["swap"]["state"],predecessor=first and first["predecessor_sha256"],
                   same_ids=int(r["off"]["output_ids"]==r["on"]["output_ids"]))
        for name,tape in zip(("off","on"),r["seat_tapes"]):
            x=r[name]
            check(len(tape)==len(x["tape"])==x["decoder_F"]==sum(x["route_F"].values()) and
                  sum(x["route_Q"].values())==x["decoder_Q"] and x["F"]==x["decoder_F"]+x["encoder_F"] and
                  set(x["route_F"])==set(x["route_Q"])<=set(ROUTES) and
                  x["encoder_Q"]==256*x["encoder_F"] and type(x["correct"]) is bool,"Unreconciled bill")
            for j,e in enumerate(tape):
                s=e["swap"]
                check(e["tick"]==j and (not e["contested"] or e["warm"]) and (not s or
                      (e["contested"] and s["take_stamp"]<s["drop_stamp"] and s["drop"]!=s["frontier"]
                       and s["width"]==64 and s["reference_rows"]!=s["alternative_rows"])),"Invalid exposure")
            totals=dict(N=x["N"],correct=int(x["correct"]),ids_sha256=digest(x["output_ids"]),
                        warm=sum(e["warm"] for e in tape),contested=sum(e["contested"] for e in tape),
                        opportunities=sum(e["swap"] is not None for e in tape),
                        total_Q=x["decoder_Q"]+x["encoder_Q"])
            totals.update({k:x[k] for k in ("decoder_F","decoder_Q","encoder_F","encoder_Q","F")})
            totals.update({f"{route}_{unit}":x["route_"+unit].get(route,0)
                           for route in ROUTES for unit in ("F","Q")})
            row.update({name+"_"+k:v for k,v in totals.items()})
        rows.append(row)
        sums.append([p.name,sha256(blob).hexdigest()])
        if i in (0,10,20):
            anchors.append(r)
    gate=gates(anchors)
    check(gate==done["gate"]==load(a.root/"qualification.json") and
          all(gate.values())==(done["outcome"]=="frozen90"),"Gate result differs from registration")
    rows.sort(key=lambda r:r["index"])
    selections=[("anchors3",[r for r in rows if r["index"] in (0,10,20)])]
    if len(rows)==90:
        selections.extend([("original30",rows[:30]),("new60",rows[30:]),("frozen90",rows)])
    cohorts=[dict(cohort=label,**summarize(subset)) for label,subset in selections]
    a.out.mkdir(parents=True,exist_ok=True)
    for filename,data in (("pairs.csv",rows),("cohorts.csv",cohorts)):
        with (a.out/filename).open("w",newline="") as f:
            w=csv.DictWriter(f,data[0],lineterminator="\n")
            w.writeheader()
            w.writerows(data)
    (a.out/"provenance.json").write_text(json.dumps(dict(registration=reg,completion=done,
          pair_files_sha256=digest(sorted(sums))),sort_keys=True,indent=2)+"\n")


if __name__=="__main__":
    main()
