"""Export a compact only for the registered zero-exposure Q32 qualification."""
import csv
import json
from pathlib import Path

from .gemma_cold_frontier_last_eval import GPU,read,sha
from .gemma_joint_verify_eval import ORDER,check,digest
from .gemma_warm_fewer_fail import align
from .gemma_warm_fewer_fail_eval import HIST,ROOT,RT


def main():
    reg=read(ROOT/"registration.json")
    anchor=ORDER[:3]
    h=digest(reg)
    check(anchor==[0,10,20] and reg["gpu"]==GPU and reg["policy"]==sha(RT/"registration.md")
          and reg["launcher"]==sha(RT/"launch.sh") and all(
              v==sha(Path(__file__).with_name(k+".py")) for k,v in reg["sources"].items()),
          "Qualification source drift")
    check(read(ROOT/"checkpoint-3.json")==dict(pairs=3,exposed=0,reg=h) and
          read(ROOT/"no-go.json")==dict(pairs=3,reason="zero anchor exposure",reg=h) and
          set(p.name for p in ROOT.glob("pair-*.json"))=={f"pair-{i:02}.json" for i in anchor}
          and not (ROOT/"complete.json").exists(),"Expected only a three-pair no-go")
    check(reg["refs"]==digest([sha(HIST/f"C32-pair-{i:02}.json") for i in range(90)]),"Reference changed")
    compact=RT/"compact"
    compact.mkdir(exist_ok=True)
    rows=[]
    for i in anchor:
        p=ROOT/f"pair-{i:02}.json"
        refp=HIST/f"C32-pair-{i:02}.json"
        r,ref=read(p),read(refp)
        off,on=r["off"],r["on"]
        a,b=r["events"]["off"],r["events"]["on"]
        check(r["reg"]==h and r["pid"]==reg["pid"] and r["ref"]==sha(refp) and
              r["index"]==i and all(r[k]==ref[k] for k in ("index","seed","dataset","task_id")) and
              ref["source"]==reg["base"] and ref["scorer_sha256"]==reg["scorer"] and
              ref["samples_sha256"]==reg["inputs"] and ref["q"]==32 and
              off==ref["on"]==on and align(off,on,a,b) is None and r["first"] is None and
              not any(v["swap"] for v in a+b),"Inexact anchor")
        rows.append(dict(index=i,task_id=r["task_id"],seed=r["seed"],pid=r["pid"],
                    replayed=True,identical_full_receipts=True,identical_event_notes=(a==b),
                    warm=sum(v["warm"] for v in a),contested=sum(v["contest"] for v in a),
                    swaps=0,N=on["N"],F=on["F"],decoder_F=on["decoder_F"],decoder_Q=on["decoder_Q"],
                    encoder_F=on["encoder_F"],encoder_Q=on["encoder_Q"],correct=on["correct"],
                    ref_sha256=r["ref"],pair_sha256=sha(p)))
    with (compact/"anchors.csv").open("w",newline="") as fp:
        w=csv.DictWriter(fp,rows[0],lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (compact/"provenance.json").write_text(json.dumps(dict(registration=reg,report_sha256=sha(Path(__file__)),
             decision=read(ROOT/"no-go.json"),raw_root=str(ROOT),pairs=3,not_run=87),
             indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    main()
