"""Recheck paired GPU receipts on cluster. Export only scalar counts and hashes."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from .gemma_verified_eos_eval import compare, digest, history, GPU, check


def theory(ids, eos, q):
    end=next((i+1 for i,x in enumerate(ids) if x in eos),None)
    if end is None:
        check(len(ids)==2048,"No EOS before length cap")
        return 0
    check(end==len(ids),"Reference delivered past first EOS")
    return (255-(end-1)%256)//q


def summarize(rows):
    n=len(rows)
    out=dict(pairs=n,exposed=sum(x["saved_F"]>0 for x in rows),
             eos_last_chunk=sum(x["eos_row"] is not None and x["saved_F"]==0 for x in rows),
             no_eos=sum(x["eos_row"] is None for x in rows),
             correct=sum(x["correct"] for x in rows),matched_ids=n,matched_scores=n,
             matched_encoder=n,full_nonterminal_blocks=sum(x["encoder_F"] for x in rows))
    for k in ("N","off_F","on_F","off_decoder_F","on_decoder_F","off_decoder_Q","on_decoder_Q",
              "off_total_Q","on_total_Q","saved_F","saved_Q","theoretical_F","theoretical_Q","encoder_F","encoder_Q"):
        out[k]=sum(x[k] for x in rows)
    for arm in ("off","on"):
        out[arm+"_macro_TPF"]=round(mean(x["N"]/x[arm+"_F"] for x in rows),6)
        out[arm+"_micro_TPF"]=round(out["N"]/out[arm+"_F"],6)
    return out


def collect(root,hist,manifest,eos,tiers):
    by_id={x["index"]:x for x in manifest["samples"]}
    pairs,cohorts,files,source=[],[],{},None
    for q in tiers:
        rows=[]
        for i in range(90):
            p=root/f"C{q}-pair-{i:02}.json"
            blob=p.read_bytes()
            r=json.loads(blob)
            hist_blob=history(hist,q,i).read_bytes()
            h=json.loads(hist_blob)
            check(r["schema"]=="verified-eos-pair-v1" and r["gpu"]==GPU and r["q"]==q and
                  r["reference_sha256"]==sha256(hist_blob).hexdigest() and
                  r["samples_sha256"]==h["samples_sha256"]==manifest["samples_sha256"] and
                  all(r[k]==h[k]==by_id[i][k] for k in ("index","task_id","dataset","seed")) and
                  r["scorer_sha256"]==h["scorer_sha256"] and (source is None or source==r["source"]),
                  "Bad or mixed registration")
            source=r["source"]
            files[p.name]=sha256(blob).hexdigest()
            off,on=r["off"],r["on"]
            compare(off,on,h,q)
            for x in (off,on):
                check(type(x["correct"]) is bool and x["encoder_Q"]==256*x["encoder_F"]
                      and x["decoder_F"]==len(x["tape"])==sum(x["route_F"].values())
                      and x["decoder_Q"]==sum(x["route_Q"].values()) and x["F"]==x["decoder_F"]+x["encoder_F"],
                      "Unreconciled bill")
            f=off["F"]-on["F"]
            hit=off["hit"] or {}
            check(f==theory(h["output_ids"],eos,q) and
                  (not hit or hit["ordinal"]==off["encoder_F"] and hit["eos_row"]==(len(h["output_ids"])-1)%256)
                  and bool(hit)==(h["output_ids"][-1] in eos),"Historical EOS theory differs from measured EOS")
            rows.append(dict(q=q,index=i,dataset=r["dataset"],task_id=r["task_id"],seed=r["seed"],
                correct=int(off["correct"]),N=off["N"],ids_sha256=digest(off["output_ids"]),
                output_length=len(off["output_ids"]),encoder_F=off["encoder_F"],encoder_Q=off["encoder_Q"],
                off_F=off["F"],on_F=on["F"],off_decoder_F=off["decoder_F"],on_decoder_F=on["decoder_F"],
                off_decoder_Q=off["decoder_Q"],on_decoder_Q=on["decoder_Q"],
                off_total_Q=off["decoder_Q"]+off["encoder_Q"],on_total_Q=on["decoder_Q"]+on["encoder_Q"],
                eos_row=hit.get("eos_row"),chunk_end=hit.get("chunk_end"),eos_tick=hit.get("tick"),
                theoretical_F=theory(h["output_ids"],eos,q),theoretical_Q=theory(h["output_ids"],eos,q)*q,
                saved_F=f,saved_Q=off["decoder_Q"]-on["decoder_Q"]))
        pairs.extend(rows)
        for key,select in (("original30",rows[:30]),("new60",rows[30:]),("frozen90",rows)):
            cohorts.append(dict(q=q,cohort=key,**summarize(select)))
    return pairs,cohorts,dict(schema="verified-eos-compact-v1",gpu=GPU,source=source,
                  samples_sha256=manifest["samples_sha256"],pair_files_sha256=digest(sorted(files.items())),
                  pairs=len(files),raw=str(root),history=str(hist),eos=sorted(eos))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ("root","history","manifest","generation-config","out"):
        ap.add_argument("--"+arg,type=Path,required=True)
    ap.add_argument("--tiers",type=int,nargs="+",choices=(32,64,128),default=(32,64,128))
    a=ap.parse_args()
    ids=json.loads(a.generation_config.read_bytes())["eos_token_id"]
    check(len(a.tiers)==len(set(a.tiers)),"Duplicate tier")
    pairs,cohorts,prov=collect(a.root,a.history,json.loads(a.manifest.read_bytes()),set(ids),a.tiers)
    a.out.mkdir(parents=True,exist_ok=True)
    for name,rows in (("pairs",pairs),("cohorts",cohorts)):
        with (a.out/(name+".csv")).open("w",newline="") as f:
            writer=csv.DictWriter(f,list(rows[0]),lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    (a.out/"provenance.json").write_text(json.dumps(prov,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    main()
