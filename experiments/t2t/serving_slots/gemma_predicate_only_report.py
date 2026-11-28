"""Offline paired audit for the A1 Q256 native-only stopping experiment."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from .gemma_native_visibility_report import read, sha, diff
from .gemma_continuous_rng_report import digest, table
from .gemma_predicate_only import compare
from .gemma_rejected_noise_report import interval


def check(x,msg):
    if not x:
        raise ValueError(msg)


def reduce_row(r,prior,native,reg):
    e,t=r["events"],r["audit"]
    check((r["off"],t["off"])==(prior["on"],prior["audit"]["on"]) and
          all(r[k]==prior[k] for k in ("seed","index","dataset","input_sha")) and
          digest(reg)==r["registration"] and reg["pid"]==r["pid"],"Historical row mismatch")
    first=compare(e["off"],e["on"],t["off"],t["on"],r["off"],r["on"])
    check(first==r["first"],"Wrong first exposure")
    row={k:r[k] for k in ("index","seed","dataset","input_sha","reference")}
    row.update(exposed=int(first is not None),first_tick=first if first is not None else -1,
               first_block=e["off"][first]["ordinal"] if first is not None else -1,
               first_age=e["off"][first]["age"] if first is not None else -1,
               off_on_first_diff=diff(r["off"]["ids"],r["on"]["ids"]))
    if first is None:
        row["first_failed"]="unexposed"
    else:
        c=e["off"][first]["criterion"]
        stable=c["changed_argmax"]==0
        confident=c["mean_entropy"]<.005
        check(not(stable and confident),"False stop opportunity")
        row["first_failed"]="stability" if confident else "confidence" if stable else "both"
    for arm in ("off","on","native"):
        x=native if arm=="native" else r[arm]
        row.update({f"{arm}_{k}":int(x[k]) for k in ("N","F","decoder_Q","encoder_F","correct")})
        df,rem=divmod(x["decoder_Q"],256)
        check(not rem and x["F"]==df+x["encoder_F"],"Invalid physical bill")
        row[arm+"_TPF"]=x["N"]/x["F"]
        row[arm+"_verify_F"]=0 if arm=="native" else x["encoder_F"]+1
        row[arm+"_sample_F"]=df-row[arm+"_verify_F"]
        if arm=="native":
            continue
        row[arm+"_native_diff"]=diff(x["ids"],native["ids"])
        row[arm+"_native_exact"]=int(x["ids"]==native["ids"])
        row[arm+"_native_first"]=int(x["ids"][:256]==native["ids"][:256])
        tape=e[arm]
        measures=[u for u in tape if u["criterion"] is not None]
        closes=[u for u in tape if u["criterion"] is None]
        check(len(measures)==row[arm+"_sample_F"] and len(closes)==row[arm+"_verify_F"]
              and len(tape)==len(t[arm]["ticks"])==df and x["mask_layer_calls"]==30*df,
              "Incomplete measured sample/verify")
        for u in measures:
            c=u["criterion"]
            check(u["hit"]==c["hit"]==bool(c["changed_argmax"]==0 and c["mean_entropy"]<.005)
                  and (c["changed_argmax"] is None)==(u["age"]==1),"Native predicate mismatch")
        row[arm+"_quiescent_nohit_ticks"]=sum(u["accepted"]==256 and not u["hit"] and u["age"]<48 for u in measures)
        row[arm+"_native_hits"]=sum(u["hit"] for u in measures)
        shadow_diff=encode_diff=limits=0
        for j,close in enumerate(closes):
            block=[u for u in tape if u["ordinal"]==j]
            s=block[-2]
            check(block[-1]==close and close["route"] in ("verify","verify_paid") and not close["dirty"]
                  and all(u["route"]=="dense_single" and u["age"]==k for k,u in enumerate(block[:-1],1))
                  and not any(u["hit"] for u in block[:-2]) and
                  (s["hit"] or s["age"]==48 or arm=="off" and s["accepted"]==256),"Wrong terminal block/clock")
            limits+=int(not s["hit"] and s["age"]==48)
            changed=digest(x["ids"][j*256:(j+1)*256])!=s["criterion"]["draft_to_eos"]
            shadow_diff+=int(changed)
            encode_diff+=int(changed and j<len(closes)-1)
        row[arm+"_limit_only"]=limits
        row[arm+"_draft_verified_diff"]=shadow_diff
        row[arm+"_draft_verified_encoder_diff"]=encode_diff
        # At the first delivered mismatch, all earlier verified encoder inputs
        # equal native. A matching pre-verify draft isolates a delivery boundary,
        # not proof of every internal native tensor being identical.
        d=row[arm+"_native_diff"]
        j=d//256 if d>=0 else -1
        block=[u for u in measures if u["ordinal"]==j]
        row[arm+"_first_mismatch_at_verify"]=int(d>=0 and bool(block) and
                block[-1]["criterion"]["draft_to_eos"]==digest(native["ids"][256*j:256*(j+1)]))
    return row


def summary(label,rows):
    g=dict(cohort=label,n=len(rows),exposed=sum(r["exposed"] for r in rows),
           gained=sum(r["on_native_exact"]>r["off_native_exact"] for r in rows),
           lost=sum(r["on_native_exact"]<r["off_native_exact"] for r in rows),
           rescues=sum(r["on_correct"]>r["off_correct"] for r in rows),
           losses=sum(r["on_correct"]<r["off_correct"] for r in rows),
           changed_outputs=sum(r["off_on_first_diff"]!=-1 for r in rows))
    for why in ("stability","confidence","both"):
        g["first_failed_"+why]=sum(r["first_failed"]==why for r in rows)
    for arm in ("off","on","native"):
        for k in ("N","F","encoder_F","decoder_Q","sample_F","verify_F","correct"):
            g[f"{arm}_{k}"]=sum(r[f"{arm}_{k}"] for r in rows)
        g[arm+"_macro_TPF"]=mean(r[arm+"_TPF"] for r in rows)
        g[arm+"_TE"]=g[arm+"_N"]/g[arm+"_decoder_Q"]
        if arm!="native":
            for k in ("native_exact","native_first","native_hits","quiescent_nohit_ticks","limit_only",
                      "draft_verified_diff","draft_verified_encoder_diff","first_mismatch_at_verify"):
                g[f"{arm}_{k}"]=sum(r[f"{arm}_{k}"] for r in rows)
    pairs=[tuple(dict(dataset=r["dataset"],correct=r[arm+"_correct"],TPF=r[arm+"_TPF"],
                exact=r[arm+"_native_exact"]) for arm in ("off","on")) for r in rows]
    for key in ("TPF","correct","exact"):
        g[key+"_lo95"],g[key+"_hi95"]=interval(pairs,lambda x:x[key]) or (None,None)
    return g


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out",type=Path)
    ap.add_argument("--anchors",action="store_true")
    args=ap.parse_args()
    rt=Path("/gs/runtime/gemma-predicate-only-a1-1128")
    roots=[Path("/gs/results")/n for n in ("gemma-predicate-only-a1-1128","gemma-continuous-rng-a1-1128",
                                            "gemma-native-visibility-a1-1128")]
    root=roots[0]
    reg=read(root/"registration.json")
    with (rt/"validated_rng/pairs.csv").open() as f:
        refs={int(x["index"]):x["row_sha"] for x in csv.DictReader(f)}
    stamp=read(root/("checkpoint-3.json" if args.anchors else "complete.json"))
    indices=[0,10,20] if args.anchors else range(90)
    check(len(indices)==stamp["n"] and digest(reg)==stamp["registration"] and reg["pid"]==stamp["pid"] and
          digest(refs)==reg["ref"] and digest(read(roots[1]/"registration.json"))==reg["prior"] and
          (args.anchors or len(list(root.glob("pair-*.json")))==90),"Incomplete or mixed registration")
    repo=Path("/gs/runtime/eos-handle-1128/repo")
    search=[p/"experiments/t2t/serving_slots" for p in (rt,Path("/gs/runtime/gemma-continuous-rng-a1-1128"),
            Path("/gs/runtime/gemma-native-visibility-a1-1128"),repo)]+[repo/"bdlm_serving/methods/t2t"]
    for filename,v in reg["source"].items():
        check(sha(next(p/filename for p in search if (p/filename).is_file()))==v,"Source drift "+filename)
    check(sha(rt/"registration.md")==reg["policy"] and sha(rt/"launch.sh")==reg["launcher"],"Prereg drift")
    rows,anchors=[],[]
    for i in indices:
        paths=[d/f"pair-{i:02}.json" for d in roots]
        r,p,v=map(read,paths)
        check(all(y["index"]==i for y in (r,p,v)) and v["seed"]==r["seed"] and
              sha(paths[1])==r["reference"]==refs[i] and sha(paths[2])==p["reference"],"Wrong paired input")
        row=reduce_row(r,p,v["native"],reg)
        row.update(sha256=sha(paths[0]),native_reference=p["reference"])
        rows.append(row)
        if i in (0,10,20):
            for arm in ("off","on"):
                for e in r["events"][arm]:
                    c=e["criterion"] or {}
                    anchors.append(dict(index=i,arm=arm,ordinal=e["ordinal"],age=e["age"],route=e["route"],
                               accepted=e["accepted"],changed_argmax=c.get("changed_argmax"),mean_entropy=c.get("mean_entropy"),
                               hit=c.get("hit")))
        if i==10:
            on=[x["criterion"] for x in r["events"]["on"] if x["criterion"]]
            check(on==r["native_criterion"],"#10 true native stopper did not reproduce")
            nt=v["native"]["tape"]
            st=[x for x in r["on"]["tape"] if x["route"]=="dense_single"]
            for n,s in zip(nt,st,strict=True):
                check(all(n[k]==s[k] for k in ("clock","sample","accepted","ids","draft","pos")), "Native tape mismatch")
            for e,c in zip((x for x in r["events"]["on"] if x["criterion"]),on,strict=True):
                anchors.append(dict(index=i,arm="native",ordinal=e["ordinal"],age=e["age"],route="native_sample",
                              accepted=None,changed_argmax=c["changed_argmax"],mean_entropy=c["mean_entropy"],hit=c["hit"]))
    args.out.mkdir(exist_ok=True,parents=True)
    groups=[("anchors3",[r for r in rows if r["index"] in (0,10,20)])]
    if not args.anchors:
        groups.extend([("original30",rows[:30]),("new60",rows[30:]),("frozen90",rows)])
    table(args.out/"cohorts.csv",[summary(k,g) for k,g in groups])
    table(args.out/"pairs.csv",rows)
    table(args.out/"anchor_ticks.csv",anchors)
    provenance=dict(registration=reg,complete=stamp,raw=str(root),control=str(roots[1]),native=str(roots[2]),
                    reducer=sha(Path(__file__)),test=sha(rt/"tests/test_gemma_predicate_only.py"))
    (args.out/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")


if __name__=="__main__":
    main()
