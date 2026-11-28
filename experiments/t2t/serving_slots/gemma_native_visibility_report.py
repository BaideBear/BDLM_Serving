"""Reduce same-stack receipts on A800; export no prompt, text, or token IDs."""
import argparse
import csv
import json
from hashlib import sha256
from itertools import takewhile
from pathlib import Path
from statistics import mean


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    return sha256(p.read_bytes()).hexdigest()


def diff(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)) if len(a)!=len(b) else -1)


def head(t):
    return [t[0],*takewhile(lambda x:x.get("clock") not in (48,None),t[1:])]


def anchor(r):
    native=r["native"]
    a=head(native["tape"])
    result={}
    for arm in ("off","on"):
        b=r[arm]
        t=head(b["tape"])
        layers=native["layers"]
        assert len(layers)==len(b["layers"]) and all(x["layer"]==y["layer"] for x,y in zip(layers,b["layers"]))
        v=dict(first_layer_output_diff=diff([l["out"] for l in layers],[l["out"] for l in b["layers"]]),
               layer0_QKV_equal=all(layers[0][k]==b["layers"][0][k] for k in ("q","k","v")),
               native_none_layers=sum(l["none"] for l in layers),arm_none_layers=sum(l["none"] for l in b["layers"]),
               first_step_equal={k:a[0][k]==t[0][k] for k in a[0]},native_head_steps=len(a),arm_head_steps=len(t),
               matching_head_steps=next((i for i,(x,y) in enumerate(zip(a,t)) if any(x[k]!=y[k]
                              for k in ("clock","ids","draft","sample","accepted"))),min(len(a),len(t))),
               native_first_block_equal=native["ids"][:256]==b["ids"][:256])
        if len(native["tape"])>len(a) and len(b["tape"])>len(t)+1:
            fresh=native["tape"][len(a)]
            strict=b["tape"][len(t)+1]
            assert fresh["clock"]==strict["clock"]==48
            v["next_block_initial_ids_equal"]=fresh["ids"]==strict["ids"]
            v["next_block_positions_equal"]=fresh["pos"]==strict["pos"]
        result[arm]=v
    return result


def main():
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.evaluation import scoring
    from .gemma_rejected_noise_report import interval
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out",type=Path)
    ap.add_argument("--anchors",action="store_true")
    args=ap.parse_args()
    root=Path("/gs/results/gemma-native-visibility-a1-1128")
    wroot=next(Path("/gs/results/flowblock-frozen90-1127/analysis/reference90").rglob("gemma-wave-reference90"))
    tasks=load_tasks(Path("/gs/code/samples.json"))
    reg=read(root/"registration.json")
    assert sha(Path(scoring.__file__))==reg["scorer"]
    indices=[0,10,20] if args.anchors else range(90)
    if not args.anchors:
        assert read(root/"complete.json")["n"]==90 and len(list(root.glob("pair-*.json")))==90
    rows=[]
    anchors={}
    for i in indices:
        hist=read(Path("/gs/results/dense-stop-1128")/f"C256-dense-stop1-{i:02}.json")
        shadow=read(Path("/gs/results/draft-shadow-1128")/f"fixed256-{i:02}.json")
        w=read(wroot/f"{i:02}-flow_w1.json")
        r=read(root/f"pair-{i:02}.json")
        assert all(x["seed"]==tasks[i]["seed"] for x in (hist,w,r,shadow)) and r["index"]==i
        assert hist["output_ids"]==r["off"]["ids"] and all(hist[k]==r["off"][k] for k in
                             ("N","correct","F","decoder_Q","encoder_F"))
        nw=w["output_ids"]
        r["historical"]=dict(ids=nw,F=w["generation_forwards"],N=w["output_tokens"],
            decoder_Q=256*len(w["trace"]),encoder_F=w["generation_forwards"]-len(w["trace"]),
            correct=bool(scoring.score(w["output_text"],tasks[i])["correct"]))
        row=dict(index=i,dataset=w["dataset"],seed=tasks[i]["seed"],sha256=sha(root/f"pair-{i:02}.json"),
                 off_reference_sha256=sha(Path("/gs/results/dense-stop-1128")/f"C256-dense-stop1-{i:02}.json"),
                 w1_reference_sha256=sha(wroot/f"{i:02}-flow_w1.json"),
                 historical_off_first_diff=diff(nw,r["off"]["ids"]),historical_native_exact=r["historical_native_equal"],
                 off_on_first_diff=diff(r["off"]["ids"],r["on"]["ids"]),
                 historical_strict_sample_F=hist["observer"]["qualified_full_ticks"],
                 historical_strict_verify_F=len(shadow["draft_shadow"]),
                 historical_w1_first_sample_F=sum(x["head"]==0 for x in w["trace"]),
                 historical_strict_first_sample_F=hist["observer"]["head_tick_tape"].split("V")[0].count("F"))
        for arm in ("native","off","on","historical"):
            x=r[arm]
            for k in ("N","F","correct","decoder_Q","encoder_F"):
                row[arm+"_"+k]=int(x[k])
            row[arm+"_TPF"]=x["N"]/x["F"]
            assert x["F"]==x["decoder_Q"]//256+x["encoder_F"]
            if arm in ("off","on"):
                row[arm+"_native_first_diff"]=diff(x["ids"],r["native"]["ids"])
                row[arm+"_native_first_block_equal"]=x["ids"][:256]==r["native"]["ids"][:256]
                row[arm+"_mask_layers"]=x["mask_layer_calls"]
        rows.append(row)
        if i in (0,10,20):
            anchors[i]=anchor(r)
    args.out.mkdir(exist_ok=True,parents=True)
    (args.out/"anchors.json").write_text(json.dumps(anchors,indent=2)+"\n")
    def table(name,rows):
        with (args.out/name).open("w",newline="") as fp:
            writer=csv.DictWriter(fp,rows[0].keys(),lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    table("pairs.csv",rows)
    groups=[("anchors3",[r for r in rows if r["index"] in (0,10,20)])]
    if len(rows)==90: groups.extend([("original30",rows[:30]),("new60",rows[30:]),("frozen90",rows)])
    out=[]
    for name,subset in groups:
        g=dict(cohort=name,n=len(subset),historical_native_exact=sum(r["historical_native_exact"] for r in subset),
               rescues=sum(r["on_correct"]>r["off_correct"] for r in subset),losses=sum(r["on_correct"]<r["off_correct"] for r in subset))
        match=[tuple(dict(dataset=r["dataset"],TPF=r[a+"_TPF"],correct=r[a+"_correct"]) for a in ("off","on")) for r in subset]
        for k in ("TPF","correct"):
            g[k+"_delta_lo95"],g[k+"_delta_hi95"]=interval(match,lambda x:x[k]) or (None,None)
        for arm in ("native","off","on","historical"):
            for k in ("N","F","correct","decoder_Q","encoder_F"):
                g[arm+"_"+k]=sum(r[arm+"_"+k] for r in subset)
            g[arm+"_macro_TPF"]=mean(r[arm+"_TPF"] for r in subset)
            g[arm+"_TE"]=g[arm+"_N"]/g[arm+"_decoder_Q"]
            if arm in ("off","on"):
                g[arm+"_native_exact_ids"]=sum(r[arm+"_native_first_diff"]==-1 for r in subset)
                g[arm+"_native_exact_first_block"]=sum(r[arm+"_native_first_block_equal"] for r in subset)
        out.append(g)
    table("cohorts.csv",out)
    rt=Path("/gs/runtime/gemma-native-visibility-a1-1128")
    assert all(sha(rt/"experiments/t2t/serving_slots"/name)==v for name,v in reg["sources"].items())
    (args.out/"provenance.json").write_text(json.dumps(dict(registration=reg,report_sha256=sha(Path(__file__)),
          sample_sha256=sha(Path("/gs/code/samples.json")),w1_root=str(wroot),raw=str(root),
          historical_w1_metadata=read(wroot/"run.json"),qualification_sha256=sha(rt/"prior-qualification.json"),
          launcher_sha256=sha(rt/"launch.sh"),prereg_sha256=sha(rt/"registration.md"),
          complete=read(root/"complete.json") if not args.anchors else None),indent=2)+"\n")


if __name__=="__main__":
    main()
