"""Reduce sealed A1 receipts. Only aggregate counters, indices and hashes leave the host."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

from .gemma_native_visibility_report import diff, read, sha

ALL_ACCEPTED = hashlib.sha256(bytes([1]) * 256).hexdigest()


def check(value, message):
    if not value:
        raise ValueError(message)


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def block_tapes(tape, native):
    blocks = []
    for tick in tape:
        if tick.get("clock") == 48:
            blocks.append([])
        if native or tick.get("route") == "dense_single":
            blocks[-1].append(tick)
    return blocks


def anchors(row, native):
    n = block_tapes(native["tape"], True)
    out = {}
    for arm in ("off", "on"):
        strict = row[arm]
        t = block_tapes(strict["tape"], False)
        check(len(t) == len(row["audit"][arm]["init"]), "Missing strict opening")
        blocks = []
        for j, (x,y) in enumerate(zip(n,t)):
            keys = ("clock","ids","pos","draft","sample","accepted")
            matching = next((k for k,(u,v) in enumerate(zip(x,y)) if any(u[z]!=v[z] for z in keys)),min(len(x),len(y)))
            blocks.append(dict(ordinal=j, initial_ids_native=x[0]["ids"] == y[0]["ids"],
                    positions_native=x[0]["pos"] == y[0]["pos"], native_steps=len(x), strict_steps=len(y),
                    exact_leading_steps=matching, strict_last_sample_all_accepted=y[-1]["accepted"] == ALL_ACCEPTED,
                    native_delivery_equal=native["ids"][256*j:256*j+256]
                        == strict["ids"][256*j:256*j+256]))
        out[arm]=dict(native_blocks=len(n), strict_blocks=len(t), aligned=blocks)
    return out


def audit(row, prior, ref, registration, i):
    check(prior["index"] == row["index"] == i and prior["seed"] == row["seed"] and
          prior["on"] == row["off"] and ref == row["reference"] and
          row["pid"] == registration["pid"] and row["registration"] == digest(registration),
          "Off does not replay full visibility-on row or registration mismatch")
    a,b = (row["audit"][x] for x in ("off","on"))
    h0,h1 = ([s for s in d["ticks"] if s["ordinal"] == 0] for d in (a,b))
    exp = len(a["init"]) > 1
    check(h0 == h1 and a["init"][0] == b["init"][0] and row["off"]["ids"][:256] == row["on"]["ids"][:256]
          and exp == (len(b["init"]) > 1), "First head not matched")
    if exp:
        check(a["init"][1]["ids"] != b["init"][1]["ids"], "Missing second-canvas effect")
    else:
        check(a == b and row["off"] == row["on"], "Unexposed mismatch")
    for arm in ("off","on"):
        v = row[arm]
        d = row["audit"][arm]
        verify = 0
        last = None
        for j,x in enumerate(d["init"]):
            t = [z for z in d["ticks"] if z["ordinal"] == j]
            check(x["ordinal"] == j and x["carried"] == bool(j and arm=="on") and
                  (not x["carried"] or x["before"] == last) and t and t[0]["before"] == x["after"] and
                  x["before"] != x["after"] and t[-1]["route"] in ("verify","verify_paid") and
                  all(s["route"] == "dense_single" for s in t[:-1]) and
                  all(u["end"] == z["before"] for u,z in zip(t,t[1:])) and
                  all((u["end"] != u["before"]) == (u["route"] == "dense_single") for u in t),"Bad block RNG tape")
            verify += 1
            last = t[-1]["end"]
        check(verify == v["encoder_F"]+1 and len(d["ticks"])*256 == v["decoder_Q"]
              and v["F"] == len(d["ticks"])+v["encoder_F"] and v["mask_layer_calls"] == 30*len(d["ticks"]),
              "Bad physical bill")
    return exp,len(h0)


def table(p, rows):
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,rows[0].keys(),lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def group(name, rows):
    from .gemma_rejected_noise_report import interval
    g=dict(cohort=name,n=len(rows),exposed=sum(r["exposed"] for r in rows),
           unexposed=len(rows)-sum(r["exposed"] for r in rows),
           rescues=sum(r["on_correct"]>r["off_correct"] for r in rows),
           losses=sum(r["on_correct"]<r["off_correct"] for r in rows),
           native_exact_gained=sum(r["on_native_exact"]>r["off_native_exact"] for r in rows),
           native_exact_lost=sum(r["on_native_exact"]<r["off_native_exact"] for r in rows),
           off_on_same_ids=sum(r["off_on_first_diff"]==-1 for r in rows))
    for arm in ("off","on","native"):
        for k in ("N","F","decoder_Q","encoder_F","correct","sample_F","verify_F"):
            g[f"{arm}_{k}"]=sum(r[f"{arm}_{k}"] for r in rows)
        g[arm+"_macro_TPF"]=mean(r[arm+"_TPF"] for r in rows)
        g[arm+"_TE"]=g[arm+"_N"]/g[arm+"_decoder_Q"]
        if arm!="native":
            for k in ("native_exact","native_first_equal","next_inits","carried","close_all_accepted"):
                g[f"{arm}_{k}"]=sum(r[f"{arm}_{k}"] for r in rows)
            g[arm+"_first_block_diff"]=len(rows)-g[arm+"_native_first_equal"]
            g[arm+"_later_only_diff"]=g[arm+"_native_first_equal"]-g[arm+"_native_exact"]
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
    rt=Path("/gs/runtime/gemma-continuous-rng-a1-1128")
    prev=Path("/gs/results/gemma-native-visibility-a1-1128")
    root=Path("/gs/results/gemma-continuous-rng-a1-1128")
    reg=read(root/"registration.json")
    repo=Path("/gs/runtime/eos-handle-1128/repo")
    src=repo/"experiments/t2t/serving_slots"
    for name,expected in reg["sources"].items():
        path=next(p for p in (rt/"experiments/t2t/serving_slots"/name,
                Path("/gs/runtime/gemma-native-visibility-a1-1128/experiments/t2t/serving_slots")/name,
                src/name,repo/"bdlm_serving/methods/t2t"/name) if p.is_file())
        check(sha(path)==expected,"Pinned source changed: "+name)
    for path,key in ((rt/"registration.md","policy"),(rt/"launch.sh","launcher"),
                     (repo/"benchmarks/frozen90/manifest.json","manifest"),(Path("/gs/code/samples.json"),"inputs")):
        check(sha(path)==reg[key],"Registered file changed: "+str(path))
    check(all(sha(Path(reg["checkpoint"])/k)==v for k,v in reg["checkpoint_configs"].items()),"Checkpoint config changed")
    with (rt/"validated_visibility/pairs.csv").open() as f:
        refs={int(r["index"]):r["sha256"] for r in csv.DictReader(f)}
    idx=[0,10,20] if args.anchors else range(90)
    stamp=read(root/("checkpoint-3.json" if args.anchors else "complete.json"))
    check(digest(refs)==reg["ref"] and len(idx)==stamp["n"] and stamp["pid"]==reg["pid"]
          and stamp["registration"]==digest(reg) and (args.anchors or len(list(root.glob("pair-*.json")))==90),
          "Incomplete or mismatched run")
    rows,anc=[],{}
    for i in idx:
        name=f"pair-{i:02}.json"
        row=read(root/name)
        prior=read(prev/name)
        check(sha(prev/name)==refs[i],"Reference hash mismatch")
        exposed,first=audit(row,prior,refs[i],reg,i)
        out=dict(index=i,dataset=row["dataset"],seed=row["seed"],input_sha=row["input_sha"],
                 row_sha=sha(root/name),reference_sha=refs[i],exposed=int(exposed),first_head_F=first,
                 first_head_event_digest=digest([t for t in row["audit"]["on"]["ticks"] if not t["ordinal"]]),
                 off_on_first_diff=diff(row["off"]["ids"],row["on"]["ids"]))
        n=prior["native"]
        for arm in ("off","on","native"):
            v=n if arm=="native" else row[arm]
            for k in ("N","F","decoder_Q","encoder_F","correct"):
                out[f"{arm}_{k}"]=int(v[k])
            df,rem=divmod(v["decoder_Q"],256)
            check(not rem and df+v["encoder_F"]==v["F"], "Bad forward bill")
            out[arm+"_TPF"]=v["N"]/v["F"]
            out[arm+"_verify_F"]=0 if arm=="native" else v["encoder_F"]+1
            out[arm+"_sample_F"]=df-out[arm+"_verify_F"]
            if arm!="native":
                out[arm+"_native_first_diff"]=diff(v["ids"],n["ids"])
                out[arm+"_native_first_equal"]=int(v["ids"][:256]==n["ids"][:256])
                out[arm+"_native_exact"]=int(v["ids"]==n["ids"])
                ini=row["audit"][arm]["init"]
                out[arm+"_next_inits"]=len(ini)-1
                out[arm+"_carried"]=sum(z["carried"] for z in ini)
                out[arm+"_close_all_accepted"]=sum(z["accepted"]==ALL_ACCEPTED for z in
                        row["audit"][arm]["ticks"] if z["route"] in ("verify","verify_paid"))
                out[arm+"_second_canvas_hash"]=ini[1]["ids"] if exposed else ""
        if i in (0,10,20):
            anc[i]=anchors(row,n)
            if exposed:
                check(not anc[i]["off"]["aligned"][1]["initial_ids_native"] and
                      anc[i]["on"]["aligned"][1]["initial_ids_native"], "Failed anchor")
        rows.append(out)
    args.out.mkdir(parents=True,exist_ok=True)
    table(args.out/"pairs.csv",rows)
    groups=[("anchors3",[r for r in rows if r["index"] in (0,10,20)])]
    if not args.anchors:
        groups += [("original30",rows[:30]),("new60",rows[30:]),("frozen90",rows)]
    table(args.out/"cohorts.csv",[group(k,g) for k,g in groups])
    (args.out/"anchors.json").write_text(json.dumps(anc,indent=2)+"\n")
    provenance=dict(registration=reg, checkpoint=stamp, raw=str(root), control=str(prev),
                    report_sha=sha(Path(__file__)),test_sha=sha(rt/"tests/test_gemma_continuous_rng.py"),
                    control_compact=sha(rt/"validated_visibility/pairs.csv"))
    (args.out/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")


if __name__ == "__main__":
    main()
