"""Reconcile sealed same-device pairs without loading a model or exporting raw data."""
import csv
import json
from pathlib import Path
from statistics import mean
from math import isclose

from .gemma_cold_frontier_last_eval import GPU, SAMPLES, SCORER, read, sha
from .gemma_joint_verify_eval import check, digest
from .gemma_rejected_noise_report import interval
from .gemma_warm_accepted_pay import ANCHORS, align, audit, gates
from .gemma_warm_accepted_pay_eval import DEADLINE, HIST, PIN, PLAN, ROOT, RT

BASE = Path("/gs/runtime/verified-eos-1128/repo")
ARMS = ("off", "on")


def bill(x, ticks, enabled):
    d = audit(x, ticks, enabled)
    last, head = 0, 0
    for e in ticks:
        check(e["paid"] == last and e["paid"] <= e["q"] and
              e["head"] in (head, head+1) and (e["head"] == head or not last) and
              0 <= e["sampled"] <= e["nominal"] <= e["q"] and
              (not e["eligible"] or (e["phase"] == "warm" and e["sampled"])) and
              e["eligible"] <= e["paid"] and (not e["eligible"] or e["eligible"] < e["nominal"]),
              "Unexplained debt/width/sample in tick")
        last, head = e["writes"], e["head"]
    check(not last and head == x["encoder_F"] and type(x["correct"]) is bool and
          d["verify_F"] == sum(x["route_F"].get(k, 0) for k in ("verify","verify_paid")) and
          d["verify_Q"] == d["verify_F"]*32 and sum(x["route_Q"].values()) == x["decoder_Q"] and
          sum(x["route_F"].values()) == len(ticks), "Unsettled close")
    for phase in ("cold", "warm", "close", "verify"):
        t = [e for e in ticks if e["phase"] == phase]
        d[phase+"_paid"] = sum(e["paid"] for e in t)
        d[phase+"_sampled"] = sum(e["sampled"] for e in t)
        d[phase+"_drains"] = sum(not e["sampled"] for e in t) if phase != "verify" else 0
    warm = [e for e in ticks if e["phase"]=="warm" and e["sampled"]]
    d.update(warm_sample_F=len(warm), warm_zero_admission_F=sum(e["admit"]==0 for e in warm),
             warm_nonpositive_net_admission_F=sum(e["admit"]<=e["revoke"] for e in warm),
             warm_admits=sum(e["admit"] for e in warm), warm_revokes=sum(e["revoke"] for e in warm),
             warm_writes=sum(e["writes"] for e in warm), opportunity_head_count=len({
                 e["head"] for e in warm if e["eligible"]}))
    d.update({k:x[k] for k in ("N","F","decoder_F","decoder_Q","encoder_F","encoder_Q")})
    d.update(correct=int(x["correct"]), TPF=x["N"]/x["F"], TE=x["N"]/x["decoder_Q"],
             output_sha256=digest(x["output_ids"]), encoder_sha256=x["encoder_digest"])
    return d


def group(name, rows, target):
    check(rows and len(rows)<=target, "Bad cohort denominator")
    s = dict(cohort=name, paired=len(rows), target=target, remaining=target-len(rows),
             exact_replays=len(rows), exposed=sum(r["exposed"] for r in rows),
             same_ids=sum(r["same_ids"] for r in rows),
             rescues=sum(r["on_correct"]>r["off_correct"] for r in rows),
             losses=sum(r["on_correct"]<r["off_correct"] for r in rows),
             TPF_wins=sum(r["on_TPF"]>r["off_TPF"] for r in rows),
             TPF_ties=sum(r["on_TPF"]==r["off_TPF"] for r in rows),
             F_lower=sum(r["on_F"]<r["off_F"] for r in rows),
             F_equal=sum(r["on_F"]==r["off_F"] for r in rows),
             decoder_Q_lower=sum(r["on_decoder_Q"]<r["off_decoder_Q"] for r in rows),
             decoder_Q_equal=sum(r["on_decoder_Q"]==r["off_decoder_Q"] for r in rows))
    for arm in ARMS:
        p = arm+"_"
        for k in rows[0]:
            if k.startswith(p) and not k.endswith(("_sha256","_TPF","_TE")):
                s[k] = (max if k==p+"peak" else sum)(r[k] for r in rows)
        s[p+"macro_TPF"] = mean(r[p+"TPF"] for r in rows)
        s[p+"TE"] = s[p+"N"]/s[p+"decoder_Q"]
    pairs = [tuple(dict(dataset=r["dataset"],TPF=r[a+"_TPF"],correct=r[a+"_correct"])
                   for a in ARMS) for r in rows]
    for k in ("TPF","correct"):
        s["delta_"+k] = mean(b[k]-a[k] for a,b in pairs)
        ci = interval(pairs, lambda x:x[k]) if len(rows)==target and target>=30 else None
        s["delta_"+k+"_lo95"], s["delta_"+k+"_hi95"] = ci or (None,None)
    # Symmetric product-difference identity for N*(1/F); neither part is a causal effect.
    s["delta_TPF_length_term"] = mean((r["on_N"]-r["off_N"])*(1/r["off_F"]+1/r["on_F"])/2 for r in rows)
    s["delta_TPF_forward_term"] = mean((r["on_N"]+r["off_N"])*(1/r["on_F"]-1/r["off_F"])/2 for r in rows)
    check(isclose(s["delta_TPF_length_term"]+s["delta_TPF_forward_term"],s["delta_TPF"],
                  abs_tol=1e-12),"TPF decomposition mismatch")
    return s


def report():
    reg, pin = read(ROOT/"registration.json"), read(PIN)
    rd = digest(reg)
    refs = [sha(HIST/f"C32-pair-{i:02}.json") for i in range(90)]
    mpath = BASE/"benchmarks/frozen90/manifest.json"
    manifest = read(mpath)
    tasks = {r["index"]:r for r in manifest["samples"]}
    src = Path(__file__).parent
    check(reg["gpu"] == pin["gpu"] == GPU and reg["deadline"] == DEADLINE and
          reg["pin"]==sha(PIN) and reg["refs"]==pin["refs"]==digest(refs) and
          pin["manifest"]==sha(mpath) and pin["inputs"]==manifest["samples_sha256"]==sha(SAMPLES) and
          pin["scorer"]==SCORER and reg["policy"]==sha(RT/"registration.md") and
          reg["launcher"]==sha(RT/"launch.sh") and
          all(sha(src/(k+".py"))==v for k,v in reg["sources"].items()) and
          all(sha(BASE/"experiments/t2t/serving_slots"/(k+".py"))==v for k,v in pin["base"].items()),
          "Source/data binding mismatch")
    marks = [s for s in ("no-go","partial","complete") if (ROOT/(s+".json")).exists()]
    check(len(marks)==1,"Require exactly one terminal marker")
    status, = marks
    mark = read(ROOT/(status+".json"))
    n = mark["pairs"]
    check(mark["reg"]==rd and 0<n<=90 and set(ROOT.glob("pair-*.json"))==
          {ROOT/f"pair-{i:02}.json" for i in PLAN[:n]} and
          (status!="complete" or n==90) and (status!="no-go" or n==6),"Not a sealed prefix")
    rows, qual = [], []
    for i in PLAN[:n]:
        path = ROOT/f"pair-{i:02}.json"
        r, ref = read(path), read(HIST/f"C32-pair-{i:02}.json")
        check(r["index"]==i and r["pid"]==reg["pid"] and r["reg"]==rd and r["ref"]==refs[i] and
              r["off"]==ref["on"] and ref["q"]==32 and ref["source"]==pin["base"] and
              ref["scorer_sha256"]==SCORER and ref["samples_sha256"]==pin["inputs"] and
              r["on"]["input_sha256"]==r["off"]["input_sha256"] and
              all(r[k]==ref[k]==tasks[i][k] for k in ("index","seed","dataset","task_id")),
              "Unmatched historical control")
        a,b = (r["ticks"][k] for k in ARMS)
        u = align(r["off"], r["on"], a, b)
        check(u==r["first"],"First divergence mismatch")
        row = {k:r[k] for k in ("index","dataset","seed","task_id")}
        row.update(exposed=int(bool(u)), first_tick=u["tick"] if u else None,
                   first_head=u["head"] if u else None, first_age=u["age"] if u else None,
                   first_physical_Q=len(u["pos"]) if u else None,
                   first_off_sample=len(u["nom"]) if u else None,
                   first_removed=len(u["skip"]) if u else 0, first_logits_sha256=u["logits"] if u else None,
                   same_ids=int(r["off"]["output_ids"]==r["on"]["output_ids"]),
                   ref_sha256=refs[i],pair_sha256=sha(path))
        for arm in ARMS:
            ticks,x = r["ticks"][arm],r[arm]
            check(audit(x,ticks,arm=="on")==r["audit"][arm],"Reported totals differ")
            row.update({arm+"_"+k:v for k,v in bill(x,ticks,arm=="on").items()})
        rows.append(row)
        if i in ANCHORS:
            qual.append(r)
    g = gates(qual) if n>=6 else None
    if g:
        check(read(ROOT/"qualification.json")==dict(gates=g,indices=list(ANCHORS),reg=rd) and
              (all(g.values())==(status!="no-go")),"Gate mismatch")
    groups = [group("anchors", [r for r in rows if r["index"] in ANCHORS],6)]
    for name,subset,target in (("original30",[r for r in rows if r["index"]<30],30),
                               ("new60",[r for r in rows if r["index"]>=30],60),
                               ("nonanchors84",[r for r in rows if r["index"] not in ANCHORS],84),
                               ("frozen90",rows,90)):
        if subset:
            groups.append(group(name if len(subset)==target else name+"_partial", subset, target))
    prov = dict(raw_root=str(ROOT),registration=reg,historical_control=str(HIST),pin_path=str(PIN),
                status=status,marker=mark,gates=g,pairs=n,unrun=sorted(set(range(90))-set(PLAN[:n])),
                cohort_selection="qualification cases are included; not an independent holdout",
                pair_hashes_digest=digest([(r["index"],r["pair_sha256"]) for r in rows]),
                report_sha256=sha(Path(__file__)),TE="sum(N)/sum(selected history-reading decoder Q)",
                macro_TPF="mean_i(N_i/(decoder_F_i+post-prefill encoder_F_i))")
    return sorted(rows,key=lambda r:r["index"]),groups,prov


def main():
    rows,groups,prov = report()
    out = RT/"compact"
    out.mkdir(exist_ok=True)
    for name,data in (("pairs",rows),("cohorts",groups)):
        with (out/(name+".csv")).open("w",newline="") as fp:
            w=csv.DictWriter(fp,data[0],lineterminator="\n")
            w.writeheader()
            w.writerows(data)
    (out/"provenance.json").write_text(json.dumps(prov,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    main()
