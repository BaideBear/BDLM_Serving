"""Offline paired scoring of the progress80 Q64 mask GPU experiment.

Run `... -m experiments.t2t.serving_slots.spawn_mask_report 30` after sealed-30,
or `... 90` after confirmed holder return. No GPU or generation is used.
"""
from collections import Counter
import sys

from experiments.t2t.serving_slots import llada_all_slots90 as reg
from experiments.t2t.serving_slots import spawn_mask_gpu as study


def score(n):
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.evaluation.scoring import score as judge
    from experiments.t2t.serving_slots.llada_spawn_pilot import compare
    from experiments.t2t.serving_slots.mask_pair_report import efficiency_interval
    from tools.prefix_cohorts import select_window_cases, load_request_set, _window_row
    from tools.t2t_results import table

    path = reg.DATA/"samples.json"
    root = study.OUT
    run = reg.read(root/"run.json")
    # Later source edits must not relabel old results; retain the original
    # run checksum and require the same judge and accounting rules at readback.
    expected = study.protocol()
    reg.require(reg.sha(path) == reg.SAMPLES_SHA and
                all(run[k] == v for k,v in expected.items() if k != "sources")
                and run["sources"]["bdlm_serving/evaluation/scoring.py"] ==
                expected["sources"]["bdlm_serving/evaluation/scoring.py"], "Wrong GPU protocol or judge")
    seal = study.sealed(n)
    if n == 90:
        receipt = reg.read(root/"returned.json")
        reg.require(receipt["restored"] and receipt["gpu"] == 0 and receipt["pid"] > 0
                    and receipt["uuid"] == study.GPU and receipt["original30"] == study.sealed(30)
                    and receipt["full90"] == seal, "Missing GPU return")
    tasks = load_tasks(path)
    on,off = study.ARMS
    sides = {on:{},off:{}}
    exposed, actions = {}, {}
    for t in tasks[:n]:
        i=t["index"]
        pair = [reg.read(root/reg.filename(i,a)) for a in (on,off)]
        compare(*pair)
        exposed[i] = bool(pair[0]["first"])
        for arm, r in zip((on,off),pair,strict=True):
            audits=study.sf.audit(r,arm,opened_control=arm==off)
            reg.require(all(t[k]==r[k] for k in reg.KEYS), "Frozen identity mismatch")
            flow = actions[i,arm] = {}
            for e in r["events"]:
                if e["phase"]=="prefill":
                    continue
                action=e["top_action"]
                bucket=flow.setdefault(action,Counter())
                bucket.update(F=1, physical_Q=e["physical_q"], mandatory_Q=e["mandatory_q"],
                              strict_T2T=e["hard_strict_edits"])
            reg.require(sum(b["F"] for b in flow.values())==audits["F"]
                        and sum(b["physical_Q"] for b in flow.values())==audits["Q"],
                        "Stage costs do not match request")
            sides[arm][i]=dict(index=i, dataset=t["dataset"], task_id=t["task_id"],
                               N=audits["N"], F=audits["F"], Q=audits["Q"],
                               physical_Q=audits["Q"], correct=bool(judge(r["output_text"],t)["correct"]),
                               failed=False, truncated=r["truncated"], tpf=r["tpf"], max_q=r["max_q"])
    wave = reg.DATA/"llada21-reference90-wave"
    meta=reg.read(wave/"run.json")
    reg.require((meta["model"],meta["suite"],meta["samples_sha256"],meta["flowblock_block_size"]) ==
                ("llada21","frozen90",reg.SAMPLES_SHA,32)
                and reg.read(wave/"complete-flow_w2.json")=={"policy":"flow_w2","requests":90},
                "Missing same-cohort B32 FlowBlock W2 reference")
    refs,model,policy,_=load_request_set(wave,tasks,"flowblock","flow_w2")
    reg.require((model,policy)==("llada21","flow_w2"), "Wrong external comparator")
    ref={r["index"]:r for r in refs}
    result,external,by_action=[],[],[]
    for w in ((30,) if n == 30 else (30,60,90)):
        selected=select_window_cases(tasks,w)
        for arm in (off,on):
            groups={}
            for t in selected:
                for act,b in actions[t["index"],arm].items():
                    groups.setdefault(act,Counter()).update(b)
            by_action.extend(dict(window=w,policy=arm,action=k,**v) for k,v in sorted(groups.items()))
        for scope in ("all","gsm8k","math500","humaneval"):
            sub=[t for t in selected if scope=="all" or t["dataset"]==scope]
            lo,hi=efficiency_interval(sub,sides[off],sides[on])
            r=_window_row(selected,sides[off],sides[on],w,scope,"llada21",off,on,64)
            result.append(dict(**r, device="gpu0", gate=.8, exposed=sum(exposed[t["index"]] for t in sub),
                              efficiency_ci_low=lo,efficiency_ci_high=hi, run_sha256=reg.sha(root/"run.json")))
            lo,hi=efficiency_interval(sub,ref,sides[on])
            external.append(dict(**_window_row(selected,ref,sides[on],w,scope,"llada21","flow_w2",on,64),
                                 efficiency_ci_low=lo,efficiency_ci_high=hi,
                                 comparison="same nominal maximum; different kernels, actual Q reported"))
    name="llada21-q64-progress80-mask-gpu"+("90" if n==90 else "-original30")+".csv"
    table(study.HERE.with_name("controls")/name,result)
    table(study.HERE.with_name("controls")/name.replace("-mask-","-vs-flow-w2-"),external)
    table(study.HERE.with_name("controls")/name.replace("-mask-","-work-"),by_action)
    return result


if __name__ == "__main__":
    if sys.argv[1:] not in (["30"],["90"]):
        raise SystemExit("Specify the sealed cohort: 30 or 90")
    score(int(sys.argv[1]))
