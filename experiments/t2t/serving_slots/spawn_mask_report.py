"""Offline paired scoring of the progress80 Q64 mask GPU experiment.

Run `... -m experiments.t2t.serving_slots.spawn_mask_report 30` after sealed-30,
or `... 90` after confirmed holder return. No GPU or generation is used.
"""
import sys

from experiments.t2t.serving_slots import llada_all_slots90 as reg
from experiments.t2t.serving_slots import spawn_mask_gpu as study


def score(n):
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.evaluation.scoring import score as judge
    from experiments.t2t.serving_slots.llada_spawn_pilot import compare
    from experiments.t2t.serving_slots.mask_pair_report import efficiency_interval
    from tools.prefix_cohorts import select_window_cases, _window_row
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
    exposed = {}
    for t in tasks[:n]:
        i=t["index"]
        pair = [reg.read(root/reg.filename(i,a)) for a in (on,off)]
        compare(*pair)
        exposed[i] = bool(pair[0]["first"])
        for arm, r in zip((on,off),pair,strict=True):
            audits=study.sf.audit(r,arm,opened_control=arm==off)
            reg.require(all(t[k]==r[k] for k in reg.KEYS), "Frozen identity mismatch")
            sides[arm][i]=dict(index=i, dataset=t["dataset"], task_id=t["task_id"],
                               N=audits["N"], F=audits["F"], Q=audits["Q"],
                               physical_Q=audits["Q"], correct=bool(judge(r["output_text"],t)["correct"]),
                               failed=False, truncated=r["truncated"], tpf=r["tpf"], max_q=r["max_q"])
    result=[]
    for w in ((30,) if n == 30 else (30,60,90)):
        selected=select_window_cases(tasks,w)
        for scope in ("all","gsm8k","math500","humaneval"):
            sub=[t for t in selected if scope=="all" or t["dataset"]==scope]
            lo,hi=efficiency_interval(sub,sides[off],sides[on])
            r=_window_row(selected,sides[off],sides[on],w,scope,"llada21",off,on,64)
            result.append(dict(**r, device="gpu0", gate=.8, exposed=sum(exposed[t["index"]] for t in sub),
                              efficiency_ci_low=lo,efficiency_ci_high=hi, run_sha256=reg.sha(root/"run.json")))
    name="llada21-q64-progress80-mask-gpu"+("90" if n==90 else "-original30")+".csv"
    table(study.HERE.with_name("controls")/name,result)
    return result


if __name__ == "__main__":
    if sys.argv[1:] not in (["30"],["90"]):
        raise SystemExit("Specify the sealed cohort: 30 or 90")
    score(int(sys.argv[1]))
