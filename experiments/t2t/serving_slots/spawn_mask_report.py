"""Offline paired scoring of the progress80 Q64 mask GPU experiment.

Run `... -m experiments.t2t.serving_slots.spawn_mask_report 30` after sealed-30,
or `... 90` after confirmed holder return. No GPU or generation is used.
"""
from collections import Counter
from statistics import mean
import sys

from experiments.t2t.serving_slots import llada_all_slots90 as reg
from experiments.t2t.serving_slots import spawn_mask_gpu as study


def commit_pairs(events):
    """Classify recorded Q64 commit ticks, not estimated resident cache equality."""
    c = Counter()
    for j, event in enumerate(events):
        if event.get("top_action") != "commit_head":
            continue
        prev = events[j-1] if j else {}
        after = events[j+1] if j+1 < len(events) else {}
        head = set(range(32 * event["block"], 32 * (event["block"]+1)))
        tail = {p+32 for p in head}
        head_debt, tail_debt = map(set,(event["mandatory_head_positions"],event["mandatory_tail_positions"]))
        reg.require(event["nominal_q"] == 64 and set(event["absolute_positions"]) == head | tail_debt
                    and event["physical_q"] == 32+len(tail_debt) and not event["sampled_q"]
                    and head_debt <= head and tail_debt <= tail, "Unexpected commit work")
        c["commits"] += 1
        c["head_debt_commits"] += bool(head_debt)
        c["tail_debt_commits"] += bool(tail_debt)
        c["previous_full_head_without_head_debt"] += (
            prev.get("block") == event["block"]
            and head <= set(prev["absolute_positions"]) and not head_debt)
        if not after:
            c["terminal_commits"] += 1
        elif after["top_action"] == "promote_verify":
            reg.require(after["block"] == event["block"]+1 and after["offer_tick"] == event["offer_tick"]+1
                        and after["context_version"] == event["context_version"]+1
                        and set(after["sampled_positions"]) == set(after["absolute_positions"]) == tail
                        and after["physical_q"] == 32 and after["nominal_q"] == 64,
                        "Not a full next-offer verification of this tail")
            c["adjacent_pairs"] += 1
            c["pairs_with_tail_debt"] += bool(tail_debt)
            c["duplicate_tail_rows"] += len(tail_debt)
        else:
            reg.require(after["top_action"] == "head_bootstrap" and after["block"] == event["block"]+1,
                        "Unrecognized post-commit transition")
            c["fresh_head_commits"] += 1
    return c


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
    exposed, actions, pair_counts = {}, {}, {}
    for t in tasks[:n]:
        i=t["index"]
        pair = [reg.read(root/reg.filename(i,a)) for a in (on,off)]
        compare(*pair)
        exposed[i] = bool(pair[0]["first"])
        for arm, r in zip((on,off),pair,strict=True):
            audits=study.sf.audit(r,arm,opened_control=arm==off)
            if arm==on:
                pair_counts[i]=commit_pairs(r["events"])
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
    result,external,by_action,work_bound=[],[],[],[]
    for w in ((30,) if n == 30 else (30,60,90)):
        selected=select_window_cases(tasks,w)
        ids=[t["index"] for t in selected]
        total=sum((pair_counts[i] for i in ids),Counter())
        raw=sides[on]
        def macro(remove):
            return mean(raw[i]["N"]/(raw[i]["F"]-(pair_counts[i]["adjacent_pairs"] if remove else 0)) for i in ids)
        total_f=sum(raw[i]["F"] for i in ids)
        total_q=sum(raw[i]["Q"] for i in ids)
        work_bound.append(dict(window=w,policy=on,**dict(total),
            measured_F=total_f,measured_Q=total_q,measured_macro_TPF=macro(False),
            frozen_trace_if_all_pairs_fused_F=total_f-total["adjacent_pairs"],
            frozen_trace_if_all_pairs_fused_Q=total_q-total["duplicate_tail_rows"],
            frozen_trace_if_all_pairs_fused_macro_TPF=macro(True),
            flow_w2_measured_macro_TPF=mean(ref[i]["N"]/ref[i]["F"] for i in ids)))
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
    table(study.HERE.with_name("controls")/name.replace("-mask-","-commit-pairs-"),work_bound)
    return result


if __name__ == "__main__":
    if sys.argv[1:] not in (["30"],["90"]):
        raise SystemExit("Specify the sealed cohort: 30 or 90")
    score(int(sys.argv[1]))
