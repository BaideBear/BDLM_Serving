"""Same-card six-case gate for Q16 agreement, with paid close in all arms."""
import argparse
from collections import Counter
import os
from pathlib import Path
from unittest.mock import patch

from tools.llada_ibs_eos_eval import BASE, CAP, CKPT, CONTROLS, DATA, KEYS, UUID, compact, digest, file_sha, read, save
from tools.llada_ibs_eos_q16_agreement import ARMS, B, Q, install_run, require

SIX = (7, 13, 25, 33, 55, 71)
LABEL = "optcap_ibs_16"


def independent_eligible(tape, excluded):
    prev, votes = None, {}
    for t in tape:
        tag = (t["block"], t["tick"])
        fresh = {p: v for p, v, prob, native in zip(t["positions"], t["proposals"],
                    t["probabilities"], t["native"], strict=True)
                    if not native and prob > .5 and v not in excluded}
        want = tuple(p for p,v in fresh.items() if prev == (tag[0], tag[1]-1) and votes.get(p) == v)
        require(tuple(t["eligible"]) == want, "Nonadjacent, stale or excluded agreement")
        require(prev is None or tag[1] > prev[1], "Nonphysical tick clock")
        yield t, want
        prev, votes = tag, fresh


def audit(row, tape, arm, *, eos, mask, excluded):
    pending, total, events, init = {}, Counter(), {}, set()
    first = tape["certificate"]
    p = len(row["prompt_ids"])
    cursor = 0
    previous_block, ready = None, 0
    require(row["verified_layers"] == 20, "Missing all-layer Q hooks")
    for j,(e,c,ids) in enumerate(zip(row["events"], row["calls"], row["forward_inputs"], strict=True)):
        pos = e["absolute_positions"]
        require(pos == sorted(set(pos)) and c["batch"] == e["physical_forwards"] == 1 and
                c["physical_rows"] == c["rows"] == e["q_tokens"] == len(pos) == len(ids),
                "Physical projection ledger mismatch")
        init.update(pos)
        if e["phase"] == "prefill":
            require(j == 0 and pos == list(range(p//B*B)), "Wrong prefill")
            continue
        require(0 < len(pos) <= Q and c["kv_rows"] == e["visible_kv"] == max(pos)+1 and
                sorted(pending) == e["dirty_before"] and
                all(dict(zip(pos,ids)).get(k) == v for k,v in pending.items()), "Lost owed input")
        total.update(F=1,Q=len(pos))
        if e["phase"] == "certificate_debt":
            require(first is not None and first["event"] == j-1 and pos == first["debt_before"] and
                    sorted(pending) == pos and not (e["written_absolute"] or e["sampled_absolute"] or
                    e["edited_positions"]) and j == len(row["events"])-1, "Invalid real debt-only payment")
            pending = {}
            total.update(paid_D=1, paid_Q=len(pos))
            continue
        require(e["phase"] == "optimus" and e["offer_tick"] == cursor and e["physical_cap"] == Q,
                "Unexpected physical ordinary step")
        block = e["block"]
        if block != previous_block:
            require(previous_block is None or (block == previous_block+1 and ready == B and not pending),
                    "Invalid block rollover")
            ready = 0
        start = block*B
        candidates = [start+i for i in e["candidate_local"]]
        planned = sorted((sorted(pending)+[v for v in candidates if v not in pending])[:Q]) if len(candidates)>Q else candidates
        require(e["cached_prefix_before"] == ready and pos == planned == [start+i for i in e["selected_local"]]
                and all(start+ready <= v < start+B for v in pos), "Changed strict IBS row selection")
        writes = tape["writes"][cursor]
        require([k for k,_ in writes] == e["written_absolute"] and
                all(k in pos and k >= p for k,_ in writes), "Unbilled native or extra write")
        pending = dict(writes)
        ready = e["cached_prefix_after"]
        require(all(k >= start+ready for k in pending), "Unpaid retired write")
        if first and first["event"] == j:
            eo = first["eos"]
            require(first["ready"] == ready and first["block"] == block and first["step"] == e["step"] and
                    max(p,start) <= eo < min(start+ready,p+CAP) and first["debt_before"] == sorted(pending) and
                    len(row["events"]) == j+1+bool(pending) and first["debt_after"] == [],
                    "Wrong certified stop boundary")
            suffix = row["output_ids"]
            require(len(suffix) == eo-p+1 and suffix[-1] == eos and eos not in suffix[:-1]
                    and mask not in suffix and row["status"] == "success", "Invalid certified output")
        events[cursor] = (e,dict(writes))
        previous_block, cursor = block, cursor+1
    require(cursor == len(tape["writes"]) and (not pending or row["request_failure"])
            and total["F"] == row["generation_forwards"] and total["Q"] == row["generation_query_rows"] ==
            row["generation_physical_padded_rows"], "Incomplete execution bill")
    taps = {t["tick"]:t for t in tape["ticks"]}
    if arm != "stock":
        require(set(taps) == {tick for tick,(e,_) in events.items() if any(
            m and p <= pos < p+CAP for pos,m in zip(e["absolute_positions"],e["selected_masked"],strict=True))},
            "Tracked row or empty-tick clock mismatch")
    for t,want in independent_eligible(tape["ticks"], excluded):
        e,w = events[t["tick"]]
        require(t["block"] == e["block"] and t["positions"] == [pos for pos,masked in
                zip(e["absolute_positions"],e["selected_masked"],strict=True) if masked and p <= pos < p+CAP],
                "Evidence read unmeasured rows")
        applied = want if arm == "apply" else ()
        require(tuple(t["applied"]) == applied and all(w.get(pos) == value for pos,value in
                zip(t["positions"],t["proposals"],strict=True) if pos in applied), "Extra did not enter ordinary debt")
        total.update(extra=len(applied),eligible=len(want))
    return dict(**{k:total[k] for k in ("F","Q","paid_D","paid_Q","extra","eligible")}, N=row["output_tokens"],
                failure=row["request_failure"], eos=row["eos"], ids_sha=digest(row["output_ids"]))


def first_fork(baseline, trial, left, right):
    require(baseline["prompt_ids"] == trial["prompt_ids"], "Changed input")
    first = next((t for t in right["ticks"] if t["applied"]), None)
    if first is None:
        require(baseline == trial and left == right, "Changed output without an extra admission")
        return None
    tick = first["tick"]
    index = tick + int(baseline["events"][0]["phase"] == "prefill")
    for k in ("events","calls","forward_inputs"):
        require(baseline[k][:index] == trial[k][:index], f"Premature fork: {k}")
    require(left["writes"][:tick] == right["writes"][:tick], "Premature write fork")
    earlier = [x for x in left["ticks"] if x["tick"] <= tick]
    later = [x for x in right["ticks"] if x["tick"] <= tick]
    require(earlier == [dict(x,applied=()) for x in later], "Native votes changed before fork")
    a,b = baseline["events"][index],trial["events"][index]
    require(baseline["calls"][index] == trial["calls"][index] and
            baseline["forward_inputs"][index] == trial["forward_inputs"][index] and all(a[k]==b[k] for k in
            ("absolute_positions","selected_masked","selected_local","candidate_local","visible_kv",
             "dirty_before","cached_prefix_before","edited_positions","cache_deferred_t2t_positions")),
            "Fork changed same-tick physical rows or native T2T")
    extra = set(first["applied"])
    before, after = map(dict,(left["writes"][tick],right["writes"][tick]))
    require(not (extra & before.keys()) and set(after) == before.keys()|extra and
            all(after[k] == v for k,v in before.items()) and b["accepted"]-a["accepted"] == len(extra)
            and set(b["written_absolute"]) == set(a["written_absolute"])|extra,
            "First fork is not only added native-debt writes")
    return dict(tick=tick, block=first["block"], rows=a["absolute_positions"],
                positions=sorted(extra), dirty_before=a["dirty_before"],
                native_t2t=a["edited_positions"])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=("check","run"))
    ap.add_argument("--panel",type=Path,required=True)
    ap.add_argument("--out",type=Path)
    a=ap.parse_args()
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.methods.optimus import fixed, upstream
    from bdlm_serving.methods.t2t import paid_agreement
    from experiments.t2t.serving_slots import ibs_prefix_pair as pair, llada_reference as ref, shared_llada_worker as worker

    panel=read(a.panel)
    require(panel["six"] == list(SIX) and panel["arms"] == list(ARMS) and panel["q"] == Q and panel["uuid"] == UUID
            and panel["cap"] == CAP and panel["evidence_sha"] == file_sha(Path(paid_agreement.__file__))
            and panel["samples_sha256"] == file_sha(DATA/"samples.json")
            and panel["ledger_sha256"] == file_sha(DATA/"fixed-family-reference-case-ledger-v1.json"),
            "Changed registered inputs")
    def controls(name):
        require(name == "Optimus" and all(file_sha(CONTROLS/k)==v for k,v in panel["opt_controls"].items()),
                "Changed Optimus controls")
        return CONTROLS
    controls("Optimus")
    for arm in ARMS:
        install_run(fixed,arm,{},set())
    if a.mode=="check":
        print("Preflight: source anchors, byte-pinned shared ledger, panel; no inference")
        return
    import torch
    require(a.out and os.environ["CUDA_VISIBLE_DEVICES"] == UUID and torch.cuda.device_count() == 1 and
            str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-") == UUID[4:],
            "Only registered H0 may run")
    h=worker.load(CKPT)
    tok,model=h["tok"],h["model"]
    eos,mask=h["eos_id"],h["mask_id"]
    excluded=set(tok.all_special_ids)|{eos,mask}|{i for i,t in tok.added_tokens_decoder.items() if t.special}
    excluded|={t["id"] for t in read(CKPT/"tokenizer.json")["added_tokens"] if t["special"]}
    tasks=load_tasks(DATA/"samples.json")
    registration=dict(panel=panel,panel_sha=file_sha(a.panel),code=os.environ["BDLM_Q16_SOURCE"],
        pid=os.getpid(),uid=os.getuid(),ticks=int(Path("/proc/self/stat").read_text().split(") ")[1].split()[19]),
        checkpoint=h["fingerprint"],scorer=h["scorer"],excluded=sorted(excluded),
        torch=str(torch.__version__))
    reg=digest(registration)
    save(a.out/"registration.json",registration)
    (a.out/"pairs").mkdir()
    rows=[]
    with patch.object(pair,"Q",Q),patch.dict(pair.LABELS,{0:LABEL}),patch.object(upstream,"control_directory",controls):
        for j,i in enumerate(SIX):
            task=tasks[i]
            hist=read(DATA/f"llada21-optimus-hardcap-endpoints/{i:02}-{LABEL}.json")
            require(all(hist[k]==task[k] for k in KEYS),"Task identity drift")
            prompt,_=worker.canonical_prompt(tok,task,hist["prompt_ids"])
            observed,tapes={},{}
            order=ARMS[j%3:]+ARMS[:j%3]
            for arm in order:
                tape=tapes[arm]=dict(certificate=None,ticks=[],writes=[])
                torch.manual_seed(task["seed"])
                with patch.object(fixed.FixedOptimus,"run",install_run(fixed,arm,tape,excluded)):
                    observed[arm]=pair.generate(model,tok,task,prompt,CAP,0)
            path=a.out/"pairs"/f"{i:02}.json"
            save(path,dict(arms=observed,tapes=tapes,reg=reg,order=order))
            stock,tracked,trial=(observed[k] for k in ARMS)
            require(stock==tracked and all(tapes["stock"][k]==tapes["tracked"][k] for k in ("certificate","writes")),
                    "Tracked no-op changed stock paid close")
            cert=tapes["stock"]["certificate"]
            if cert is None:
                ref.verify_row(hist,stock)
                require(hist["events"] == stock["events"], "Stock trace differs from history")
            else:
                n=cert["event"]+1
                require(stock["output_ids"] == hist["output_ids"] and stock["events"][:n]==hist["events"][:n] and
                        all(all(c[k]==d[k] for k in ("batch","rows","physical_rows","kv_rows"))
                        for c,d in zip(stock["calls"][:n],hist["calls"][:n],strict=True)),
                        "Historical stock Q16 prefix or ID drift")
            bills={k:audit(v,tapes[k],k,eos=eos,mask=mask,excluded=excluded) for k,v in observed.items()}
            for arm in ARMS:
                r=observed[arm]
                bills[arm]["correct"]=r["status"]=="success" and bool(scoring.score(r["output_text"],task)["correct"])
            f=first_fork(tracked,trial,tapes["tracked"],tapes["apply"])
            stamp=dict(index=i,reg=reg,arms=bills,first=f,raw_sha=file_sha(path),order=order)
            rows.append(stamp)
            print(compact(stamp),flush=True)
    totals={arm:{k:sum(x["arms"][arm][k] for x in rows) for k in
                ("F","Q","N","extra","eligible","paid_D","paid_Q","correct")} for arm in ARMS}
    losses=[x["index"] for x in rows if x["arms"]["tracked"]["correct"] and not x["arms"]["apply"]["correct"]]
    failures=[x["index"] for x in rows if x["arms"]["apply"]["failure"] and not x["arms"]["tracked"]["failure"]]
    benefit=all(totals["apply"][k]<totals["tracked"][k] for k in ("F","Q")) and not losses and not failures
    save(a.out/"complete-six.json",dict(reg=reg,indices=SIX,totals=totals,losses=losses,
         new_failures=failures,benefit_gate=benefit,first_forks=sum(x["first"] is not None for x in rows)))


if __name__=="__main__":
    main()
