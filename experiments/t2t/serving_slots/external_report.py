"""Compare the frozen90 high-first Gemma method with real external runs."""
import argparse
import csv
import json
from pathlib import Path

from bdlm_serving.evaluation.frozen import sha
from bdlm_serving.runtime.optimus_limits import validate_trace
from experiments.t2t.prefix_window.score_run import link_requests, snapshot
from experiments.t2t.serving_slots.switch_report import audit
from tools.prefix_cohorts import load_cases, load_request_set, select_window_cases, _window_row


def read(path):
    return json.loads(path.read_text())


def measured_decoder(row, policy):
    """One request: physical projections and token admissions are separate ledgers."""
    cap = "_cap_" in policy
    if cap:
        validate_trace(row)
    events = row["events"]
    if events[0][0] != "encoder" or any(tag not in ("encoder", "decoder")
                                         for tag, _ in events[1:]):
        raise ValueError("Invalid Gemma stage record")
    observed = [q for tag, q in events[1:] if tag == "decoder"]
    reencode = [q for tag, q in events[1:] if tag == "encoder"]
    if (not observed or any(q != 256 for q in reencode)
            or observed != [s["q"] for s in row["trace"]]
            or sum(observed)+sum(reencode) != row["generation_physical_padded_rows"]
            or len(events)-1 != row["generation_forwards"]):
        raise ValueError("Observed decoder rows differ from reported work")
    work = []
    for s in row["trace"]:
        q, admitted = s["q"], s["accepted"]
        if not 0 <= admitted <= q:
            raise ValueError("Admissions outside projected rows")
        offer = s["offer"] if cap else int(policy.rsplit("_", 1)[1])
        proposed = s["proposed_q"] if cap else q
        work.append((q, admitted, max(0, q-offer), max(0, admitted-offer), proposed-q))
    return work, sum(reencode)


def compare(data, out, capped=None):
    samples = data/"samples.json"
    cases = load_cases(samples)
    digest = sha(samples)
    mixed = "slottrace_256n_256n_128_128"
    cache = out/".inputs"
    own_dir = snapshot(data/"gemma-slots-native256-128",mixed,digest,cache)
    def by_index(folder, p):
        rows, model, policy, _ = load_request_set(folder,cases,"Gemma",p)
        if (model,policy) != ("gemma",p):
            raise ValueError("Wrong model or arm")
        return {r["index"]:r for r in rows}
    own = by_index(own_dir, mixed)
    low = {}
    for i in own:
        row = read(own_dir/f"{i:02d}-{mixed}.json")
        audit(row, ("256n","256n",128,128))
        low[i] = [e["q"] for e in row["events"] if e.get("slot_offer")==128]
    dirs = [(data/"gemma-optimus-reference90",["ibs_128","obs_128"]),
            (data/"gemma-optimus-primary90",["ibs_256","obs_256"]),
            (data/"gemma-wave-reference90",["flow_w1","flow_w2"])]
    if capped:
        meta = read(capped/"run.json")
        if (not meta["hard_q_cap"] or meta["cap_clock"]!="all_decoder_forwards_request_local"
                or meta["cap_excludes"]!="encoder" or meta["actual_token_edits"]!="M2T only"):
            raise ValueError("Wrong capped-mode registration")
        dirs.append((capped,meta["policies"]))
    lines = []
    for folder,policies in dirs:
        meta = read(folder/"run.json")
        if meta["samples_sha256"]!=digest or meta["suite"]!="frozen90":
            raise ValueError(f"Wrong suite for {folder}")
        for p in policies:
            if p not in meta["policies"] or read(folder/f"complete-{p}.json")!=dict(policy=p,requests=90):
                raise ValueError(f"Missing or incomplete {p}")
            rows = by_index(link_requests(folder,p,cache),p)
            is_optimus = p.startswith(("ibs_","obs_"))
            if is_optimus:
                caps = "_cap_" in p
                ledgers = {i:measured_decoder(read(folder/f"{i:02d}-{p}.json"),p) for i in rows}
            for w in (30,60,90):
                subset=select_window_cases(cases,w)
                s=_window_row(subset,rows,own,w,"all","gemma",p,mixed,None)
                limits=[q for r in subset for q in low[r["index"]]]
                s.update(mixed_low_rounds=len(limits),mixed_low_over=sum(q>128 for q in limits),
                         mixed_low_max=max(limits),mixed_low_excess=sum(max(0,q-128) for q in limits))
                if is_optimus:
                    work = [ledgers[r["index"]] for r in subset]
                    qs, edits, over, edits_over, cuts = zip(*(e for steps, _ in work for e in steps))
                    s.update(baseline_decoder_steps=len(qs),baseline_max_decoder_q=max(qs),
                             baseline_decoder_over=sum(x>0 for x in over),baseline_decoder_excess=sum(over),
                             baseline_actual_rows_cut=sum(cuts) if caps else "",
                             baseline_requests_over=sum(any(e[2] for e in steps) for steps, _ in work),
                             baseline_requests_clipped=sum(any(e[4] for e in steps) for steps, _ in work) if caps else "",
                             baseline_decoder_Q=sum(qs),baseline_reencode_Q=sum(q for _, q in work),
                             baseline_max_admitted=max(edits),baseline_total_admitted=sum(edits),
                             baseline_admission_over_steps=sum(x>0 for x in edits_over))
                lines.append(s)
    with (out/"gemma-mixed-vs-external.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in lines for k in r)),lineterminator="\n")
        writer.writeheader()
        writer.writerows(lines)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("data","out","capped"):
        parser.add_argument("--"+key,type=Path,required=key!="capped")
    compare(**vars(parser.parse_args()))
