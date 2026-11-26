"""Compare the frozen90 high-first Gemma method with real external runs."""
import argparse
import csv
import json
from pathlib import Path

from bdlm_serving.evaluation.frozen import sha
from experiments.t2t.prefix_window.score_run import link_requests, snapshot
from experiments.t2t.serving_slots.switch_report import audit
from tools.prefix_cohorts import load_cases, load_request_set, select_window_cases, _window_row


def read(path):
    return json.loads(path.read_text())


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
            for w in (30,60,90):
                subset=select_window_cases(cases,w)
                s=_window_row(subset,rows,own,w,"all","gemma",p,mixed,None)
                limits=[q for r in subset for q in low[r["index"]]]
                s.update(mixed_low_rounds=len(limits),mixed_low_over=sum(q>128 for q in limits),
                         mixed_low_max=max(limits),mixed_low_excess=sum(max(0,q-128) for q in limits))
                if p.startswith(("ibs_","obs_")):
                    cap = "_cap_" in p
                    events=[e for r in subset for e in read(folder/f"{r['index']:02d}-{p}.json")["trace"]]
                    allowances=[e["offer"] if cap else int(p.rsplit("_",1)[1]) for e in events]
                    excess=[max(0,e["q"]-b) for e,b in zip(events,allowances,strict=True)]
                    s.update(baseline_decoder_steps=len(events),baseline_max_decoder_q=max(e["q"] for e in events),
                             baseline_decoder_over=sum(x>0 for x in excess),baseline_decoder_excess=sum(excess),
                             baseline_actual_rows_cut=sum(e["proposed_q"]-e["q"] for e in events) if cap else "")
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
