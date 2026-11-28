"""H4-only instrumented replay of two diagnostic losses; no policy changes."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

BASE = Path("/data1/project_data/gx-erqw-20261123")
DEPLOY = BASE / "runtime/formal90-7a38ffc"
DATA = BASE / "runtime/t2t-native-1125/prefix90"
FROZEN = BASE / "results/formal90-7a38ffc/H4/llada21/receipts.ndjson"
VERSION = "a18f108aef1dbc607f3b21b2a4d169b04f5ae5dab01be400d015d07249d406f3"
LEDGER = "ff9caccbf776416f0237ddcc73af774fbe567c57929b359583f70fba128d64ce"
CASES = (1, 67)  # Lost cases in the H4 shard; partial vs exactly aligned prompt.


def digest(x):
    return sha256(json.dumps(x, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def check(ok, why):
    if not ok:
        raise ValueError(why)


def save(path, obj):
    with path.open("x") as f:
        json.dump(obj, f, separators=(",", ":"), allow_nan=False)


def controls(name):
    check(name == "Optimus", "wrong IBS upstream")
    p = BASE / "sources/optimus-controls"
    hashes = {"sequence.py": "99c80047d517113579e04fd57443471a759a738dc30b206ba1342e0d4f9b3da2",
              "unmasking.py": "bbc3c3328e5497aa08aed6e04150ab0383f95dc06b4ff570a69adaed4a769746"}
    for k, v in hashes.items():
        check(sha256((p / k).read_bytes()).hexdigest() == v, f"IBS source mismatch: {k}")
    return p


def normalized(a, b, prompt):
    # A written IBS token is the actual input when that debt is next paid.
    tape = []
    pending = {}
    for e, feed in zip(a["events"], a["forward_inputs"], strict=True):
        if e["phase"] == "prefill":
            continue
        rows = e["absolute_positions"]
        for pos, token in zip(rows, feed, strict=True):
            if pos in pending:
                tape[pending.pop(pos)]["writes"].append([pos, token])
        masks = [p for p, m in zip(rows, e["selected_masked"], strict=True) if m]
        tick = len(tape)
        tape.append(dict(rows=rows, mask=masks, editable=[p for p in rows if p>=prompt and p not in masks],
                         debt=e["dirty_before"], writes=[]))
        for p in e["written_absolute"]:
            pending[p] = tick
    check(not pending, "IBS writes without a physical next input")
    for event in tape:
        event["writes"].sort()
    shared = []
    for e in b["trace"]:
        pred = {p: v[1] for p, v in zip(e["sampled"], e.get("proposals", []), strict=True)}
        writes = sorted([p, pred[p]] for p in e["written_absolute"])
        shared.append(dict(rows=e["rows"], mask=[p for p in e["sampled"] if p in set(e["mask"]+e["cold"])],
                           editable=[p for p in e["sampled"] if p not in set(e["mask"]+e["cold"])],
                           debt=e["dirty_before"], writes=writes))
    return tape, shared


def compact(a, b, eos, idx):
    x, y = normalized(a, b, len(a["prompt_ids"]))
    diffs = {}
    for k in ("rows", "mask", "editable", "writes"):
        j = next((j for j, (u, v) in enumerate(zip(x, y)) if u[k] != v[k]), None)
        diffs[k] = None if j is None else dict(tick=j, ibs=x[j], shared=y[j])
    stem = 0
    for l, r in zip(a["output_ids"], b["output_ids"]):
        if l != r:
            break
        stem += 1
    def terminal(t, ids):
        return dict(output_len=len(ids), digest=digest(ids), last_id=ids[-1],
                    first_eos_write=next(([j,p] for j,e in enumerate(t) for p,v in e["writes"] if v==eos),None),
                    last_tick=len(t)-1, last_event=t[-1],
                    delivered_eos_offset=ids.index(eos) if eos in ids else None)
    return dict(index=idx, task_id=a["task_id"], seed=a["seed"], input_sha256=b["input_sha256"],
                mask_id=b["observed_mask_id"], eos_id=eos, prefill=[a["prefill_rows"], b["prefill_Q"]],
                first=diffs, first_output_difference=[stem,a["output_ids"][stem:stem+1], b["output_ids"][stem:stem+1]],
                ibs_terminal=terminal(x,a["output_ids"]), shared_terminal=terminal(y,b["output_ids"]),
                ibs_bill=[a["output_tokens"],a["generation_forwards"],a["generation_query_rows"]],
                shared_bill=[b[k] for k in ("N","F","Q")],correct=[True,b["correct"]],
                first_ibs=x[:5], first_shared=y[:5], ibs_events_sha256=digest(a["events"]),
                shared_events_sha256=digest(b["trace"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("check","run","report"))
    ap.add_argument("output",type=Path)
    arg=ap.parse_args()
    from experiments.t2t.serving_slots import shared_fixed_shard as s, shared_llada_worker as w, ibs_prefix_pair as pair
    from bdlm_serving.methods.optimus import upstream
    from bdlm_serving.methods.t2t.unified_rows import Params
    from bdlm_serving.evaluation import scoring

    base = json.loads((DEPLOY/"registration.json").read_bytes())
    s.check_sources(base)
    check(s.identity(base)==VERSION and base["params"]["llada21"]=={"prefix_share":"1/8","open_at":"1/4"}, "release")
    controls("Optimus")
    ledger = (DATA/"fixed-family-reference-case-ledger-v1.json").read_bytes()
    check(sha256(ledger).hexdigest()==LEDGER, "wrong IBS ledger")
    ledger = json.loads(ledger)["arms"]["llada21:optcap_ibs_4"]["items"]
    tasks = s.load_tasks(DATA/"samples.json")
    formal = {r["index"]:r for r in map(json.loads,FROZEN.read_bytes().splitlines()) if r["offer"]==4}
    check(len(formal)==15, "incomplete formal H4")
    for i in CASES:
        f=formal[i]
        check((f["version"],f["board"],f["stage"],f["model"],f["correct"],ledger[i]["correct"])==
              (VERSION,"n164:H4","formal90","llada21",False,True),"not registered H4 loss")
        check(all(f[k]==ledger[i][k]==tasks[i][k] for k in ("index","dataset","task_id","seed")), "case identity")
    if arg.action=="check":
        return
    if arg.action=="run":
        for i in CASES:
            for arm in ("ibs","shared"):
                check(not (arg.output/f"{i}-{arm}.json").exists(),"existing trajectory")
        s.check_board("n164:H4")
        h=w.load(Path("/data0/ysy/models/LLaDA2.1-mini"))
    for i in CASES:
        paths=[arg.output/f"{i}-{arm}.json" for arm in ("ibs","shared")]
        old=json.loads((DATA/f"llada21-optimus-hardcap-low/{i:02d}-optcap_ibs_4.json").read_bytes())
        if arg.action=="run":
            prompt,_=w.canonical_prompt(h["tok"], tasks[i], old["prompt_ids"])
            check(digest(prompt)==formal[i]["input_sha256"],"prompt ids")
            original=w.evaluate_fixed
            captured=[]
            def tap(*args, **kwargs):
                r=original(*args, **kwargs)
                captured.append(r)
                return r
            with patch.object(w,"evaluate_fixed",tap):
                r=w.evaluate(h,tasks[i],4,Params())
            check(all(r[k]==formal[i][k] for k in ("N","F","Q","correct","input_sha256","status",
                      "encoder_F","decoder_F","decoder_Q","roles","max_decoder_Q","failure","prefill_Q")),"formal replay")
            b,=captured
            b.update(observed_mask_id=h["mask_id"], observed_eos_id=h["eos_id"])
            save(paths[1],b)
            with patch.object(pair,"Q",4), patch.dict(pair.LABELS,{0:"optcap_ibs_4"}), \
                 patch.object(upstream,"control_directory",controls):
                a=pair.generate(h["model"],h["tok"],tasks[i],prompt,2048,0)
            save(paths[0],a)
        else:
            a,b=[json.loads(p.read_bytes()) for p in paths]
        check(all(b[k]==formal[i][k] for k in ("N","F","Q","correct","input_sha256","prefill_Q",
                                                "max_decoder_Q","failure")), "shared bill")
        check(all(a[k]==old[k] for k in ("output_ids","prompt_ids","generation_forwards",
                 "generation_query_rows","output_tokens","eos","truncated","events","extra")), "IBS replay")
        check(scoring.score(a["output_text"],tasks[i])["correct"]==ledger[i]["correct"] and
              [ledger[i][k] for k in ("N","F","Q")]==[a[k] for k in ("output_tokens","generation_forwards",
                "generation_query_rows")], "IBS score or bill")
        summary=compact(a,b,b["observed_eos_id"],i)
        print(json.dumps(summary,separators=(",",":")),flush=True)


if __name__=="__main__":
    main()
