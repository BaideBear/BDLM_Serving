"""Matched frozen90 Flow W2 projection-only contrast, one H4, raw traces stay remote."""
import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

ROOT = Path("/data1/project_data/gx-erqw-20261123")
DEPLOY = ROOT/"runtime/formal90-7a38ffc"
DATA = ROOT/"runtime/t2t-native-1125/prefix90"
D = ROOT/"results/llada-flow-active-w2-h4-v1"
REG = "a2561d031948307b04f29b95ed9893d7f8fb309c26ba538d8d4dbe03698eb97b"
ARMS = ("flow_w2", "active_w2")


def check(ok, why):
    if not ok:
        raise ValueError(why)


def js(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read(p):
    return json.loads(p.read_bytes())


def save(p, x):
    with p.open("x") as f:
        f.write(js(x)+"\n")


def hashed(x):
    return sha256(js(x).encode()).hexdigest()


def one(h, task, ids, arm):
    from bdlm_serving.methods.flowblock.llada_active_w2 import ActiveW2
    from bdlm_serving.runtime.batch import ForwardBatcher, ModelFacade
    import torch
    m, tok = h["model"], h["tok"]
    batch = ForwardBatcher(m, "llada21", 1)
    facade = ModelFacade(batch, f"{task['index']}/{arm}", task["seed"])
    engine = ActiveW2(facade, h["mask_id"], h["eos_id"], dict(
        confidence_threshold=.5, editing_threshold=.5, max_post_steps=16), compact=arm==ARMS[1])
    prompt = torch.tensor([ids], device=m.device)
    tape = [[] for _ in m.model.layers]
    with ExitStack() as hooks:
        for lay, w in zip(m.model.layers, tape, strict=True):
            hooks.enter_context(lay.attention.query_key_value.register_forward_pre_hook(
                lambda _, args, w=w: w.append(int(args[0].shape[1]))))
        (raw, events, extra), = batch.run([lambda: engine.run(prompt, 2048)])
    gen = raw[0, len(ids):len(ids)+2048].tolist()
    check(torch.equal(raw[:, :len(ids)], prompt), "prompt was modified")
    eos = h["eos_id"]
    cut = gen.index(eos)+1 if eos in gen else len(gen)
    seq = gen[:cut]
    check(len(seq) <= 2048 and (seq[-1] == eos or len(seq)==2048), "EOS handling")
    special = set(tok.all_special_ids) | {eos, h["mask_id"]}
    return dict(index=task["index"], task_id=task["task_id"], dataset=task["dataset"],
                seed=task["seed"], arm=arm, upstream_sources=extra["provenance"], official=extra["official_metrics"],
                official_return_length=extra["official_return_length"], prompt_ids=ids,
                output_ids=seq, output_text=tok.decode(seq, skip_special_tokens=True),
                N=sum(t not in special for t in seq), eos=seq[-1]==eos, cap=2048,
                remaining_masks=seq.count(h["mask_id"]), layers=tape, calls=facade.calls,
                events=events, ticks=engine.ticks)


def audit(r, task):
    from bdlm_serving.evaluation.scoring import score
    dense = r["arm"] == ARMS[0]
    calls, ev, ticks = r["calls"], r["events"], r["ticks"]
    widths = [c["physical_rows"] for c in calls]
    check(len(r["layers"]) == 20 and all(w == widths for w in r["layers"]) and
          widths == [e["q_tokens"] for e in ev] == [c["rows"] for c in calls] and
          all(c["batch"]==1 for c in calls), "native 20-layer physical bill")
    check(ev[0]["phase"] == "prefill" and len(ev)==len(ticks)+1 and
          widths[0] == max(32,len(r["prompt_ids"])//32*32), "prefill")
    one_slot = 0
    for t, e, c in zip(ticks, ev[1:], calls[1:], strict=True):
        n = t["open"]
        w = 64 if dense else 32*n
        check(n in (1,2) and len(t["slots"]) == n and t["written_KV"] == t["physical_width"] ==
              e["q_tokens"] == c["physical_rows"] == w and
              e["absolute_positions"][:n*32] == list(range(t["head"],t["head"]+n*32)) and
              e["kv_length"] == c["kv_rows"], "paid width/actual open blocks")
        one_slot += n == 1
    f = len(ticks)
    metrics, = r["official"]
    check(f == metrics["total_forward_calls"] == metrics["steps"] and metrics["window_size"]==2
          and metrics["spawn_signal"] == "acceptance_ratio" and metrics["spawn_threshold"]==.8 and
          metrics["block_length"]==32 and r["remaining_masks"]==0, "upstream decode")
    return dict(N=r["N"], F=f, Q=sum(widths[1:]), prefill_Q=widths[0], one=one_slot, two=f-one_slot,
                correct=bool(score(r["output_text"],task)["correct"]), eos=r["eos"],
                ids_sha256=hashed(r["output_ids"]))


def compare(a,b):
    for j, (u,v) in enumerate(zip(a["ticks"],b["ticks"])):
        check(all(u[k]==v[k] for k in ("head","commit","right","open")), "unexplained gate fork")
        for s,t in zip(u["slots"],v["slots"],strict=True):
            check(all(s[k]==t[k] for k in ("ids","prompt","pe0","fin0")), "incoming state fork")
        if u["slots"] != v["slots"]:
            for slot,(s,t) in enumerate(zip(u["slots"],v["slots"],strict=True)):
                if s!=t:
                    ids = [(p,k,l) for p,(k,l) in enumerate(zip(s["new"],t["new"],strict=True)) if k!=l]
                    return dict(tick=j,slot=slot,open=u["open"],head=u["head"],
                                first_token=ids[:1],frontier=[s["frontier"],t["frontier"]],
                                finished=[s["fin"],t["fin"]],post=[s["pe"],t["pe"]])
    check(len(a["ticks"])==len(b["ticks"]) and a["output_ids"]==b["output_ids"],
          "unexplained terminal fork")
    return None


def main():
    from experiments.t2t.serving_slots import shared_fixed_shard as s, shared_llada_worker as w
    from bdlm_serving.methods import flowblock
    from bdlm_serving.methods.flowblock import control
    from bdlm_serving.evaluation.scoring import score
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("check","run","report"))
    mode=p.parse_args().mode
    if (D/"llada_active_w2.py").is_file():
        flowblock.__path__.insert(0,str(D))
    from bdlm_serving.methods.flowblock.llada_active_w2 import PACKED
    base=read(DEPLOY/"registration.json")
    s.check_sources(base)
    reg=read(D/"registration.json")
    check(sha256((D/"registration.json").read_bytes()).hexdigest()==REG and
          sha256((DATA/"samples.json").read_bytes()).hexdigest()==reg["samples_sha256"], "registration")
    controls=ROOT/"sources/flowblock-controls"
    check(sha256((controls/"flowblock_packed.py").read_bytes()).hexdigest()==PACKED and
          os.environ.get("PACKED_FULL_CACHE","0")!="1","upstream source/cache")
    tasks=s.load_tasks(DATA/"samples.json")
    ledger=read(DATA/"fixed-family-reference-case-ledger-v1.json")
    ls={r["index"]:r for r in ledger["arms"]["llada21:flow_w2"]["items"]}
    check(len(ls)==90 and ledger["scorer"]==w.SCORER_SHA,"reference ledger/scorer")
    order=reg["pilot"]+[i for i in range(90) if i not in reg["pilot"]]
    check(len(set(order))==len(order)==90, "indices")
    if mode=="check":
        print("frozen90 panel, source and adapter available")
        return
    if mode=="run":
        check(not any(D.glob("??-*.json")), "run directory not empty")
        s.check_board(reg["board"])
        h=w.load(Path("/data0/ysy/models/LLaDA2.1-mini"))
    pairs=[]
    with patch.object(control,"control_directory", lambda _: controls):
        for k,i in enumerate(order):
            task=tasks[i]
            old=read(DATA/f"llada21-reference90-wave/{i:02d}-flow_w2.json")
            check(all(task[f]==old[f] for f in ("seed","index","task_id","dataset")) and
                  old["generation_cap"]==2048 and old["policy"]=="flow_w2", "historical identity")
            hist_score=bool(score(old["output_text"],task)["correct"])
            rows,stats={},{}
            for arm in (ARMS if k%2==0 else ARMS[::-1]):
                path=D/f"{i:02d}-{arm}.json"
                if mode=="run":
                    check(time.time()<reg["deadline"]-90, "deadline")
                    ids,_=w.canonical_prompt(h["tok"],task,old["prompt_ids"])
                    r=one(h,task,ids,arm)
                    save(path,r)
                else:
                    if not path.exists():
                        return
                    r=read(path)
                check(r["arm"]==arm and all(r[f]==task[f] for f in ("index","seed","task_id","dataset"))
                      and r["prompt_ids"]==old["prompt_ids"] and r["upstream_sources"]==old["extra"]["provenance"],
                      "case/source identity")
                rows[arm],stats[arm]=r,audit(r,task)
            a,b=(rows[arm] for arm in ARMS)
            x,y=(stats[arm] for arm in ARMS)
            history=(old["status"]=="success" and not old["request_failure"] and
                     a["output_ids"]==old["output_ids"] and a["output_text"]==old["output_text"] and
                     all(a[f]==ls[i][f] for f in ("index","task_id","dataset","seed")) and
                     all(x[f]==ls[i][f] for f in ("N","F","Q","correct")) and
                     old["generation_physical_padded_rows"]==x["Q"] and x["prefill_Q"]==old["prefill_rows"] and
                     a["official"]==old["extra"]["official_metrics"]
                     and a["official_return_length"]==old["extra"]["official_return_length"] and
                     (x["N"],x["F"],x["Q"],x["eos"],x["correct"])==(old["output_tokens"],
                     old["generation_forwards"],old["generation_query_rows"],old["eos"],hist_score))
            if history:
                check(all(all(e[f]==z[f] for f in ("absolute_positions","kv_length","q_tokens"))
                          for e,z in zip(a["events"][1:],old["events"][1:],strict=True)), "archived physical rows")
            pair=dict(index=i,seed=task["seed"],task_id=task["task_id"],dataset=task["dataset"],**stats,
                      reg=REG,prompt_len=len(a["prompt_ids"]),prompt_sha256=hashed(a["prompt_ids"]),
                      first_decision_fork=compare(a,b),ids_equal=a["output_ids"]==b["output_ids"],
                      fresh_matches_archive=history)
            pairs.append(pair)
            print(js(pair),flush=True)
            if k>=6:
                check(history, "fresh control differs from historical reference")
            if k==5:
                check(all(z["fresh_matches_archive"] for z in pairs) and
                      any(z["prompt_len"]%32 for z in pairs) and
                      all(sum(z["active_w2"][v] for z in pairs)>0 for v in ("one","two","eos")),
                      "pilot gate failed: do not extend")
                if mode=="run":
                    save(D/"qualified.json",dict(cases=order[:6],pairs_sha256=hashed(pairs),reg=REG))
    if mode=="run":
        save(D/"complete.json",dict(pairs=90,requests=180,reg=REG,pairs_sha256=hashed(pairs)))


if __name__=="__main__":
    main()
