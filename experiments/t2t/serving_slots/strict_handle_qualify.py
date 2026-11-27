"""CUDA qualification of a public single-request API, not a policy ranking."""
import argparse
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path
import time

from bdlm_serving.methods.t2t.serving_profiles import open_strict_gemma

TIERS = (32,64,128,256,512)


def offer_cycle(tiers=TIERS):
    # Euler cycle of directed offer pairs, including the five self transitions.
    arcs = {q:list(reversed(tiers)) for q in tiers}
    path, finished = [tiers[0]], []
    while path:
        q = path[-1]
        if arcs[q]:
            path.append(arcs[q].pop())
        else:
            finished.append(path.pop())
    return tuple(reversed(finished))[:-1]


def reference_path(root, q, index, profile="fixed90_candidate"):
    if profile in ("fixed90_fused","fixed90_dense_fused"):
        if q in (128,512):
            return root/"profile-close-1128"/f"C{q}-profile-fuse1-{index:02}.json"
        if q==256 and profile=="fixed90_dense_fused":
            return root/"capacity-head-1128"/f"C256-capacity1-{index:02}.json"
        return root/"paid-close-1128"/f"C{q}-fuse1-{index:02}.json"
    if q == 128:
        return root/"cross-race-1128"/f"C128-gateoff-slack0-4-pref1of4-{index:02}.json"
    if q == 512:
        return root/"cross-race-1128"/f"C512-gatehalf-slack0-4-pref1of8-{index:02}.json"
    return root/"prefix-rank-1128"/f"C{q}-rankright-reuse0-{index:02}.json"


def check(ok, message):
    if not ok:
        raise ValueError(message)


def drive(model, data, task, schedule, *, deadline, profile="fixed90_candidate"):
    from bdlm_serving.runtime.diffusiongemma import observe
    from .gemma_strict_eval import encoder_tape
    from .gemma_unified_audit import projectors
    from .mask_pair_gemma import input_digest
    origin = data["input_ids"].shape[1]
    debt, ages, ledger = set(), {}, [("encoder", origin)]
    transitions, routes = Counter(), Counter()
    old, head, verify, encoded, d_q = None, 0, 0, 0, 0
    parked_pay = physical_downshift = 0
    cfg = dict(seed=task["seed"], initial_offer=schedule[0], profile=profile)
    with ExitStack() as stack:
        events = stack.enter_context(observe(model))
        proj, positions = stack.enter_context(projectors(model))
        enc = stack.enter_context(encoder_tape(model))
        h = open_strict_gemma(model, data, **cfg)
        stack.callback(h.close)
        check(h.opened_prefill_q == origin and events == ledger, "open prefill")
        tick = 0
        while h.status == "ready":
            check(time.time() < deadline and tick < 12000, "deadline or no bounded progress")
            q = schedule[tick % len(schedule)]
            e = h.step(q)
            rows = positions[-1]
            sampled, changed = set(e["sampled_absolute"]), set(e["written_absolute"])
            check(e["tick"] == tick and e["ordinal"] == head and e["positions"] == rows
                  and len(rows) == e["q"] == e["decoder_Q"] == e["history_Q"] <= q
                  and len(set(rows)) == len(rows)
                  and all(t[-1] == len(rows) for t in proj.values()), "unmeasured physical projection")
            check(e["dirty_before_absolute"] == sorted(debt) and set(sorted(debt)[:q]) <= set(rows)
                  and changed <= sampled <= set(rows), "unpaid old or unmeasured new writes")
            if old is not None:
                transitions[old,q] += 1
                if old > q:
                    physical_downshift += int(bool(debt))
                    parked_pay += sum(p >= origin+256*(head+1) for p in debt.intersection(rows))
            for p in e["parts"]:
                ordinal = p["ordinal"]
                check(p["age_before"] == ages.get(ordinal, 0) and
                      p["age_after"] == p["age_before"] + bool(p["sampled"]), "sampler clock reset")
                ages[ordinal] = p["age_after"]
                if p["phase"] == "verify":
                    if e["route"]=="verify_paid":
                        check(profile!="fixed90_candidate" and verify==0 and len(e["parts"])==1
                              and debt and debt<=set(rows) and e["head_initialized_before"]==256
                              and e["promoted"] is None, "unpaid or reordered fused verification")
                    else:
                        check(not debt,"pending write on plain verification")
                    check(ordinal == head and not sampled
                          and p["rows"] == list(range(verify, verify+len(p["rows"]))), "terminal sweep")
                    verify += len(p["rows"])
            debt = (debt - set(rows)) | changed
            check(sorted(debt) == e["dirty_after_absolute"], "lost dirty across offers")
            ledger.append(("decoder",len(rows)))
            if e["encoded"]:
                check(verify == 256 and e["encoder_Q"] == 256, "encoder before verification")
                ledger.append(("encoder",256))
                head += 1
                encoded += 1
                verify = 0
            routes[e["route"]] += 1
            tick, old, d_q = tick+1, q, d_q+len(rows)
        ids = h.result()
        check(verify == 256 and len(proj) == 85 and events == ledger and len(enc) == encoded+1
              and enc[0][0] == data["input_ids"][0].tolist() and not debt
              and len(positions) == tick and all(len(t) == tick for t in proj.values()), "close/accounting")
        for j,(block, pos) in enumerate(enc[1:]):
            check(list(ids[j*256:(j+1)*256]) == block
                  and pos == list(range(origin+256*j, origin+256*(j+1))), "bad committed encoder input")
    return dict(input_sha256=input_digest(data), index=task["index"], seed=task["seed"],
                task_id=task["task_id"], dataset=task["dataset"], F=tick+encoded, decoder_F=tick,
                decoder_Q=d_q, encoder_F=encoded, encoder_Q=256*encoded, routes=dict(routes),
                directed_pairs={f"{a}>{b}":v for (a,b),v in sorted(transitions.items())},
                downshifts_with_debt=physical_downshift, parked_rows_paid_on_downshift=parked_pay,
                decoder_projection_modules=len(proj), output_digest=sha256(
                    json.dumps(ids,separators=(",",":")).encode()).hexdigest()), ids


def main(argv=None):
    import torch
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from .mask_pair_gemma import SAMPLES_SHA
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint","samples","receipts","out"):
        ap.add_argument("--"+key, type=Path, required=True)
    ap.add_argument("--profile",choices=("fixed90_candidate","fixed90_fused","fixed90_dense_fused"),
                    default="fixed90_candidate")
    ap.add_argument("--deadline",type=int,required=True)
    a = ap.parse_args(argv)
    check(torch.cuda.is_available() and torch.cuda.device_count()==1 and
          sha256(a.samples.read_bytes()).hexdigest()==SAMPLES_SHA, "GPU-only frozen qualification")
    tasks = load_tasks(a.samples)
    a.out.mkdir(parents=True,exist_ok=True)
    proc = AutoProcessor.from_pretrained(a.checkpoint, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(a.checkpoint,local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa", experts_implementation="grouped_mm",
        device_map={"":"cuda:0"}).eval()
    jobs = [(f"fixed{q}",(q,)) for q in TIERS] + [("switch25",offer_cycle()),
                                                    ("park", (512,)*5+(32,)*3+(512,))]
    with torch.inference_mode():
        for i in (0,10,20):
            t = tasks[i]
            data = proc.apply_chat_template([{"role":"user","content":t["prompt"]}], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
            for name, schedule in jobs:
                dst = a.out/f"{name}-{i:02}.json"
                if dst.exists():
                    previous = json.loads(dst.read_bytes())
                    check(previous.get("profile","fixed90_candidate")==a.profile, "wrong profile in destination")
                    continue
                row, ids = drive(model,data,t,schedule,deadline=a.deadline,profile=a.profile)
                if len(schedule)==1:
                    src = reference_path(a.receipts,schedule[0],i,a.profile)
                    blob = src.read_bytes()
                    ref = json.loads(blob)
                    same = list(ids)==ref["output_ids"] and all(row[k]==ref[k] for k in
                            ("index","seed","task_id","dataset","input_sha256","F","decoder_F","encoder_F","decoder_Q","encoder_Q"))
                    row.update(reference_sha256=sha256(blob).hexdigest(), identical=same)
                row.update(schema="strict-handle-gemma-qualification-v1", suite=name, profile=a.profile, samples_sha256=SAMPLES_SHA,
                           gpu=torch.cuda.get_device_name(), run_pid=os.getpid(), timestamp=time.time())
                tmp = dst.with_suffix(".tmp")
                tmp.write_text(json.dumps(row,sort_keys=True)+"\n")
                tmp.replace(dst)
                print(name,i,row["decoder_F"],row.get("identical"),flush=True)
                check(row.get("identical",True), "fixed decoder differs from frozen control")


def llada_main(argv=None):
    """Separate pinned environment. Admission/edits and changed inputs stay on GPU."""
    import torch
    from .cross_race_llada import load_gpu
    from .llada_fixed_family_eval import fixed_cross, TIERS as slots
    from .mask_pair_gemma import SAMPLES_SHA
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.methods.t2t.serving_profiles import open_strict_llada
    from bdlm_serving.methods.t2t.budget_family import Law
    ap = argparse.ArgumentParser(description=llada_main.__doc__)
    for k in ("checkpoint","samples","reference","out"):
        ap.add_argument("--"+k, type=Path, required=True)
    ap.add_argument("--deadline",type=int,required=True)
    a = ap.parse_args(argv)
    check(torch.cuda.is_available() and torch.cuda.device_count()==1 and
          sha256(a.samples.read_bytes()).hexdigest()==SAMPLES_SHA, "GPU-only frozen qualification")
    tok, model = load_gpu(a, lambda:None)
    a.out.mkdir(parents=True,exist_ok=True)
    tasks = load_tasks(a.samples)
    with torch.inference_mode():
        for i in (0,10,20):
            t = tasks[i]
            ref = json.loads((a.reference/f"{i:02d}-native32.json").read_text())
            check(ref["index"]==i and ref["generation_cap"]==2048, "unregistered LLaDA reference")
            x = torch.tensor([ref["prompt_ids"]],device=model.device)
            jobs = [(f"fixed{q}", (q,)) for q in slots] + [
                ("switch25", offer_cycle(slots)), ("park",(64,)*5+(4,)*3+(64,))]
            for name, schedule in jobs:
                dst = a.out/f"{name}-{i:02}.json"
                if dst.exists():
                    continue
                kwargs = dict(seed=t["seed"], mask_id=tok.convert_tokens_to_ids("<|mask|>"),
                              eos_ids=[tok.eos_token_id], coverage="head",top_gate="half")
                debt, widths, pos, directed = set(), [], [], Counter()
                parked, switches, clock, d_q, tick, old = 0, 0, 0, 0, 0, None
                def hook(_m, _args, kw):
                    p=kw["position_ids"][0]
                    mask=kw["attention_mask"][0,0]
                    expected=p[:,None]//32 >= torch.arange(mask.shape[1],device=p.device)[None,:]//32
                    check(torch.equal(mask,torch.zeros_like(mask).masked_fill(
                        ~expected,torch.finfo(mask.dtype).min)), "wrong physical block mask")
                    pos.append(p.tolist())
                with ExitStack() as hooks:
                    hooks.enter_context(model.model.layers[0].attention.register_forward_pre_hook(hook,with_kwargs=True))
                    hooks.enter_context(model.lm_head.register_forward_pre_hook(
                        lambda _,args:widths.append(args[0].shape[1])))
                    h = open_strict_llada(model,x,**kwargs)
                    hooks.callback(h.close)
                    check(pos==[list(range(x.shape[1]))], "wrong prompt prefill")
                    while h.status=="ready":
                        check(time.time()<a.deadline, "deadline")
                        q=schedule[tick % len(schedule)]
                        e=h.step(q)
                        rows=set(e["rows"])
                        check(e["rows"]==pos[-1] and e["q"]==len(rows)==widths[-1]<=q
                              and e["layer_physical_q"]==[len(rows)]*20 and e["encoder_F"]==0
                              and e["dirty_before"]==sorted(debt) and set(sorted(debt)[:q])<=rows,
                              "wrong projection or debt")
                        if old is not None:
                            directed[old,q]+=1
                            if old>q:
                                switches+=bool(debt)
                                parked+=sum(p >= (e["block"]+1)*32 for p in rows.intersection(debt))
                        clock+=bool(e["sampled"])
                        debt=(debt-rows)|set(e["written_absolute"])
                        check(e["dirty_after"]==sorted(debt) and e["sample_clock"]==clock
                              and set(e["written_absolute"])<=set(e["sampled"])<=rows,
                              "dirty or sampler clock mismatch")
                        tick, d_q, old = tick+1, d_q+len(rows), q
                    ids=h.result()
                    check(not debt and len(pos)==len(widths)==tick+1, "unaccounted close")
                row = dict(schema="strict-handle-llada-qualification-v1", suite=name, index=i, seed=t["seed"],
                           dataset=t["dataset"], task_id=t["task_id"], samples_sha256=SAMPLES_SHA,
                           F=tick, decoder_F=tick, decoder_Q=d_q, encoder_F=0, encoder_Q=0,
                           directed_pairs={f"{a}>{b}":v for (a,b),v in sorted(directed.items())},
                           downshifts_with_debt=switches, parked_rows_paid_on_downshift=parked,
                           decoder_projection_modules=20, gpu=torch.cuda.get_device_name(),
                           timestamp=time.time(), run_pid=os.getpid(),
                           output_digest=sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest())
                if len(schedule)==1:
                    q=schedule[0]
                    fixed=fixed_cross(model,tok,t,ref["prompt_ids"],q,Law(),mode="half" if q==64 else "head",
                                      borrow=False,coverage="head" if q==64 else "sparse")
                    row["identical"]=(list(ids)==fixed["output_ids"] and fixed["F"]==tick
                                      and fixed["Q"]==d_q and pos[1:]==[e["rows"] for e in fixed["trace"]])
                tmp=dst.with_suffix(".tmp")
                tmp.write_text(json.dumps(row,sort_keys=True)+"\n")
                tmp.replace(dst)
                print(name,i,tick,row.get("identical"),flush=True)
                check(row.get("identical",True),"fixed LLaDA positions/output differ from control")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model", choices=("gemma", "llada"), default="gemma")
    options, rest = parser.parse_known_args()
    (main if options.model == "gemma" else llada_main)(rest)
