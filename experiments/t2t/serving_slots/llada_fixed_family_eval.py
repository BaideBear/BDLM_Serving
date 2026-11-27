"""Independent frozen90 LLaDA requests at constant physical decoder caps."""
import argparse
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace

from bdlm_serving.evaluation import scoring
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.methods.t2t.budget_family import Law
from . import llada_all_slots90 as reg
from .llada_mask_pilot import CKPT
from .mask_probe_llada import verify_checkpoint
from .top_transition90 import resolve_reference, validate_prompts

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
TIERS = (4, 8, 16, 32, 64)
ORDER = (0, 10, 20, *(i for i in range(90) if i not in (0,10,20)))
SCORER_SHA = "a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692"
OUT = reg.DATA/"llada21-fixed-family-v1"


def fixed(model, tok, task, prompt, q, law, *, ablation="00"):
    """Registered batch-one oracle; constructing the request always starts a new bank."""
    if ablation not in ("00", "10", "11") or q not in TIERS:
        raise ValueError("Expected a registered ablation and fixed offer")
    return _fixed(model, tok, task, prompt, q, law, ablation=ablation)


def _fixed(model, tok, task, prompt, q, law, *, ablation=None, tail_rule="eager"):
    import torch
    from bdlm_serving.methods.t2t.llada_budget_request import LLaDABudgetRequest

    mask, eos = tok.convert_tokens_to_ids("<|mask|>"), tok.eos_token_id
    x = torch.tensor([prompt],device=model.device)
    attention, sampled, head_width, retired_ids = [], [], [], []
    stats, phases = Counter(), Counter()
    origin = len(prompt)
    failure = None

    def at_layer(_, args, kw):
        pos, bias = kw["position_ids"][0], kw["attention_mask"][0,0]
        cols = torch.arange(bias.shape[-1],device=pos.device)
        allow = pos[:,None]//32 >= cols[None,:]//32
        reg.require(torch.equal(bias,torch.zeros_like(bias).masked_fill(~allow,torch.finfo(bias.dtype).min)),
                    "First layer received a non-causal cross-block mask")
        attention.append(pos.tolist())

    with torch.inference_mode(), ExitStack() as ctx:
        ctx.enter_context(model.model.layers[0].attention.register_forward_pre_hook(at_layer,with_kwargs=True))
        ctx.enter_context(model.lm_head.register_forward_pre_hook(lambda _,args: head_width.append(args[0].shape[1])))
        req = LLaDABudgetRequest(model,x,mask_id=mask,eos_ids=[eos],max_new_tokens=reg.CAP,
                                seed=task["seed"],law=law,confidence_threshold=.9,editing_threshold=.5,
                                tail_rule=tail_rule, ablation=ablation)
        reg.require(attention == [list(range(origin))] and head_width == [origin],"Wrong pure prompt prefill")
        debt = set()
        while req.status == "ready":
            lo = req.head
            try:
                e = req.step(q)
            except RuntimeError as exc:
                # A nonconvergent request has zero delivered output. Other failures
                # (bad physical proof, CUDA or loader errors) invalidate the run.
                if str(exc) != "No head retirement within 258 paid calls; not a valid completion":
                    raise
                failure = str(exc)
                break
            rows = set(e["rows"])
            w = len(rows)
            build = e["phase"] == "build_sample"
            if build:
                start = e["rows"][0]
                reg.require(ablation == "11" and q < 64 and not debt and start % 32 == 0 and
                            start in (lo, lo+32) and e["rows"] == list(range(start,min(start+32,req.end)))
                            and e["cold"] == e["sampled"] == e["rows"], "Invalid whole-block decode")
            reg.require(e["tick"] == len(attention)-2 and e["rows"] == attention[-1] and
                        e["q"] == head_width[-1] == w <= (32 if build else q) and e["offer"] == q and
                        e["layer_physical_q"] == [w]*20 and e["dirty_before"] == sorted(debt) and
                        rows & debt == set(sorted(debt)[:q]),"Decode-width or debt accounting mismatch")
            writes = set(e["written_absolute"])
            reg.require(writes == set(e["admitted"]+e["edited"]+e["revoked"]) and
                        writes <= set(e["sampled"]) <= rows,"Unmeasured update")
            debt = (debt-rows) | writes
            reg.require(debt == set(e["dirty_after"]), "Lost or fabricated debt")
            for block in e["retired_blocks"]:
                start, end = max(origin,32*block), min(req.end,32*block+32)
                ids = req.x[0,start:end].tolist()
                reg.require(not debt.intersection(range(start,end)) and mask not in ids and
                            end <= req.bank.last_end, "Premature retirement")
                retired_ids.extend(ids[:ids.index(eos)+1] if eos in ids else ids)
            pref = set(e["prefix"])
            stats.update(dirty=len(rows & set(e["dirty_before"])), prefix=len(pref),
                         prefix_edits=len(pref & set(e["edited"])), edited=len(e["edited"]),
                         m2t=len(e["admitted"]), revoked=len(e["revoked"]), cold=len(e["cold"]),
                         clean_unresolved=len(e["mask"]), dual_block_F=int(any(r<lo+32 for r in rows)
                         and any(r>=lo+32 for r in rows)))
            phases.update([e["phase"]])
            sampled.append(e)
    gen = attention[1:]
    widths = list(map(len,gen))
    reg.require(widths == head_width[1:] and widths and widths == [e["q"] for e in sampled],
                "Unpaid projection")
    if failure:
        reg.require(len(widths)==len(sampled) and req.status=="failed","Invalid progress failure")
        ids = []
    else:
        ids = req.tokens
        reg.require(retired_ids == ids and len(ids) in range(1,reg.CAP+1) and not debt and
                    mask not in ids and eos not in ids[:-1] and (ids[-1]==eos or len(ids)==reg.CAP)
                    and len(gen)==len(sampled), "Incomplete or incorrect output")
    special = set(tok.all_special_ids)|{mask,eos}
    n, f, paid = sum(v not in special for v in ids), len(widths), sum(widths)
    score = {"correct":False,"reason":"generation_failed"} if failure else scoring.score(
        tok.decode(ids,skip_special_tokens=True),task)
    return dict(**{k:task[k] for k in reg.KEYS}, q=q, law=[law.num,law.den],
                **({"ablation":ablation, "build_F":phases["build_sample"],
                    "build_Q":sum(e["q"] for e in sampled if e["phase"]=="build_sample"),
                    "over_offer_F":sum(w>q for w in widths)} if ablation is not None else {}),
                input_sha256=sha256(json.dumps(prompt,separators=(",",":")).encode()).hexdigest(),
                output_ids=ids, score=score, correct=bool(score["correct"]), failure=failure,
                N=n,F=f,Q=paid,TPF=n/f,efficiency=n/paid,max_decoder_Q=max(widths),
                projection_modules=20,decoder_Q=paid,encoder_Q=0,prefill_Q=origin,
                row_visits=dict(stats),phase_F=dict(phases),trace=sampled)


def assert_dense_joint_match(*cases):
    """Check three independently evaluated Q64 rows, never synthesize a control row."""
    reg.require(len(cases)==3 and {c["ablation"] for c in cases}=={"00","10","11"}
                and all(c["q"]==64 and not c["build_F"] for c in cases), "Expected three Q64 arms")
    for key in (*reg.KEYS, "law", "input_sha256", "failure", "trace", "output_ids", "score", "N", "F", "Q"):
        reg.require(all(c[key]==cases[0][key] for c in cases[1:]), f"Q64 arms diverged: {key}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiers",type=int,choices=TIERS,nargs="+",default=(32,64,16,8,4))
    ap.add_argument("--num",type=int,choices=(0,1),default=1)
    modes=ap.add_mutually_exclusive_group()
    modes.add_argument("--tail-pilot",action="store_true",help="Three fixed Q64 arms on frozen 0/10/20, CPU cores 44-47")
    modes.add_argument("--tail-full90",action="store_true",help="Fresh eager/fill-head pairs on frozen90 at Q64")
    a = ap.parse_args()
    tail=a.tail_pilot or a.tail_full90
    if tail and (tuple(a.tiers)!=(64,) or a.num!=1):
        ap.error("Tail contrasts require --tiers 64 and the 1/8 law")
    cores=range(44,48) if tail else range(36,40)
    reg.require(len(a.tiers)==len(set(a.tiers)) and os.environ.get("CUDA_VISIBLE_DEVICES")==""
                and os.sched_getaffinity(0)==set(cores), "Use the registered four-core CPU lane")
    import torch, transformers, transformers.utils as u
    if not hasattr(u,"LossKwargs"):
        from typing import TypedDict
        u.LossKwargs=TypedDict("LossKwargs",{"num_items_in_batch":int},total=False)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bdlm_serving.runtime.serial import install_llada_rectangular_mask
    reg.require(str(torch.__version__)=="2.9.0+cu128" and transformers.__version__=="4.57.1" and
                not torch.cuda.is_available() and reg.sha(Path(scoring.__file__))==SCORER_SHA,
                "Runtime, scorer or device mismatch")
    os.umask(0o077)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    samples = reg.DATA/"samples.json"
    tasks = load_tasks(samples)
    ref = resolve_reference(SimpleNamespace(samples=samples))
    law = Law(a.num,8)
    ordinary=OUT/"cpu"/f"law-{a.num}-8"
    pilot_root=reg.DATA/"llada21-family-tail-cpu3-v1"
    root=pilot_root if a.tail_pilot else reg.DATA/"llada21-family-tail-cpu90-v1" if a.tail_full90 else ordinary
    indices=(0,10,20) if a.tail_pilot else ORDER
    rules=(("eager","defer_tail","fill_head") if a.tail_pilot else ("eager","fill_head") if a.tail_full90 else ("eager",))
    if a.tail_full90:
        seal=reg.read(pilot_root/"complete.json")
        reg.require(seal["run"]==reg.sha(pilot_root/"run.json") and len(seal["files"])==9 and
                    all(reg.sha(pilot_root/k)==v for k,v in seal["files"].items()),"Unsealed tail pilot")
    reg.require(reg.sha(samples)==reg.SAMPLES_SHA,"Unexpected frozen inputs")
    root.mkdir(parents=True,exist_ok=False)
    paths = [HERE, *(REPO/p for p in ("bdlm_serving/methods/t2t/budget_family.py",
        "bdlm_serving/methods/t2t/llada_budget_request.py","bdlm_serving/methods/t2t/llada_sparse.py",
        "bdlm_serving/methods/optimus/upstream.py","bdlm_serving/runtime/serial.py"))]
    hashes = {p.name:reg.sha(p) for p in paths}
    meta = root/"run.json"
    reg.save(meta,dict(tiers=a.tiers,indices=indices,samples=reg.sha(samples),scoring=SCORER_SHA,
        **({"tail_rules":rules,"cores":list(cores),"eager_reference":reg.sha(ordinary/"run.json")} if tail else {}),
        **({"pilot_seal":reg.sha(pilot_root/"complete.json")} if a.tail_full90 else {}),
        files=hashes,law=[law.num,law.den],checkpoint=str(CKPT),device="cpu",
        torch=torch.__version__,transformers=transformers.__version__,length_cap=reg.CAP,
        m2t=.9,t2t=.5,retirement="all_resident_no_masks_no_token_debt",case_failure_score=0))
    print("Verifying weights and 90 prompt digests",flush=True)
    verify_checkpoint(CKPT)
    opts = dict(trust_remote_code=True,local_files_only=True)
    tok = AutoTokenizer.from_pretrained(CKPT,**opts)
    validate_prompts(tok,tasks,ref)
    model=AutoModelForCausalLM.from_pretrained(CKPT,**opts,dtype=torch.bfloat16,
                    attn_implementation="sdpa",device_map={"":"cpu"}).eval()
    install_llada_rectangular_mask(model,allow_selected_cache=True)
    receipts = {}
    for i in indices:
        prompt=reg.read(ref/reg.filename(i,"native32"))["prompt_ids"]
        for q in a.tiers:
            for rule in rules:
                suffix=f"-{rule}" if tail else ""
                dst=root/f"{i:02}-q{q}{suffix}.json"
                reg.require(not dst.exists() and all(reg.sha(p)==hashes[p.name] for p in paths),"Source/row changed")
                print(f"Running fixed Q{q} index={i} {rule}",flush=True)
                row=_fixed(model,tok,tasks[i],prompt,q,law,tail_rule=rule)
                if a.tail_pilot:
                    refrow=reg.read(ordinary/f"{i:02}-q{q}.json")
                    reg.require(all(row[k]==refrow[k] for k in (*reg.KEYS,"q","law","input_sha256")) and
                                refrow["registration_sha256"]==reg.sha(ordinary/"run.json"),"Unpaired fixed request")
                    row["tail_rule"]=rule
                    if rule=="eager":
                        reg.require(all(row[k]==refrow[k] for k in ("trace","output_ids","score","N","F","Q")),
                                    "Rerun differs from original CPU control")
                    else:
                        row["first_intervention"]=compare_tail(row,refrow)
                elif a.tail_full90:
                    row["tail_rule"]=rule
                    if i in (0,10,20):
                        pilot_row=reg.read(pilot_root/dst.name)
                        reg.require(all(row[k]==pilot_row[k] for k in ("trace","output_ids","score","N","F","Q")),
                                    "Expanded cohort does not reproduce its pilot")
                    if rule!="eager":
                        refrow=reg.read(root/f"{i:02}-q64-eager.json")
                        reg.require(all(row[k]==refrow[k] for k in (*reg.KEYS,"q","law","input_sha256"))
                                    and refrow["registration_sha256"]==reg.sha(meta),"Unpaired full90 treatment")
                        row["first_intervention"]=compare_tail(row,refrow)
                row["registration_sha256"]=reg.sha(meta)
                reg.save(dst,row)
                receipts[dst.name]=reg.sha(dst)
                print(f"Sealed {i:02} Q{q} {rule}: correct={row['correct']}, N/F={row['N']}/{row['F']}, paid={row['Q']}",flush=True)
    reg.save(root/"complete.json",dict(run=reg.sha(meta),files=receipts))


def compare_tail(case,ref):
    rule=case["tail_rule"]
    for i,(e,old) in enumerate(zip(case["trace"],ref["trace"])):
        info=e.get("tail_intervention")
        if info is None:
            reg.require(e==old,"Trajectories diverged before treatment")
            continue
        reg.require(info["rule"]==rule and all(e[k]==old[k] for k in ("offer","block","dirty_before","dirty","cold","visible_kv")),
                    "Different input at first treatment")
        split=(1+e["block"])*32
        if rule=="defer_tail":
            withheld=set(info["withheld"])
            reg.require(withheld and withheld=={p for p in old["cold"] if p>=split} and
                        e["rows"]==old["rows"] and set(e["sampled"])==set(old["sampled"])-withheld,
                        "Changed projection or wrong tail suppression")
        else:
            extra=set(info["added"])
            reg.require(rule=="fill_head" and extra and max(extra)<split and not extra.intersection(old["rows"])
                        and set(e["rows"])==set(old["rows"])|extra
                        and set(e["sampled"])==set(old["sampled"])|extra,"Changed original seats")
        return i
    reg.require(all(case[k]==ref[k] for k in ("trace","output_ids","score","F","Q")),"Unexplained no-op")
    return None


if __name__=="__main__":
    main()
