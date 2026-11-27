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


def fixed(model, tok, task, prompt, q, law):
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
                                seed=task["seed"],law=law,confidence_threshold=.9,editing_threshold=.5)
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
            reg.require(e["tick"] == len(attention)-2 and e["rows"] == attention[-1] and
                        e["q"] == head_width[-1] == w <= q and e["offer"] == q and
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
    reg.require(widths == head_width[1:] and widths and max(widths)<=q, "Unpaid projection")
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
                input_sha256=sha256(json.dumps(prompt,separators=(",",":")).encode()).hexdigest(),
                output_ids=ids, score=score, correct=bool(score["correct"]), failure=failure,
                N=n,F=f,Q=paid,TPF=n/f,efficiency=n/paid,max_decoder_Q=max(widths),
                projection_modules=20,decoder_Q=paid,encoder_Q=0,prefill_Q=origin,
                row_visits=dict(stats),phase_F=dict(phases),trace=sampled)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiers",type=int,choices=TIERS,nargs="+",default=(32,64,16,8,4))
    ap.add_argument("--num",type=int,choices=(0,1),default=1)
    a = ap.parse_args()
    reg.require(len(a.tiers)==len(set(a.tiers)) and os.environ.get("CUDA_VISIBLE_DEVICES")==""
                and os.sched_getaffinity(0)==set(range(36,40)), "Use the registered four-core CPU lane")
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
    root = OUT/"cpu"/f"law-{a.num}-8"
    root.mkdir(parents=True,exist_ok=False)
    paths = [HERE, *(REPO/p for p in ("bdlm_serving/methods/t2t/budget_family.py",
        "bdlm_serving/methods/t2t/llada_budget_request.py","bdlm_serving/methods/t2t/llada_sparse.py",
        "bdlm_serving/methods/optimus/upstream.py","bdlm_serving/runtime/serial.py"))]
    hashes = {p.name:reg.sha(p) for p in paths}
    meta = root/"run.json"
    reg.save(meta,dict(tiers=a.tiers,indices=ORDER,samples=reg.sha(samples),scoring=SCORER_SHA,
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
    for i in ORDER:
        prompt=reg.read(ref/reg.filename(i,"native32"))["prompt_ids"]
        for q in a.tiers:
            dst=root/f"{i:02}-q{q}.json"
            reg.require(not dst.exists() and all(reg.sha(p)==hashes[p.name] for p in paths),"Source/row changed")
            print(f"Running fixed Q{q} index={i}",flush=True)
            row=fixed(model,tok,tasks[i],prompt,q,law)
            row["registration_sha256"]=reg.sha(meta)
            reg.save(dst,row)
            receipts[dst.name]=reg.sha(dst)
            print(f"Sealed {i:02} Q{q}: correct={row['correct']}, N/F={row['N']}/{row['F']}, paid={row['Q']}",flush=True)
    reg.save(root/"complete.json",dict(run=reg.sha(meta),files=receipts))


if __name__=="__main__":
    main()
