"""Fork real progress80 commit states; compare two charged calls to one.

CPU mechanism probe, not generation throughput or a benchmark policy. Run:
CUDA_VISIBLE_DEVICES='' taskset -c 32-39 ../venv/bin/python -m \
experiments.t2t.serving_slots.commit_pair_probe <fresh cluster output.json>
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path


def fork_probe(k, s):
    import torch
    from transformers.cache_utils import DynamicCache
    from bdlm_serving.methods.t2t.llada_sparse import _AuditedDenseCache

    # Keep the original request untouched so every test starts on reference history.
    a, b = deepcopy(s), deepcopy(s)
    head, start = s.block, s.block*32
    cut, end = start+32, start+64
    if (k._eos_or_end(s, head) or not k._tail_resident(s) or s.committed_end!=start
            or (s.x[:,start:cut]==s.mask_id).any()):
        raise ValueError("Not a nonterminal rolling promotion")
    readouts=[]
    def capture(_m, _args, out):
        readouts.append(out.logits.detach().clone())
    with k.model.register_forward_hook(capture):
        c=k.step(a,64)
        v=k.step(a,64)
    assert (c["top_action"],v["top_action"])==("commit_head","promote_verify")
    assert len(readouts)==2 and v["sampled_q"]==32 and c["sampled_q"]==0
    temp=DynamicCache.from_legacy_cache(b.cache.to_legacy_cache())
    k._prepare_hard_metadata(b,[head,head+1])
    logits=k._execute(b,torch.arange(start,end),end,_AuditedDenseCache(temp,64,end),
                      "denoise","probe_commit_verify",False)
    layers=temp.to_legacy_cache()
    for i,(key,val) in enumerate(layers):
        b.cache.update(key[...,start:cut,:],val[...,start:cut,:],i)
    b.committed_end=cut
    b.context_version+=1
    k._advance_one(b)
    b.left_done=False
    b.left_post_steps=b.step_in_block=0
    b.resident=layers
    k._prepare_hard_metadata(b,[b.block])
    k._apply_pass(b,logits[:,32:],[b.block],active_width=32,saturated=True,full_two_block=False)
    b.phase="commit" if b.left_done else "denoise"
    b.offer_tick+=1
    b.previous_offer=64
    assert a.offer_tick==b.offer_tick+1 and b.events[-1]["layer_physical_q"]==[64]*len(layers)

    def compare_kv(left,right):
        if len(left)!=len(right) or not left:
            raise ValueError("Layer count changed")
        diffs=[]
        for pair,other in zip(left,right,strict=True):
            if any(x.shape!=y.shape for x,y in zip(pair,other,strict=True)):
                raise ValueError("Cache shape changed")
            diffs.append(max((x.float()-y.float()).abs().max().item() for x,y in zip(pair,other,strict=True)))
        return {"layers":len(diffs),"nonexact_layers":sum(d!=0 for d in diffs),"max_abs":max(diffs)}

    def debt(st):
        return [i for i,(w,q) in enumerate(zip(st.last_write,st.last_query)) if w>=0 and w>=q]
    def rng_equal(x,y):
        return x is None and y is None or x is not None and y is not None and torch.equal(x,y)
    ca=compare_kv(a.cache.to_legacy_cache(),b.cache.to_legacy_cache())
    ra=compare_kv(a.resident,b.resident)
    diff=(readouts[1].float()-logits[:,32:].float()).abs().max().item()
    prefix_untouched=all(torch.equal(before,after[...,:start,:])
        for st in (a,b) for orig,new in zip(s.cache.to_legacy_cache(),st.cache.to_legacy_cache(),strict=True)
        for before,after in zip(orig,new,strict=True))
    same_status=all(getattr(a,k)==getattr(b,k) for k in ("block","committed_end","context_version","phase",
        "left_history","right_history","left_done","right_done","left_post_steps","right_post_steps","step_in_block"))
    out=dict(block=head,tail_debt=len(c["mandatory_tail_positions"]),seq_q=[c["physical_q"],32],joint_q=64,
             committed=ca,resident=ra,logit_max_abs=diff,
             tail_argmax_changes=int((readouts[1].argmax(-1)!=logits[:,32:].argmax(-1)).sum()),
             canvas_equal=torch.equal(a.x,b.x),rng_equal=rng_equal(a.rng_cpu,b.rng_cpu) and rng_equal(a.rng_cuda,b.rng_cuda),
             debt_equal=debt(a)==debt(b),metadata_equal=same_status,original_prefix_unchanged=prefix_untouched)
    # A caller can next offer Q4 on either state. No implicit extra Q64 tick.
    x,y=k.step(a,4),k.step(b,4)
    out["next_q4_geometry_equal"]=all(x[v]==y[v] for v in ("absolute_positions","sampled_positions",
                                                      "refreshed_pending","physical_q","top_action"))
    out["next_q4_canvas_equal"]=torch.equal(a.x,b.x)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    out=parser.parse_args().output
    if out.exists() or os.environ.get("CUDA_VISIBLE_DEVICES")!="" or len(os.sched_getaffinity(0))!=8:
        raise ValueError("Fresh output, hidden CUDA and eight pinned CPU cores required")
    import torch
    import transformers.utils as utils
    if not hasattr(utils,"LossKwargs"):
        from typing import TypedDict
        utils.LossKwargs=TypedDict("LossKwargs", {"num_items_in_batch":int}, total=False)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.t2t.serving_slots import spawn_mask_gpu as study
    from experiments.t2t.serving_slots import llada_all_slots90 as reg
    from experiments.t2t.serving_slots.llada_mask_pilot import CKPT, CFG, INDICES
    from experiments.t2t.serving_slots.mask_probe_llada import verify_checkpoint
    from experiments.t2t.serving_slots.llada_spawn import SpawnGatedKernel
    from bdlm_serving.runtime.serial import install_llada_rectangular_mask
    from bdlm_serving.evaluation.frozen90 import load_tasks
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    samples=reg.DATA/"samples.json"
    assert reg.sha(samples)==reg.SAMPLES_SHA
    tasks=load_tasks(samples)
    verify_checkpoint(CKPT)
    opts=dict(trust_remote_code=True,local_files_only=True)
    tok=AutoTokenizer.from_pretrained(CKPT,**opts)
    model=AutoModelForCausalLM.from_pretrained(CKPT,**opts,dtype=torch.bfloat16,
                attn_implementation="sdpa",device_map={"":"cpu"}).eval()
    install_llada_rectangular_mask(model,allow_selected_cache=True)
    results=[]
    with torch.inference_mode():
        for i in INDICES:
            r=reg.read(study.OUT/reg.filename(i,"progress80-causal"))
            assert all(r[key]==tasks[i][key] for key in reg.KEYS)
            k=SpawnGatedKernel(model,spawn_threshold=.8)
            s=k.create(torch.tensor([r["prompt_ids"]]),CFG,reg.CAP,r["seed"],tok.convert_tokens_to_ids("<|mask|>"),
                       [tok.eos_token_id])
            for _ in range(128):
                if s.phase=="done":
                    break
                if ((s.phase=="commit" or k._limit_reached(s,s.block)) and k._tail_resident(s)
                        and not k._eos_or_end(s,s.block)):
                    row=dict(index=i,**fork_probe(k,s))
                    results.append(row)
                    print(json.dumps(row),flush=True)
                    if sum(x["index"]==i for x in results)==2:
                        break
                k.step(s,64)
    data=dict(device="cpu",dtype="bfloat16",torch=str(torch.__version__),indices=list(INDICES),gate=.8,
              checkpoint_manifest=reg.sha(reg.REPO/"benchmarks/checkpoints/llada21.json"),
              frozen_sha256=reg.SAMPLES_SHA,source_sha256=reg.sha(Path(__file__)),cases=results)
    with out.open("x") as f:
        json.dump(data,f,indent=2,allow_nan=False)
        f.write("\n")
    if not results:
        raise SystemExit("No eligible promotions")


if __name__=="__main__":
    main()
