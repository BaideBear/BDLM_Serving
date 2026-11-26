"""Real-weight paired execution, not a frozen90 quality run (short generation cap)."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bdlm_serving.methods.t2t.llada_sparse import LLaDASparseKernel
from bdlm_serving.methods.t2t.verified_commit import VerifiedCommitKernel
from bdlm_serving.runtime.serial import install_llada_rectangular_mask
from experiments.t2t.serving_slots.mask_probe_llada import verify_checkpoint


@torch.inference_mode()
def measure(model, ids, cfg, q, length, seed, mask, eos):
    kernels = [cls(model,q,0,q,policy="hard_prefix_zero")
               for cls in (LLaDASparseKernel, VerifiedCommitKernel)]
    a,b = [k.create(ids,cfg,length,seed,mask,eos) for k in kernels]
    charged = [Counter(),Counter()]
    all_actions = Counter()
    commits = []
    for _ in range(2048):
        if a.phase == "done":
            break
        actions=[]
        for i,(k,s) in enumerate(zip(kernels,(a,b),strict=True)):
            widths=[]
            hook=k.projections[0].register_forward_pre_hook(
                lambda _m,args: widths.append(args[0].shape[1]))
            try:
                old=len(s.events)
                actions.append(k.step(s))
                if widths != [e["physical_q"] for e in s.events[old:]]:
                    raise ValueError("Physical hook disagrees with kernel ledger")
                if actions[-1]!="prefill":
                    charged[i].update(widths)
            finally:
                hook.remove()
        if (actions[0]!=actions[1] or (a.phase,a.block,a.context_version,a.committed_end)!=(
                b.phase,b.block,b.context_version,b.committed_end) or not torch.equal(a.x,b.x)
                or not torch.equal(a.rng_cpu,b.rng_cpu) or not torch.equal(a.rng_cuda,b.rng_cuda)):
            raise ValueError("Paired requests diverged")
        all_actions[actions[0]]+=1
        if actions[0]=="commit":
            left,right=a.cache.to_legacy_cache(),b.cache.to_legacy_cache()
            if (not left or len(left)!=len(kernels[0].projections) or
                    a.cache.get_seq_length()!=a.committed_end or b.cache.get_seq_length()!=b.committed_end or
                    any(not torch.equal(x,y) for l,r in zip(left,right,strict=True)
                        for x,y in zip(l,r,strict=True))):
                raise ValueError("Different committed KV")
            commits.append(dict(**b.commits[-1],committed_end=b.committed_end,
                                exact_layers=len(left)))
    if a.phase!="done" or not any(c["promoted"] for c in commits):
        raise ValueError("Run did not finish or exercise promotion")
    n=sum(c["promoted"] for c in commits)
    if charged[0]!=charged[1]+Counter({32:n}):
        raise ValueError("Unexpected work outside promotion")
    return dict(seed=seed,gen_length=length,q=q,prompt_rows=ids.shape[1],actions=all_actions,
                correct_task_score=None,exact_canvas_and_rng_at_every_step=True,commits=commits,
                generation=[dict(F=sum(w.values()),physical_Q=sum(k*v for k,v in w.items()))
                            for w in charged])


@torch.inference_mode()
def measure_top(model,ids,cfg,length,seed,mask,eos):
    from experiments.t2t.serving_slots.llada_top_transition import TopTransitionKernel
    kernel=TopTransitionKernel(model)
    state=kernel.create(ids,cfg,length,seed,mask,eos)
    offers=(32,64,32,32,64)
    history=[]
    for tick in range(2048):
        if state.phase=="done":
            break
        q=offers[tick%len(offers)]
        e=kernel.step(state,q)
        if e["layer_physical_q"] != [e["physical_q"]]*len(kernel.projections):
            raise ValueError("Per-layer Q mismatch")
        history.append({key:e[key] for key in ("offer_tick","top_action","block","nominal_q",
                        "physical_q","overflow","sampled_q","refresh_only_positions","edited")})
    if state.phase!="done" or not any(e["top_action"]=="fused_growth" for e in history):
        raise ValueError("Top-tier route incomplete or did not grow a tail")
    return dict(seed=seed,gen_length=length,prompt_rows=ids.shape[1],offers=offers,
                correct_task_score=None,generation=dict(F=len(history),physical_Q=sum(
                    e["physical_q"] for e in history)),layers_checked=len(kernel.projections),
                steps=history)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--original",type=Path,required=True,help="One archived LLaDA OBS/IBS request")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--q",type=int,choices=(4,8,16,32),default=8)
    p.add_argument("--top",action="store_true",help="Check the external 32/64 route instead")
    p.add_argument("--length",type=int,default=38)
    p.add_argument("--threads",type=int,default=12)
    args=p.parse_args()
    if args.length<1 or args.out.exists():
        raise ValueError("Need positive length and a fresh output")
    torch.set_num_threads(args.threads)
    verify_checkpoint(args.checkpoint)
    opts=dict(trust_remote_code=True,local_files_only=True)
    tok=AutoTokenizer.from_pretrained(args.checkpoint,**opts)
    model=AutoModelForCausalLM.from_pretrained(args.checkpoint,**opts,dtype=torch.bfloat16,
            attn_implementation="sdpa",device_map={"":"cpu"}).eval()
    install_llada_rectangular_mask(model,allow_selected_cache=True)
    row=json.loads(args.original.read_text())
    cfg=dict(confidence_threshold=.5,editing_threshold=.5,max_post_steps=16)
    ids=torch.tensor([row["prompt_ids"]])
    common=(args.length,row["seed"],tok.convert_tokens_to_ids("<|mask|>"),[tok.eos_token_id])
    result=(measure_top(model,ids,cfg,*common) if args.top else measure(model,ids,cfg,args.q,*common))
    result.update(task_id=row["task_id"],device="cpu",dtype="bfloat16",torch=str(torch.__version__),
                  thresholds=cfg)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":
    main()
