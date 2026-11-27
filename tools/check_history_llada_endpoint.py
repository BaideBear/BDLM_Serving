"""Real-weight, same-input and rollout controls, never full-cohort scores."""
import argparse
import json
import os
from pathlib import Path
from typing import TypedDict
import torch
import transformers.utils as utils
if not hasattr(utils,"LossKwargs"):
    class LossKwargs(TypedDict,total=False):
        num_items_in_batch:int
    utils.LossKwargs=LossKwargs
from transformers import AutoModelForCausalLM,AutoTokenizer
from bdlm_serving.runtime.serial import SerialKernel,install_llada_rectangular_mask
from bdlm_serving.methods.t2t.history_llada import LLadaHistory


class LayerAudit(LLadaHistory):
    """Compare outputs on the same post-RoPE inputs, before routing can diverge."""
    def __init__(self, model):
        self.rows=[]
        super().__init__(model)

    def _call(self, module, q, k, v, mask, **kw):
        output=super()._call(module,q,k,v,mask,**kw)
        if self.active is None:
            return output
        native=self.native(module,q,k,v,mask,**kw)[0].float()
        groups=q.shape[1]//k.shape[1]
        k32=k.float().repeat_interleave(groups,dim=1)
        v32=v.float().repeat_interleave(groups,dim=1)
        scores=q.float() @ k32.transpose(-1,-2) * kw["scaling"]
        if mask is not None:
            if mask.dtype==torch.bool:
                scores=scores.masked_fill(~mask,-torch.inf)
            else:
                scores=scores+mask.float()
        elif kw.get("is_causal",getattr(module,"is_causal",True)) and q.shape[-2]>1:
            scores=scores.masked_fill(~torch.ones_like(scores,dtype=torch.bool).tril(),-torch.inf)
        ref=(scores.softmax(-1) @ v32).transpose(1,2)
        def error(x):
            diff=x-ref
            return dict(max_abs=diff.abs().max().item(),
                        relative_rms=(diff.norm()/ref.norm().clamp_min(1e-12)).item())
        self.rows.append(dict(layer=module.layer_idx,history=k.shape[-2]-q.shape[-2],
                              native=error(native),split=error(output[0].float())))
        return output


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--references",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--indices",type=int,nargs="+",default=[0,10,20])
    p.add_argument("--layer-audit",action="store_true")
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    opts={"trust_remote_code":True,"local_files_only":True}
    tok=AutoTokenizer.from_pretrained(args.checkpoint,**opts)
    model=AutoModelForCausalLM.from_pretrained(args.checkpoint,**opts,dtype=torch.bfloat16,
                                               attn_implementation="sdpa").eval()
    install_llada_rectangular_mask(model,allow_selected_cache=True)
    if "DG_SIGNAL" in os.environ:
        import runpy
        runpy.run_path(os.environ["HISTORY_PREPARE"],run_name="bdlm_prepare")["prepare"](model)
    else:
        model.to("cuda")
    adapter=(LayerAudit if args.layer_audit else LLadaHistory)(model)
    eng=SerialKernel(model,"llada21")
    cfg=dict(confidence_threshold=.5,editing_threshold=.5,max_post_steps=16)
    mask=tok.convert_tokens_to_ids("<|mask|>")
    args.out.mkdir(parents=True,exist_ok=True)
    for idx in args.indices:
        ref=json.loads((args.references/f"{idx:02d}-native32.json").read_text())
        prompt=torch.tensor([ref["prompt_ids"]],device=model.device)
        def new():
            return eng.create(prompt,cfg,ref["generation_cap"],ref["seed"],mask,[tok.eos_token_id])
        a,b=new(),new()
        summaries=[]
        # Same-state comparisons: if trajectories diverge, re-anchor to native.
        for j in range(10):
            if a.phase=="done":
                break
            eng.capture_logits=True
            phase=eng.step(a)
            native=eng.last_logits
            if phase=="denoise":
                if args.layer_audit:
                    adapter.rows.clear()
                with adapter.activate([list(range(32))]):
                    eng.step(b)
                logits=eng.last_logits
                delta=(logits.float()-native.float()).abs()
                row=dict(iter=j,block=a.block,max_abs=delta.max().item(),
                         mean_abs=delta.mean().item(),top1_equal=int((logits.argmax(-1)==native.argmax(-1)).sum()),
                         actions_equal=torch.equal(a.x,b.x),native_edits=a.events[-1]["edited"],
                         split_edits=b.events[-1]["edited"])
                if not torch.isfinite(delta).all():
                    raise RuntimeError("nonfinite difference")
                if args.layer_audit:
                    row["layers"]=list(adapter.rows)
                summaries.append(row)
                print(idx,{k:v for k,v in row.items() if k!="layers"},flush=True)
            else:
                eng.step(b)
            if not torch.equal(a.x,b.x) or a.phase!=b.phase:
                b=eng.snapshot(a)
        args.out.joinpath(f"{idx:02d}-fixed-input.json").write_text(json.dumps(summaries))
    adapter.close()


if __name__=="__main__":
    main()
