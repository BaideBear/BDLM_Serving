"""A1 same-stack native versus strict Q256; only unrestricted mask representation varies."""
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from .gemma_cold_frontier_last_eval import board,sha,read,CKPT,SCORER,SAMPLES
if __name__=="__main__":
    board(False)

from .gemma_native_visibility import representation
from .gemma_joint_verify_eval import ORDER,check
from .gemma_unified_audit import projectors
from .gemma_wave_kv import _attention_functions
from .mask_pair_gemma import input_digest,SAMPLES_SHA
from bdlm_serving.runtime.diffusiongemma import observe,account
from bdlm_serving.methods.t2t.serving_profiles import open_strict_gemma
from bdlm_serving.evaluation import scoring
from bdlm_serving.evaluation.frozen90 import load_tasks

ROOT=Path("/gs/results/gemma-native-visibility-a1-1128")
HIST=Path("/gs/results/dense-stop-1128")
W1=Path("/gs/results/flowblock-frozen90-1127/analysis/reference90/runtime/t2t-native-1125/prefix90/gemma-wave-reference90")
PROFILE="dense_stop_joint_verified_eos"


def save(path,obj):
    temp=path.with_suffix(".writing")
    temp.write_text(json.dumps(obj,sort_keys=True)+"\n")
    temp.replace(path)


def h(t):
    import torch
    if t is None:
        return None
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


def run(model,proc,data,task,arm,probe):
    import torch
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import EntropyBoundSampler
    origin=data["input_ids"].shape[1]
    eos=set(model.generation_config.eos_token_id)
    special=set(proc.tokenizer.all_special_ids)|eos
    tape,layers,used=[],[],[]
    debt=set()
    verified=[]
    enc=[]
    ordinal=0
    def forward(*args,**kw):
        ids=kw["decoder_input_ids"]
        first=len(tape)==0
        rec=dict(ids=h(ids),pos=h(kw["decoder_position_ids"]),SC=h(kw.get("self_conditioning_logits")) if first else None)
        tape.append(rec)
        out=base(*args,**kw)
        rec.update(draft=h(out.logits.argmax(-1)),raw=h(out.logits) if first else None)
        return out
    base=model.forward
    impl=_attention_functions()
    orig=impl["sdpa"]
    attns={id(l.self_attn) for l in model.model.decoder.layers}
    def attention(mod,q,k,v,mask,**kw):
        detailed=id(mod) in attns and len(tape)==1
        out=orig(mod,q,k,v,mask,**kw)
        if detailed:
            layers.append(dict(layer=mod.layer_idx,q=h(q),k=h(k),v=h(v),none=mask is None,
                               causal=kw["is_causal"],out=h(out[0])))
        return out
    ac=EntropyBoundSampler.accept_canvas
    def accept(s,ids,sampled,scores,t):
        out=ac(s,ids,sampled,scores,t)
        tape[-1].update(clock=int(t),sample=h(sampled),accepted=h(s.accepted_token_mask),
                        scores=h(scores) if len(tape)==1 else None)
        return out
    torch.manual_seed(task["seed"])
    with torch.inference_mode(),ExitStack() as stack:
        phys=stack.enter_context(observe(model))
        projs,pos=stack.enter_context(projectors(model))
        if probe:
            stack.enter_context(patch.object(model,"forward",forward))
            stack.enter_context(patch.object(EntropyBoundSampler,"accept_canvas",accept))
            impl["sdpa"]=attention
            stack.callback(impl.__setitem__,"sdpa",orig)
        if arm=="native":
            result=model.generate(**data,max_new_tokens=2048)
            tokens,_=account(phys,result.sequences[0,origin:].tolist(),origin,256,2048,eos,special,
                             model.generation_config.pad_token_id,result.tokens_per_forward.item())
        else:
            hand=open_strict_gemma(model,data,seed=task["seed"],initial_offer=256,profile=PROFILE)
            req=hand._request
            tops=[]
            stack.callback(model.register_forward_hook(lambda m,a,out:tops.append(out.logits[0].argmax(-1).tolist())).remove)
            stack.enter_context(_encoder_inputs(model,enc))
            while hand.status=="ready":
                with representation(req,arm=="on",used):
                    e=hand.step(256)
                p,=e["parts"]
                rows=pos[-1]
                check(rows==list(range(origin+256*ordinal,origin+256*(ordinal+1)))==e["positions"] and
                      e["dirty_before_absolute"]==sorted(debt) and set(rows)>=debt and
                      len(tops)==len(pos) and e["ordinal"]==ordinal and
                      p["age_after"]==p["age_before"]+bool(p["sampled"]),"Incorrect physical or clock ledger")
                debt={origin+256*ordinal+j for k in ("input_changed","admitted","revoked") for j in p[k]}
                check(debt==set(e["dirty_after_absolute"]),"Lost write")
                if e["route"] in ("verify","verify_paid"):
                    check(not debt and not p["sampled"],"Unpaid or sampling verify")
                    block=tops[-1]
                    end=next((i+1 for i,v in enumerate(block) if v in eos),256)
                    verified.extend(block[:end])
                    if e["encoded"]:
                        check(end==256 and not eos.intersection(block) and enc[-1]==block,"Wrong encoder inputs")
                        ordinal+=1
                else:
                    check(e["route"]=="dense_single" and len(p["sampled"])==256 and not e["encoded"],"Unexpected route")
                if probe:
                    tape[-1]["route"]=e["route"]
            tokens=list(hand.result())
            check(not debt and verified==tokens and len(enc)==ordinal and len(used)==len(pos)*len(attns),
                  "Invalid close, encoding, or mask exposure")
    check(phys[0]==("encoder",origin) and len(pos)==sum(k=="decoder" for k,_ in phys)
          and len(projs)==85 and all(v==[256]*len(pos) for v in projs.values())
          and (not probe or len(tape)==len(pos) and len(layers)==len(attns)),"Physical trace mismatch")
    return dict(ids=tokens,N=sum(i not in special for i in tokens),F=len(phys)-1,decoder_Q=256*len(pos),
                encoder_F=len(phys)-1-len(pos),correct=bool(scoring.score(
                    proc.tokenizer.decode(tokens,skip_special_tokens=True),task)["correct"]),
                tape=tape,layers=layers,mask_layer_calls=len(used))


from contextlib import contextmanager
@contextmanager
def _encoder_inputs(model,out):
    def pre(m,args,kw):
        out.append(kw["input_ids"][0].tolist())
    with model.model.encoder.register_forward_pre_hook(pre,with_kwargs=True):
        yield


def main():
    import torch
    import transformers
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as native
    from transformers import AutoProcessor,DiffusionGemmaForBlockDiffusion
    from time import time
    board(True)
    reg=dict(pid=os.getpid(),start=time(),start_ticks=Path("/proc/self/stat").read_text().split(") ")[1].split()[19],
             inputs=sha(SAMPLES),scorer=sha(Path(scoring.__file__)),checkpoint=str(CKPT),
             native=sha(Path(native.__file__)),torch=str(torch.__version__),tf=transformers.__version__,
             sources={p.name:sha(p) for p in Path(__file__).parent.glob("gemma_native_visibility*.py")})
    check(reg["inputs"]==SAMPLES_SHA and reg["scorer"]==SCORER and reg["tf"]=="5.12.0" and
          reg["native"]=="cded57b7e365f0265b1c088e993c358b5f70acdf48055fce396540b4d9d691bc" and
          reg["torch"]=="2.6.0+cu124","Bad stack")
    ROOT.mkdir(exist_ok=False)
    save(ROOT/"registration.json",reg)
    print("REGISTERED",os.getpid(),flush=True)
    tasks=load_tasks(SAMPLES)
    proc=AutoProcessor.from_pretrained(CKPT,local_files_only=True)
    model=DiffusionGemmaForBlockDiffusion.from_pretrained(CKPT,local_files_only=True,dtype=torch.bfloat16,
          attn_implementation="sdpa",experts_implementation="grouped_mm",device_map={"":"cuda:0"}).eval()
    gates=[]
    for j,i in enumerate(ORDER):
        task=tasks[i]
        hist=read(HIST/f"C256-dense-stop1-{i:02}.json")
        w=read(W1/f"{i:02}-flow_w1.json")
        check(all(w[k]==hist[k]==task[k] for k in ("task_id","seed","index","dataset"))
              and hist["scorer_sha256"]==SCORER and hist["samples_sha256"]==SAMPLES_SHA,"Bad old receipts")
        data=proc.apply_chat_template([{"role":"user","content":task["prompt"]}],tokenize=True,
               add_generation_prompt=True,return_dict=True,return_tensors="pt").to(model.device)
        check(input_digest(data)==hist["input_sha256"],"Prompt mismatch")
        r=dict(index=i,seed=task["seed"])
        for arm in (("native","off","on") if j%2==0 else ("on","off","native")):
            board(True)
            a=r[arm]=run(model,proc,data,task,arm,j<3)
            if arm=="off":
                check(a["ids"]==hist["output_ids"] and all(a[k]==hist[k] for k in
                      ("F","N","correct","encoder_F","decoder_Q")),"Historical off mismatch")
        r["historical_native_equal"]=r["native"]["ids"]==w["output_ids"] and r["native"]["F"]==w["generation_forwards"]
        save(ROOT/f"pair-{i:02}.json",r)
        if j<3:
            a,b,c=[r[k]["tape"][0] for k in ("native","off","on")]
            gates.append((a["raw"]==c["raw"],a["raw"]!=b["raw"]))
        if j==2:
            expand=all(x for x,_ in gates) and any(y for _,y in gates)
            print("ANCHORS",gates,"EXPAND",expand,flush=True)
            if not expand: break
    save(ROOT/"complete.json",dict(n=j+1,pid=os.getpid(),gates=gates))


if __name__=="__main__":
    main()
