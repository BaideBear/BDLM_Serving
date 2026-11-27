"""Matched fixed90_fused sparse runs, with independent physical and EOS tapes."""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import time

from .gemma_joint_verify_eval import GPU, ORDER, QUAL, available, check, digest
from .layout_sweep import write


def history(root,q,i):
    if q==128:
        return root/"profile-close-1128"/f"C128-profile-fuse1-{i:02}.json"
    return root/"paid-close-1128"/f"C{q}-fuse1-{i:02}.json"


def run(model,data,task,q,early):
    from bdlm_serving.runtime.diffusiongemma import observe
    from .gemma_strict_eval import encoder_tape
    from .gemma_unified_audit import projectors
    from .gemma_verified_eos import B, VerifiedEOSRequest
    from .mask_pair_gemma import input_digest
    prompt=data["input_ids"][0].tolist()
    origin=len(prompt)
    route_f,route_q=Counter(),Counter()
    tape, ledger, enc_expected=[],[("encoder",origin)],[(prompt,list(range(origin)))]
    tokens, sealed, debt=[],[],set()
    top=[]  # Independent argmax from the actual model output.
    age=0
    hit=None
    with ExitStack() as stack:
        physical=stack.enter_context(observe(model))
        proj,pos=stack.enter_context(projectors(model))
        enc=stack.enter_context(encoder_tape(model))
        stack.enter_context(model.register_forward_hook(lambda _,a,out: top.append(out.logits[0].argmax(-1).tolist())))
        req=VerifiedEOSRequest(model,data,seed=task["seed"],q=q,early=early)
        while req.status=="ready":
            tick=len(tape)
            check(tick<4096,"Unbounded request")
            e=req.step()
            p,=e["parts"]
            route=e["route"]
            rows=pos[-1]
            width=len(rows)
            ordinal=len(enc_expected)-1
            base=origin+B*ordinal
            check(len(pos)==len(top)==tick+1 and e["tick"]==tick and e["offer"]==q and
                  e["q"]==e["decoder_Q"]==len(top[-1])==width<=q and e["decoder_F"]==1 and
                  e["ordinal"]==p["ordinal"]==ordinal and e["positions"]==rows==[base+j for j in p["rows"]]
                  and rows==sorted(set(rows)) and len(proj)==85 and all(v[-1]==width for v in proj.values())
                  and e["opened"] is None and e["promoted"] is None,"Physical position, width or head mismatch")
            sampled=e["sampled_absolute"]
            writes={base+j for k in ("admitted","revoked","input_changed") for j in p[k]}
            check(e["dirty_before_absolute"]==sorted(debt) and set(sorted(debt)[:q])<=set(rows)
                  and sorted(writes)==e["written_absolute"] and writes<=set(sampled)<=set(rows)
                  and p["age_before"]==age and p["age_after"]==age+bool(sampled),"Dirty or clock mismatch")
            debt=(debt-set(rows))|writes
            age=p["age_after"]
            check(e["dirty_after_absolute"]==sorted(debt),"Lost write")
            ledger.append(("decoder",width))
            check(e["encoder_F"]==int(e["encoded"]) and e["encoder_Q"]==B*e["encoder_F"],"Encoder billing")
            route_f[route]+=1
            route_q[route]+=width
            # Equal fingerprints through the actual EOS chunk include sampling changes and raw argmax.
            tape.append(digest([{k:v for k,v in e.items() if k!="verified_eos_end"},top[-1]]))
            if route not in ("verify","verify_paid"):
                check(not sealed and not e["encoded"] and "verified_eos_end" not in e,"Draft short circuit")
                continue
            start=len(sealed)
            check(p["phase"]=="verify" and not sampled and not debt and width==q and
                  rows==list(range(base+start,base+start+q)) and start+q<=B,"Unordered or unpaid verification")
            sealed.extend(top[-1])
            local=next((k for k in range(start,start+q) if sealed[k] in req.eos),None)
            if local is not None and hit is None:
                hit=dict(tick=tick,ordinal=ordinal,eos_row=local,chunk_start=start,chunk_end=start+q,
                         theoretical_F=(B-start-q)//q,theoretical_Q=B-start-q)
            cut=bool(early and hit and len(sealed)<B)
            check(("verified_eos_end" in e)==cut and (not cut or e["verified_eos_end"]==local+1),
                  "Missed or spurious partial termination")
            if cut or len(sealed)==B:
                tokens.extend(sealed if hit is None else sealed[:hit["eos_row"]+1])
                terminal=hit is not None or len(tokens)==2048
                check(e["encoded"]== (not terminal) and (req.status=="done")==terminal,
                      "Encoder or termination crossed an EOS boundary")
                if e["encoded"]:
                    check(len(sealed)==B and not hit,"Incomplete nonterminal commit")
                    enc_expected.append((sealed,list(range(base,base+B))))
                    ledger.append(("encoder",B))
                    age=0
                sealed=[]
        check(not debt and not sealed and req.tokens==tokens and (tokens[-1] in req.eos or len(tokens)==2048)
              and enc==enc_expected and physical==ledger and all(len(v)==len(tape) for v in proj.values()),
              "Final token, measured argmax, encoder input or physical bill mismatch")
    return dict(output_ids=tokens,tape=tape,input_sha256=input_digest(data),hit=hit,encoder_digest=digest(enc),
                decoder_F=len(tape),decoder_Q=sum(route_q.values()),encoder_F=len(enc)-1,
                encoder_Q=B*(len(enc)-1),F=len(tape)+len(enc)-1,route_F=dict(route_f),route_Q=dict(route_q))


def compare(off,on,h,q):
    for k in ("output_ids","correct","N","decoder_F","decoder_Q","encoder_F","encoder_Q","F","route_F","route_Q","input_sha256"):
        check(off[k]==h[k],f"Fixed history replay failed: {k}")
    for k in ("output_ids","correct","N","encoder_F","encoder_Q","encoder_digest","input_sha256","hit"):
        check(on[k]==off[k],f"Pair changed {k}")
    hit=off["hit"]
    f,w=(hit["theoretical_F"],hit["theoretical_Q"]) if hit else (0,0)
    check(on["tape"]==off["tape"][:len(on["tape"])] and off["F"]-on["F"]==off["decoder_F"]-on["decoder_F"]==f
          and off["decoder_Q"]-on["decoder_Q"]==w==f*q
          and (not hit or len(on["tape"])==hit["tick"]+1),"Unaligned shared trajectory or wrong saving")
    for k,delta in (("route_F",f),("route_Q",w)):
        a=Counter(on[k])
        a["verify"]+=delta
        check(a==Counter(off[k]),"Only ordinary post-EOS terminal chunks may be removed")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("out","history","samples","checkpoint"):
        ap.add_argument("--"+k,type=Path,required=True)
    ap.add_argument("--deadline",type=int,required=True)
    a=ap.parse_args()
    available(False)
    import torch, transformers
    from transformers import AutoProcessor,DiffusionGemmaForBlockDiffusion
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen import sha
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from .mask_pair_gemma import SAMPLES_SHA
    check(sha(a.samples)==SAMPLES_SHA and torch.cuda.device_count()==1 and transformers.__version__=="5.12.0"
          and a.checkpoint.name=="diffusiongemma-cd40195","Expected registered A7 configuration")
    sources={n:sha(Path(__file__).with_name(n+".py")) for n in ("gemma_verified_eos","gemma_verified_eos_eval",
             "gemma_fixed_dual","gemma_strict_sampler","gemma_strict_request","gemma_unified_audit","gemma_strict_rows")}
    tasks=load_tasks(a.samples)
    a.out.mkdir(parents=True,exist_ok=True)
    proc=AutoProcessor.from_pretrained(a.checkpoint,local_files_only=True)
    available(False)
    model=DiffusionGemmaForBlockDiffusion.from_pretrained(a.checkpoint,local_files_only=True,dtype=torch.bfloat16,
          attn_implementation="sdpa",experts_implementation="grouped_mm",device_map={"":"cuda:0"}).eval()
    special=set(proc.tokenizer.all_special_ids)|set(model.generation_config.eos_token_id)
    scorer=sha(Path(scoring.__file__))
    with torch.inference_mode():
        for q in (32,64,128):
            opportunities=0
            for j,i in enumerate(ORDER):
                available(True)
                check(time.time()<a.deadline,"Time limit")
                dst=a.out/f"C{q}-pair-{i:02}.json"
                src=history(a.history,q,i)
                if dst.exists():
                    r=json.loads(dst.read_bytes())
                    check(r["source"]==sources and r["reference_sha256"]==sha(src) and r["index"]==i and r["q"]==q,
                          "Cannot resume mixed run")
                else:
                    task=tasks[i]
                    h=json.loads(src.read_bytes())
                    check(h["budget"]==q and h["gate"]=="off" and h["fuse"] is True and h["slack"] is False
                          and h["prefix"]==[1,4 if q==128 else 8] and h["samples_sha256"]==SAMPLES_SHA
                          and h["scorer_sha256"]==scorer and h["torch"]==str(torch.__version__)
                          and all(h[k]==task[k] for k in ("index","seed","dataset","task_id")),"Wrong reference")
                    data=proc.apply_chat_template([{"role":"user","content":task["prompt"]}],tokenize=True,
                              add_generation_prompt=True,return_dict=True,return_tensors="pt").to(model.device)
                    runs={}
                    for early in ((False,True) if j%2==0 else (True,False)):
                        row=run(model,data,task,q,early)
                        ids=row["output_ids"]
                        text=proc.tokenizer.decode(ids,skip_special_tokens=True)
                        check(text==h["output_text"],"Delivery text drift")
                        row.update(N=sum(x not in special for x in ids),correct=bool(scoring.score(text,task)["correct"]))
                        runs[early]=row
                    off,on=runs[False],runs[True]
                    compare(off,on,h,q)
                    r={k:task[k] for k in ("index","seed","dataset","task_id")}
                    r.update(schema="verified-eos-pair-v1",gpu=GPU,q=q,source=sources,samples_sha256=SAMPLES_SHA,
                             scorer_sha256=scorer,reference_sha256=sha(src),off=off,on=on)
                    write(dst,r)
                f=r["off"]["F"]-r["on"]["F"]
                if j<3:
                    opportunities+=int(f>0)
                if j==2:
                    check(ORDER[:3]==list(QUAL) and opportunities>0,"No exercised EOS remainder on anchors")
                    print("QUALIFIED",q,opportunities,"/ 3",flush=True)
                if j<3 or (j+1)%10==0:
                    print("PAIRS",q,j+1,i,f,flush=True)


if __name__=="__main__":
    main()
