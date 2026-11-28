"""H4-only first-call comparison. Six prefills, 24 decoder calls, no rollout."""
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

D=Path(__file__).resolve().parent
BASE=D.parent
sys.path.insert(0,str(BASE))
from llada_flow_active_w2_study import DATA, DEPLOY, ROOT, REG, check, hashed, js, read, save
from bdlm_serving.methods import flowblock
flowblock.__path__.insert(0,str(BASE))
from bdlm_serving.methods.flowblock import control
from bdlm_serving.methods.flowblock.llada_active_w2 import ActiveW2
from bdlm_serving.runtime.batch import ForwardBatcher, ModelFacade
from experiments.t2t.serving_slots import shared_fixed_shard as shard, shared_llada_worker as worker

NAMES=("dense","repeat","zero_tail","compact")


def fingerprint(t):
    return sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def compare(a,b):
    x,y=a["logits"],b["logits"]
    check(x.shape==y.shape and x.dtype==y.dtype and
          torch.isfinite(x).all() and torch.isfinite(y).all(),"nonfinite or different logit schema")
    diff=(x.float()-y.float()).abs()
    bit=x.contiguous().view(torch.uint8)!=y.contiguous().view(torch.uint8)
    rows=[]
    for i in range(32):
        rows.append(dict(row=i,bit_equal=not bit[i].any().item(),changed_logits=(x[i]!=y[i]).sum().item(),
                         max_abs=diff[i].max().item(),top1_changed=a["top"][i]!=b["top"][i],
                         choice_changed=a["new"][i]!=b["new"][i],
                         acceptance_changed=a["accept"][i]!=b["accept"][i],
                         edit_changed=a["edit"][i]!=b["edit"][i]))
    return dict(rows=rows,max_abs=diff.max().item(),bit_equal=not bit.any().item(),
                decision_vector_equal=all(a[k]==b[k] for k in ("new","fin","pe","frontier")))


def probe(handle,task):
    m,tok=handle["model"],handle["tok"]
    mask_id=handle["mask_id"]
    ref=read(BASE/f'{task["index"]:02d}-flow_w2.json')
    active=read(BASE/f'{task["index"]:02d}-active_w2.json')
    ids,_=worker.canonical_prompt(tok,task,ref["prompt_ids"])
    batch=ForwardBatcher(m,"llada21",1)
    fac=ModelFacade(batch,str(task["index"]),task["seed"])
    engine=ActiveW2(fac,mask_id,handle["eos_id"],dict(confidence_threshold=.5,
                    editing_threshold=.5,max_post_steps=16),compact=False)
    results,inputs={},{}
    class EndFirst(Exception):
        pass

    class ProbeBackend:
        device=m.device
        def __call__(self, unused, **kw):
            check(len(engine.ticks)==1,"not first forward")
            t=engine.ticks[-1]
            check(t["open"]==1 and t["commit"]==0 and t["right"]==1,"not single open head")
            original=kw["past_key_values"]._data
            head=t["head"]
            dst=torch.arange(head,head+64,device=m.device)[None]
            tail=original.shape[4]-64
            token=(ids+[mask_id]*64)[head:head+32]
            pm=[p<len(ids) for p in range(head,head+32)]
            embeds=kw["inputs_embeds"]
            canonical=torch.tensor([token+[mask_id]*32],device=m.device)
            check(torch.equal(embeds,fac.model.word_embeddings(canonical)),"embedded real token IDs")
            k= torch.arange(original.shape[4],device=m.device)
            expected=(k<head)|((k>=tail)&(k<tail+32))
            check((kw["attention_mask"][:,:32]==expected).all(),"visible padded column")
            inputs.update(head=head,tail=tail,kv_sha=fingerprint(original),head_embed_sha=fingerprint(embeds[:,:32]),
                          mask_sha=fingerprint(kw["attention_mask"]),pos_sha=fingerprint(kw["position_ids"]),
                          kv_len=original.shape[4],embed_dtype=str(embeds.dtype),prompt_rows=sum(pm))
            for name in NAMES:
                take=32 if name=="compact" else 64
                t["physical_width"]=take
                new=embeds.clone()
                if name=="zero_tail":
                    new[:,32:].zero_()
                    check(not torch.equal(embeds[:,32:],new[:,32:]),"no perturbation")
                check(torch.equal(embeds[:,:32],new[:,:32]),"changed head embedding")
                data=original.clone()
                args=dict(kw,inputs_embeds=new,past_key_values=SimpleNamespace(_data=data))
                output=engine.backend(None,**args)
                engine.owner._packed_kv_writeback(data,output.past_key_values,dst,tail,64)
                check(t["written_KV"]==take,"wrong scatter")
                logits=output.logits[:,:32].contiguous()
                top,p=engine.owner.decoder._top1_and_confidence(logits)
                loc=torch.tensor([token],device=m.device)
                t["slots"]=[]
                engine.owner._joint_decode_slot_packed(logits,loc,torch.tensor([pm],device=m.device),
                        torch.tensor([0],device=m.device),torch.tensor([False],device=m.device),
                        torch.tensor([True],device=m.device))
                s,=t["slots"]
                if name in ("dense","compact"):
                    case=active if name=="compact" else ref
                    check(s==case["ticks"][0]["slots"][0],"did not reproduce sealed first decision")
                results[name]=dict(logits=logits[0].detach().cpu(),top=top[0].tolist(),confidence=p[0].tolist(),
                    new=s["new"],pe=s["pe"],fin=s["fin"],frontier=s["frontier"],
                    accept=[a==mask_id and b!=a for a,b in zip(token,s["new"])],
                    edit=[a!=mask_id and b!=a for a,b in zip(token,s["new"])],
                    active_written_kv_sha=fingerprint(data[...,head:head+32,:]),
                    tail_embed_sha=fingerprint(new[:,32:]),logit_dtype=str(logits.dtype))
            check(fingerprint(original)==inputs["kv_sha"],"branched state was mutated")
            raise EndFirst()

    engine.owner.model=ProbeBackend()
    tape=[[] for _ in m.model.layers]
    def run():
        try:
            engine.run(torch.tensor([ids],device=m.device),2048)
        except EndFirst:
            return
        raise RuntimeError("continued past first forward")
    with ExitStack() as stack:
        for lay,rec in zip(m.model.layers,tape,strict=True):
            stack.enter_context(lay.attention.query_key_value.register_forward_pre_hook(
                lambda _,args,rec=rec:rec.append(args[0].shape[1])))
        batch.run([run])
    bills=[max(32,len(ids)//32*32),64,64,64,32]
    check(len(tape)==20 and all(x==bills for x in tape) and
          [c["physical_rows"] for c in fac.calls]==bills,"physical bill")
    comparisons={k:compare(results["dense"],results[k]) for k in NAMES[1:]}
    for rec in results.values():
        rec["head_logits_sha256"]=fingerprint(rec.pop("logits"))
    return dict(index=task["index"],seed=task["seed"],task_id=task["task_id"],inputs=inputs,
                prompt_sha256=hashed(ids),layer_q=tape,calls=fac.calls,outputs=results,comparisons=comparisons)


def main():
    reg=read(BASE/"registration.json")
    preg=D/"registration.json"
    check(sha256(preg.read_bytes()).hexdigest()=="d550a8918d35867eb8e4034651713182e6e8ae5b14ea3702e231e8fc7e03ddf1"
          and read(preg)["pilot"]==reg["pilot"],"diagnostic version")
    check(read(BASE/"complete.json")["requests"]==180 and
          read(BASE/"qualified.json")["cases"]==reg["pilot"] and
          read(BASE/"complete.json")["reg"]==REG,"parent not sealed")
    stages=[json.loads(x)["phase"] for x in (BASE/"handoff.ndjson").read_text().splitlines()]
    check(stages==["before","returned"] and not (D/"complete.json").exists(),"parent handoff or duplicate")
    shard.check_sources(read(DEPLOY/"registration.json"))
    shard.check_board(reg["board"])
    h=worker.load(Path("/data0/ysy/models/LLaDA2.1-mini"))
    tasks=shard.load_tasks(DATA/"samples.json")
    rows=[]
    with patch.object(control,"control_directory",lambda _:ROOT/"sources/flowblock-controls"):
        for idx in reg["pilot"]:
            r=probe(h,tasks[idx])
            save(D/f"{idx:02d}-rows.json",r)
            compact={name:dict(equal=c["bit_equal"],max_abs=c["max_abs"],
                         decisions_equal=c["decision_vector_equal"],
                         written_head_KV_equal=r["outputs"]["dense"]["active_written_kv_sha"]==
                            r["outputs"][name]["active_written_kv_sha"],
                         **{key:sum(int(z[key]) for z in c["rows"]) for key in
                            ("top1_changed","acceptance_changed","edit_changed","choice_changed","changed_logits")},
                         changed_rows=sum(not z["bit_equal"] for z in c["rows"])) for name,c in r["comparisons"].items()}
            row=dict(index=idx,comparisons=compact,prefill_Q=r["layer_q"][0][0],
                     layer_q=r["layer_q"][0][1:],layers=len(r["layer_q"]),row_receipt_sha256=hashed(r))
            print(js(row),flush=True)
            rows.append(row)
    save(D/"complete.json",dict(cases=reg["pilot"],prefill_F=6,decoder_F=24,decoder_Q=1344,
                                receipt_sha256=hashed(rows),parent=REG))


if __name__=="__main__":
    main()
