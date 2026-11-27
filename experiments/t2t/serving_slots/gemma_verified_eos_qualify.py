"""Check fixed replay and changed offers inside real, contiguous Gemma verification."""
import argparse
import json
from pathlib import Path

from .gemma_joint_verify_eval import QUAL, available, check, digest
from .layout_sweep import write
from .strict_handle_qualify import TIERS, drive, offer_cycle, reference_path, single

PROFILE = "dense_stop_joint_verified_eos"
BILL = ("F","decoder_F","decoder_Q","encoder_F","encoder_Q","input_sha256")


class Tape:
    def __init__(self, early):
        self.early=early
        self.ticks,self.widths,self.full,self.block=[],[],[],[]
        self.hit=None

    def install(self,stack,model,eos,origin):
        self.eos,self.origin,self.pending=set(eos),origin,[]
        def top(_,inputs,out):
            self.pending.append(out.logits[0].argmax(-1).tolist())
        stack.enter_context(model.register_forward_hook(top))

    def observe(self,e,rows):
        top,=self.pending
        self.pending.clear()
        check(len(top)==len(rows),"No matching measured argmax")
        self.ticks.append(digest([{k:v for k,v in e.items() if k!="verified_eos_end"},top]))
        self.widths.append((e["route"],len(rows)))
        p=e["parts"][0]
        stop=None
        if p["phase"]=="verify":
            start=len(self.block)
            loc=p["rows"]
            check(p['ordinal']==e['ordinal'] and loc==list(range(start,start+len(loc))),"Verify gap")
            at=dict(zip(rows,top,strict=True))
            self.block.extend(at[self.origin+256*e["ordinal"]+j] for j in loc)
            n=next((i+1 for i,v in enumerate(self.block) if v in self.eos),None)
            if (self.early and n and len(self.block)<256 and e['fullstop']['resident']==1
                    and e['route'] in ('verify','verify_paid')):
                check(start<n and not e["dirty_after_absolute"] and not e["sampled_absolute"]
                      and not e["encoded"],"EOS did not follow a newly paid ordered prefix")
                stop=n
                self.hit=dict(tick=e["tick"],offer=e["offer"],ordinal=e["ordinal"],
                              start=start,end=len(self.block),eos_end=n)
            if e["encoded"]:
                check(len(self.block)==256 and not n,"Premature encoder")
                self.full.extend(self.block)
                self.block=[]
        check(e.get("verified_eos_end")==stop,"Unverified shortcut or missed eligible EOS")

    def finish(self,ids):
        check(not self.pending and self.block,"Missing last verified block")
        n=next((i+1 for i,v in enumerate(self.block) if v in self.eos),len(self.block))
        check(list(ids)==self.full+self.block[:n],"Output differs from measured argmax")
        return self.hit is not None


class ChangeInsideVerify:
    def __init__(self,first,second):
        self.first,self.second=first,second
        self.active=False
        self.changes=0

    def offer(self,tick,head):
        return self.second if self.active else self.first

    def observe(self,e):
        verify=e["parts"][0]["phase"]=="verify"
        if self.active and verify and self.last==self.first:
            self.changes+=1
        self.active=verify and not e["encoded"]
        self.last=e["offer"]

    def finish(self):
        return {"live_changes":self.changes}


def fixed_reference(root,q,i):
    from bdlm_serving.evaluation.frozen import sha
    from .mask_pair_gemma import SAMPLES_SHA
    src=(root/"verified-eos-1128"/f"C{q}-pair-{i:02d}.json" if q<256 else
         reference_path(root,q,i,single.PROFILE))
    rec=json.loads(src.read_bytes())
    check(rec["samples_sha256"]==SAMPLES_SHA and rec["index"]==i,"Wrong frozen history")
    if q<256:
        check(rec["q"]==q and rec["schema"]=="verified-eos-pair-v1","Wrong sparse EOS reference")
        data=rec["on"]
    else:
        data=rec
        check((q==256 and data["budget"]==256 and data["fullstop"]) or
              (q==512 and data["schema"]=="single512-v1" and data["on"] is True),"Wrong dense reference")
    return rec,data,sha(src)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("out","history","checkpoint","samples"):
        ap.add_argument("--"+k,type=Path,required=True)
    ap.add_argument("--deadline",type=int,required=True)
    a=ap.parse_args()
    available(False)
    import torch, transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from bdlm_serving.evaluation.frozen import sha
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from .mask_pair_gemma import SAMPLES_SHA
    check(sha(a.samples)==SAMPLES_SHA and torch.cuda.device_count()==1 and transformers.__version__=="5.12.0"
          and a.checkpoint.name=="diffusiongemma-cd40195","Wrong data or runtime")
    tasks=load_tasks(a.samples)
    a.out.mkdir(parents=True,exist_ok=True)
    check(not any(a.out.iterdir()),"Qualification needs a fresh directory")
    proc=AutoProcessor.from_pretrained(a.checkpoint,local_files_only=True)
    model=DiffusionGemmaForBlockDiffusion.from_pretrained(a.checkpoint,local_files_only=True,dtype=torch.bfloat16,
          attn_implementation="sdpa",experts_implementation="grouped_mm",device_map={"":"cuda:0"}).eval()
    summaries=[]
    with torch.inference_mode():
        for i in QUAL:
            task=tasks[i]
            data=proc.apply_chat_template([{"role":"user","content":task["prompt"]}],tokenize=True,
                                          add_generation_prompt=True,return_dict=True,return_tensors="pt").to(model.device)
            jobs=[(f"fixed{q}",(q,)) for q in TIERS]+[("up",(32,128)),("down",(128,32)),("switch25",offer_cycle()),
                                                      ("park",(512,)*5+(32,)*3+(512,))]
            for name,sched in jobs:
                available(True)
                def run(early):
                    t=Tape(early)
                    b=ChangeInsideVerify(*sched) if name in ("up","down") else None
                    row,ids=drive(model,data,task,sched,deadline=a.deadline,eos_audit=t,boundary=b,
                                  profile=PROFILE if early else single.PROFILE)
                    return row,ids,t
                row,ids,on=run(True)
                if len(sched)==1:
                    rec,ref,ref_sha=fixed_reference(a.history,sched[0],i)
                    check(all(task[k]==rec[k] for k in ("index","task_id","dataset","seed")) and
                          all(row[k]==ref[k] for k in BILL) and
                          row["output_digest"]==(ref["ids"] if "ids" in ref else digest(ref["output_ids"])),
                          "Public handle fixed replay differs")
                    row["reference_sha256"]=ref_sha
                else:
                    old,old_ids,off=run(False)
                    suffix=off.widths[len(on.ticks):]
                    check(ids==old_ids and on.ticks==off.ticks[:len(on.ticks)] and
                          all(r=="verify" for r,w in suffix) and bool(suffix)==bool(on.hit)
                          and row["encoder_F"]==old["encoder_F"] and old["F"]-row["F"]==len(suffix)
                          and old["decoder_Q"]-row["decoder_Q"]==sum(w for r,w in suffix),
                          "Changed common trajectory or did not save exactly the post-EOS suffix")
                    row.update(saved_F=len(suffix),saved_Q=sum(w for r,w in suffix))
                row.update(suite=name,samples_sha256=SAMPLES_SHA,identical=True,early=on.hit,profile=PROFILE)
                write(a.out/f"{name}-{i:02}.json",row)
                summaries.append(row)
                print(name,i,"checked",on.hit,flush=True)
    for name,q in (("up",128),("down",32)):
        check(any(r["suite"]==name and r["early"] and r["early"]["offer"]==q and r["early"]["start"]>0
                  and r["top_boundary"]["live_changes"]>0 for r in summaries),"No verified EOS under changed offer")
    check(len(set().union(*(r["directed_pairs"] for r in summaries if r["suite"]=="switch25")))==25 and
          sum(r["parked_rows_paid_on_downshift"] for r in summaries)>0,"Missing transitions or parked write")
    write(a.out/"complete.json",dict(profile=PROFILE,suites=len(summaries),fixed=15,mixed=12,
                                      cases=list(QUAL),inputs={p.name:sha(p) for p in sorted(a.out.glob("*.json"))}))


if __name__=="__main__":
    main()
