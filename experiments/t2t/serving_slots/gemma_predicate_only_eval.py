"""A1: continuous RNG + unrestricted visibility, native-only stopping selector."""
import csv
import os
from contextlib import contextmanager, ExitStack
from pathlib import Path
from time import time
from unittest.mock import patch

from .gemma_cold_frontier_last_eval import board, CKPT, DEADLINE, GPU, SAMPLES, read, sha

if __name__ == "__main__":
    board(False)

from . import gemma_continuous_rng_eval as rng, gemma_fixed_dual as dual
from .gemma_predicate_only import native_canvas, compare
from .gemma_joint_verify_eval import ORDER, check

vis, api, digest = rng.vis, rng.api, rng.digest
ROOT = Path("/gs/results/gemma-predicate-only-a1-1128")
RT = Path("/gs/runtime/gemma-predicate-only-a1-1128")


@contextmanager
def criterion_tape(eos):
    """Observe the actual pinned criterion and its own categorical entropy call."""
    import torch
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import StableAndConfidentStoppingCriteria as C
    cls = torch.distributions.Categorical
    original, entropy = C.__call__, cls.entropy
    log = []

    def criterion(self, canvas, logits, **kw):
        check((self.stability_threshold,self.confidence_threshold)==(1,.005) and
              list(canvas.shape)==[1,256], "Unexpected native stop scope/config")
        prev=self.argmax_canvas_history
        changes=None if prev is None else int((prev[-1]!=canvas).sum().item())
        values=[]
        def tap(dist):
            out=entropy(dist)
            values.append(float(out.mean(-1).item()))
            return out
        with patch.object(cls,"entropy",tap):
            out=original(self,canvas,logits,**kw)
        check(len(values)==1 and bool(out.item())==bool(changes==0 and values[0]<.005), "Wrong native decision")
        ids=canvas[0].tolist()
        end=next((i+1 for i,v in enumerate(ids) if v in eos),256)
        log.append(dict(changed_argmax=changes, mean_entropy=values[0],hit=bool(out.item()),
                        draft=digest(ids),draft_to_eos=digest(ids[:end])))
        return out
    with patch.object(C,"__call__",criterion):
        yield log


def run(model, proc, data, task, enabled, probe):
    events=[]
    base=api.StrictHandle.step
    with ExitStack() as stack:
        log=stack.enter_context(criterion_tape(set(model.generation_config.eos_token_id)))
        if enabled:
            stack.enter_context(patch.object(dual,"Canvas",native_canvas(dual.Canvas)))
        def step(h,q):
            req=h._request
            rows=req.rows
            check(q==req.fixed_q==256 and req.tail is None and not req.rebuild and req.full_head_at_b
                  and req.fullstop and req.stop_observer.criterion is not None, "Non-solo or unqualified arm")
            n=len(log)
            e=base(h,q)
            p,=e["parts"]
            sampled=bool(p["sampled"])
            check(len(log)==n+int(sampled) and len(p["rows"])==256 and
                  (not sampled or len(p["sampled"])==256 and e["route"]=="dense_single") and
                  e["fullstop"]["hit"]==(sampled and log[-1]["hit"]), "Missing actual criterion, projection or clock")
            events.append(dict(ordinal=e["ordinal"],tick=e["tick"],age=p["age_after"],accepted=len(rows.accepted),
                               route=e["route"],dirty=len(rows.dirty),hit=bool(e["fullstop"]["hit"]),
                               criterion=log[-1] if sampled else None))
            return e
        stack.enter_context(patch.object(api.StrictHandle,"step",step))
        r,t=rng.run(model,proc,data,task,True,probe)
    for s,v in zip(t["ticks"],events,strict=True):
        check(s["route"]==v["route"] and s["ordinal"]==v["ordinal"],"Mismatched tapes")
    for j in range(r["encoder_F"]+1):
        samples=[e for e in events if e["ordinal"]==j and e["criterion"]]
        last=samples[-1]
        check(samples[0]["criterion"]["changed_argmax"] is None and
              all(x["criterion"]["changed_argmax"] is not None and x["age"]==k for k,x in enumerate(samples[1:],2))
              and not any(e["hit"] for e in samples[:-1]) and
              (last["hit"] or last["age"]==48 or not enabled and last["accepted"]==256),"Wrong terminal scope")
    return r,t,events


def qualify10(row,native,trace):
    j=row["first"]
    a,b=(row["events"][k] for k in ("off","on"))
    check(j is not None and (a[j]["ordinal"],a[j]["age"],a[j]["accepted"],a[j]["hit"])==(1,8,256,False)
          and b[j+1]["age"]==9 and b[j+1]["hit"],"No #10 native-clock ablation")
    on=row["on"]["tape"]
    strict=[t for t in on if t["route"]=="dense_single"]
    ref=native["tape"]
    measured=[e["criterion"] for e in b if e["criterion"]]
    check(measured==trace and len(trace)==len(ref)==len(strict) and
          all(all(x[k]==y[k] for k in ("ids","clock","draft","pos","sample","accepted"))
              for x,y in zip(strict,ref)), "#10 native denoising or real stopper mismatch")


def main():
    import torch
    import transformers
    from transformers import AutoProcessor,DiffusionGemmaForBlockDiffusion
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as native
    from . import gemma_fullstop, gemma_predicate_only
    board(True)
    prior=read(rng.ROOT/"registration.json")
    with (RT/"validated_rng/pairs.csv").open() as f:
        refs={int(x["index"]):x["row_sha"] for x in csv.DictReader(f)}
    tasks=vis.load_tasks(SAMPLES)
    reg=dict(pid=os.getpid(),start=time(),gpu=GPU,deadline=DEADLINE,boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
             start_ticks=Path("/proc/self/stat").read_text().split(") ")[1].split()[19],
             source={p.name:sha(p) for p in map(Path,(__file__,gemma_predicate_only.__file__,
                dual.__file__,gemma_fullstop.__file__,rng.__file__,vis.__file__,api.__file__))},
             native=sha(Path(native.__file__)),torch=str(torch.__version__),tf=transformers.__version__,
             inputs=sha(SAMPLES),scorer=sha(Path(vis.scoring.__file__)),checkpoint=str(CKPT),
             ref=digest(refs), prior=digest(prior), policy=sha(RT/"registration.md"), launcher=sha(RT/"launch.sh"))
    check(all(reg[k]==prior[k] for k in ("native","torch","tf","inputs","scorer","checkpoint","gpu")) and
          len(tasks)==len(refs)==90 and set(refs)==set(ORDER)==set(range(90)) and
          all(sha(rng.ROOT/f"pair-{i:02}.json")==ref for i,ref in refs.items()) and torch.cuda.device_count()==1,
          "Bad stack, input or prior row")
    ROOT.mkdir()
    vis.save(ROOT/"registration.json",reg)
    print("REGISTERED",reg["pid"],reg["policy"],flush=True)
    proc=AutoProcessor.from_pretrained(CKPT,local_files_only=True)
    board(True)
    model=DiffusionGemmaForBlockDiffusion.from_pretrained(CKPT,local_files_only=True,dtype=torch.bfloat16,
          attn_implementation="sdpa",experts_implementation="grouped_mm",device_map={"":"cuda:0"}).eval()
    for n,i in enumerate(ORDER,1):
        task=tasks[i]
        pp=rng.ROOT/f"pair-{i:02}.json"
        prev=read(pp)
        data=proc.apply_chat_template([{"role":"user","content":task["prompt"]}],tokenize=True,
               add_generation_prompt=True,return_dict=True,return_tensors="pt").to(model.device)
        inp=vis.input_digest(data)
        check(sha(pp)==refs[i] and inp==prev["input_sha"] and all(task[k]==prev[k] for k in ("seed","index","dataset")),
              "Bad case reference")
        row=dict(index=i,seed=task["seed"],dataset=task["dataset"],input_sha=inp,pid=os.getpid(),
                 registration=digest(reg),reference=refs[i],audit={},events={})
        for enabled in ((False,True) if n%2 else (True,False)):
            board(True)
            arm="on" if enabled else "off"
            x,t,e=run(model,proc,data,task,enabled,n<=3)
            if not enabled:
                check((x,t)==(prev["on"],prev["audit"]["on"]),f"#{i} full off replay failed")
            row[arm],row["audit"][arm],row["events"][arm]=x,t,e
        row["first"]=compare(row["events"]["off"],row["events"]["on"],row["audit"]["off"],row["audit"]["on"],
                             row["off"],row["on"])
        if i==10:
            board(True)
            with criterion_tape(set(model.generation_config.eos_token_id)) as nt:
                native_row=vis.run(model,proc,data,task,"native",True)
            nr=vis.ROOT/f"pair-{i:02}.json"
            check(sha(nr)==prev["reference"] and native_row==read(nr)["native"],"Native #10 replay changed")
            row["native_criterion"]=nt
            qualify10(row,native_row,nt)
        vis.save(ROOT/f"pair-{i:02}.json",row)
        if n in (3,30,90):
            stamp=dict(n=n,pid=os.getpid(),registration=digest(reg))
            vis.save(ROOT/f"checkpoint-{n}.json",stamp)
            print("CHECKPOINT",n,flush=True)
    vis.save(ROOT/"complete.json",stamp)


if __name__=="__main__":
    main()
