"""LLaDA strict IBS: separate M2T acceptance from optional prefix refresh.

The four arms cross .9/.5 M2T with 0/2 spare prefix rows. All use the same
B32 controller, .5 T2T rule, Q8 physical cap and 2048-token generation cap.
"""
from contextlib import ExitStack
import os
from pathlib import Path
from types import SimpleNamespace

from . import llada_all_slots90 as reg
from .ibs_prefix_pair import audit


ARMS = ((.9,0), (.5,0), (.9,2), (.5,2))
CPU_OUT = reg.DATA / "llada21-strict-q8-m2t-prefix-cpu3-v1"


def generate(model, tokenizer, task, prompt, threshold, k):
    import torch
    from bdlm_serving.evaluation.outcomes import capture_optimus_progress_failure
    from bdlm_serving.methods.optimus.fixed import FixedOptimus
    from bdlm_serving.methods.t2t.online import package
    from bdlm_serving.runtime.batch import ForwardBatcher, ModelFacade

    if (threshold, k) not in ARMS:
        raise ValueError("Unregistered threshold/prefix combination")
    feed = []

    class Observed(ForwardBatcher):
        def execute(self, jobs):
            job, = jobs
            feed.append(job["ids"][0].tolist())
            return super().execute(jobs)

    batch = Observed(model, "llada21", 1)
    facade = ModelFacade(batch, str(task["index"]), task["seed"])
    engine = FixedOptimus(facade, "llada21", 8, "ibs", physical_caps=(8,), prefix_refresh=k)
    engine.unmask.dllm_config.confidence_threshold = threshold
    mask, eos = tokenizer.convert_tokens_to_ids("<|mask|>"), tokenizer.eos_token_id
    x = torch.tensor([prompt], device=model.device)

    def one():
        try:
            return (*engine.run(x, reg.CAP, task["seed"], mask, [eos]), None)
        except RuntimeError as exc:
            return capture_optimus_progress_failure(exc, engine)

    tapes = [[] for _ in model.model.layers]
    with ExitStack() as stack:
        for layer,tape in zip(model.model.layers,tapes,strict=True):
            stack.enter_context(layer.attention.query_key_value.register_forward_pre_hook(
                lambda _, args, t=tape: t.append(args[0].shape[1])))
        (raw, events, config, failure), = batch.run([one])
    widths = [call["physical_rows"] for call in facade.calls]
    reg.require(len(tapes) == 20 and all(t == widths for t in tapes) and
                torch.equal(raw[:,:len(prompt)],x), "Layer width/prompt changed")
    config.update(m2t_threshold=threshold, t2t_threshold=.5)
    policy = f"strict_ibs_q8_m{round(100*threshold)}_prefix{k}"
    work = [(e["phase"],e["q_tokens"]) for e in events if e["phase"]!="prefill"]
    ids = [] if failure else raw[0,len(prompt):len(prompt)+reg.CAP].tolist()
    row = package(tokenizer, ids, work, [eos], set(tokenizer.all_special_ids)|{mask,eos},
                  sum(widths)-sum(q for _,q in work), model="llada21", policy=policy,
                  events=events, calls=facade.calls, forward_inputs=feed, prompt_ids=prompt,
                  generation_cap=reg.CAP, request_failure=failure, config=config,
                  t2t_edits=sum(e.get("edited",0) for e in events),
                  **{key:task[key] for key in reg.KEYS})
    if failure:
        row.update(status="generation_failed", truncated=False)
    audited = audit(row, k)
    return row, audited


def cpu():
    import torch
    import transformers.utils as u
    from typing import TypedDict
    if not hasattr(u, "LossKwargs"):
        u.LossKwargs = TypedDict("LossKwargs", {"num_items_in_batch":int}, total=False)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.evaluation.scoring import score
    from bdlm_serving.runtime.serial import install_llada_rectangular_mask
    from .llada_mask_pilot import CKPT
    from .mask_probe_llada import verify_checkpoint
    from .top_transition90 import resolve_reference, validate_prompts

    reg.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available()
                and os.sched_getaffinity(0)==set(range(36,40)), "Expected reserved CPU pilot lane")
    tasks = load_tasks(reg.DATA/"samples.json")
    subset = [tasks[i] for i in (0,10,20)]
    CPU_OUT.mkdir(mode=0o700,exist_ok=False)
    ref = resolve_reference(SimpleNamespace(samples=reg.DATA/"samples.json"))
    verify_checkpoint(CKPT)
    opts = dict(local_files_only=True,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(CKPT,**opts)
    validate_prompts(tokenizer,subset,ref)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model=AutoModelForCausalLM.from_pretrained(CKPT,**opts,dtype=torch.bfloat16,
                    attn_implementation="sdpa",device_map={"":"cpu"}).eval()
    install_llada_rectangular_mask(model,allow_selected_cache=True)
    reg.save(CPU_OUT/"run.json",dict(samples_sha256=reg.SAMPLES_SHA,indices=[0,10,20],
             arms=ARMS,cap=reg.CAP,q=8,kernel_sha256=reg.sha(Path(__file__)),
             protocol="preselected CPU three-case factorial; not frozen90 quality"))
    summaries=[]
    for task in subset:
        i=task["index"]
        prompt=reg.read(ref/reg.filename(i,"native32"))["prompt_ids"]
        for threshold,k in ARMS:
            row, w=generate(model,tokenizer,task,prompt,threshold,k)
            row["score"] = score(row["output_text"],task) if not row["request_failure"] else {"correct":False}
            reg.save(CPU_OUT/reg.filename(i,row["policy"]),row)
            summary=dict(index=i,threshold=threshold,k=k,correct=row["score"]["correct"],
                       tpf=row["tpf"],efficiency=row["token_efficiency"],failed=bool(row["request_failure"]),
                       max_q=row["max_q"],**w)
            summaries.append(summary)
            print(summary,flush=True)
    reg.save(CPU_OUT/"complete.json",dict(rows=summaries,run_sha256=reg.sha(CPU_OUT/"run.json")))


if __name__=="__main__":
    cpu()
