"""A1-only fixed nine, both budgets; plain upstream replay plus matched off/on."""
from contextlib import ExitStack
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from .gemma_ibs_prefix_bridge import CASES,TIERS,audit,instrument,match,require

GPU="GPU-861c9b5d-1e73-2550-f832-b73988bc64ff"
DEADLINE=1795849200
RT=Path("/gs/runtime/gemma-ibs-prefix-bridge-a1-1128")
RESULT=Path("/gs/results/gemma-ibs-prefix-bridge-a1-1128")
CHECKPOINT=Path("/gs/models/diffusiongemma-cd40195")
SAMPLES=Path("/gs/code/samples.json")
PINS={"runtime":"a09d04477361cb2a260dac85ce2195c0bfee9218306b7cfb96d7ddf4f1dba9af",
      "limits":"842f078e06b5f79b336165b060d1212e947e01182e167379a1d1dbb9f62b5376",
      "scorer":"a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692",
      "samples":"bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed",
      "model_manifest":"ceb951734eb354728d7ba984093739119addbeda981b554af46d5dc105071f53",
      "native":"e7cdf58bf216bfae7462f207983dc0d228122e7056f3e1650a268fc9ce46e768",
      "controls_module":"f21a4704697cace53b19dfbbbf94bbabc877241b9c979e15b0d40e9a2ea0d223"}
CONTROL_HASHES={"sequence.py":"99c80047d517113579e04fd57443471a759a738dc30b206ba1342e0d4f9b3da2",
                "unmasking.py":"bbc3c3328e5497aa08aed6e04150ab0383f95dc06b4ff570a69adaed4a769746"}


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    return sha256(p.read_bytes()).hexdigest()


def digest(x):
    return sha256(json.dumps(x,separators=(",",":"),sort_keys=True).encode()).hexdigest()


def write(p,obj):
    tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,separators=(",",":"),allow_nan=False)+"\n")
    tmp.replace(p)


def board(own):
    def query(flag):
        return subprocess.check_output(["nvidia-smi","-i","1",flag,"--format=csv,noheader,nounits"],text=True).strip()
    require(time.time()<DEADLINE and os.environ.get("CUDA_VISIBLE_DEVICES")==GPU and
            query("--query-gpu=uuid")==GPU,"Deadline or device")
    require(all(own and line==f"{os.getpid()}, {GPU}" for line in
            query("--query-compute-apps=pid,gpu_uuid").splitlines()),"A1 occupied")
    if not own:
        require(query("--query-gpu=memory.used")=="0","A1 memory occupied")


def run(model,proc,data,unmask,controls,task,q,mode):
    import torch
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.runtime import diffusiongemma_optimus as rt
    from .gemma_strict_eval import encoder_tape,projectors
    prompt=data["input_ids"][0].tolist()
    eos=set(model.generation_config.eos_token_id)
    special=set(proc.tokenizer.all_special_ids)|eos|{model.generation_config.pad_token_id}
    torch.manual_seed(task["seed"])
    with ExitStack() as stack:
        stack.enter_context(torch.inference_mode())
        events=stack.enter_context(rt.observe(model))
        enc=stack.enter_context(encoder_tape(model))
        layers,positions=stack.enter_context(projectors(model))
        notes=None if mode=="plain" else stack.enter_context(instrument(rt,model,unmask,q,mode=="on"))
        seq,trace=rt.generate(model,data,controls,unmask,"ibs",q,caps=(q,))
    widths=[x["q"] for x in trace]
    require(len(layers)==85 and all(v==widths for v in layers.values()) and
            positions==[[len(prompt)+i for i in x["positions"]] for x in trace] and
            all(0<w<=q for w in widths),"Wrong physical rows/widths")
    n=sum(t not in special for t in seq)
    score=scoring.score(proc.tokenizer.decode(seq,skip_special_tokens=True),task)
    physical=dict(decoder_F=len(trace),decoder_Q=sum(widths),encoder_F=len(enc)-1,
                  encoder_Q=sum(len(p) for _,p in enc[1:]),F=len(events)-1)
    out=dict(output_ids=seq,trace=trace,events=events,enc=enc,N=n,correct=bool(score["correct"]),
             scoring=score["metric"],**physical)
    if mode!="on":
        f,p=physical["F"],physical["decoder_Q"]+physical["encoder_Q"]
        rt.validate_trace(dict(policy=f"ibs_cap_{q}",events=events,trace=trace,generation_forwards=f,
                               generation_query_rows=p,generation_physical_padded_rows=p,max_q=max(
                                   w for _,w in events[1:]),max_decoder_q=max(widths)))
    if notes is not None:
        bill=audit(notes,trace,events,seq,enc,prompt,eos,q,mode=="on")
        require(all(bill[k]==v for k,v in physical.items()) and
                bill["admissions"]==bill["dirty_paid"]==256*bill["heads"] and
                bill["base_Q"]+bill["extra_Q"]==bill["decoder_Q"],"Receipt imbalance")
        out.update(notes=notes,bill=bill)
    return out


def main():
    board(False)
    import torch
    from transformers import AutoProcessor,DiffusionGemmaForBlockDiffusion
    from bdlm_serving.evaluation import frozen,frozen90,scoring
    from bdlm_serving.methods.optimus import upstream
    from bdlm_serving.runtime import diffusiongemma_optimus as rt,optimus_limits
    from .mask_pair_gemma import input_digest
    snapshot=read(RT/"checkpoint.json")
    paths=dict(runtime=rt.__file__,limits=optimus_limits.__file__,scorer=scoring.__file__,
               samples=SAMPLES,model_manifest=frozen.ROOT/"benchmarks/checkpoints/diffusiongemma.json",
               native=frozen.ROOT/"archive/diffusiongemma-native/config.json",controls_module=upstream.__file__)
    require({k:sha(Path(v)) for k,v in paths.items()}==PINS and
            snapshot["manifest"]==PINS["model_manifest"] and snapshot["checkpoint"]==str(CHECKPOINT) and
            snapshot["entries"]==19 and version("math-verify")=="0.8.0" and version("transformers")=="5.12.0"
            and str(torch.__version__)=="2.10.0+cu128" and torch.cuda.device_count()==1,"Wrong pinned environment")
    controls,strategy,hashes=upstream.load_controls()
    require(hashes==CONTROL_HASHES and upstream.COMMIT=="7343a37b2a69337caa452f17b59727dc6665f039",
            "Wrong Optimus controls")
    tasks=frozen90.load_tasks(SAMPLES)
    require([tasks[i]["dataset"] for i in CASES]==["gsm8k","math500","humaneval"]*3,"Wrong frozen nine")
    spec=dict(confidence_threshold=.9,block_length=256,denoising_steps=256,
              unmasking_strategy=strategy.LOW_CONFIDENCE_DYNAMIC)
    unmask=controls.UnmaskingProcessor(SimpleNamespace(**spec))
    reg=dict(pid=os.getpid(),uuid=GPU,boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
             start_ticks=Path("/proc/self/stat").read_text().split(") ")[1].split()[19],
             deadline=DEADLINE,registration=sha(RT/"registration.md"),launch=sha(RT/"launch.sh"),
             pins=PINS,model=snapshot,optimus=hashes,torch=str(torch.__version__),
             case_indices=CASES,tiers=TIERS,sources={k:sha(Path(__file__).with_name(k+".py")) for k in
                ("gemma_ibs_prefix_bridge","gemma_ibs_prefix_bridge_eval")})
    board(True)
    RESULT.mkdir(mode=0o700)
    write(RESULT/"registration.json",reg)
    print("REGISTERED",reg["pid"],flush=True)
    proc=AutoProcessor.from_pretrained(CHECKPOINT,local_files_only=True)
    board(True)
    model=DiffusionGemmaForBlockDiffusion.from_pretrained(CHECKPOINT,local_files_only=True,dtype=torch.bfloat16,
        attn_implementation="sdpa",experts_implementation="grouped_mm",device_map={"":"cuda:0"}).eval()
    require(model.config.canvas_length==256,"Not B256")
    # Interleave budgets; reverse off/on order alternately within each tier.
    durations=[]
    for j,i in enumerate(CASES):
        t=tasks[i]
        data=proc.apply_chat_template([{"role":"user","content":t["prompt"]}],tokenize=True,
                      add_generation_prompt=True,return_dict=True,return_tensors="pt").to(model.device)
        for b,q in enumerate(TIERS):
            if durations and DEADLINE-time.time()<1.5*max(durations)+10:
                write(RESULT/"partial.json",dict(n=len(durations),reg=digest(reg)))
                return
            st=time.monotonic()
            pair={}
            for mode in (("plain","off","on") if (j+b)%2==0 else ("on","plain","off")):
                board(True)
                pair[mode]=run(model,proc,data,unmask,controls,t,q,mode)
            plain=pair.pop("plain")
            require(plain=={k:v for k,v in pair["off"].items() if k not in ("notes","bill")},
                    "Original same-board IBS did not replay")
            first=match(pair["off"],pair["on"])
            pair.update(index=i,seed=t["seed"],dataset=t["dataset"],task_id=t["task_id"],q=q,
                        eos=sorted(model.generation_config.eos_token_id),input_sha=input_digest(data),
                        prompt_sha=digest(data["input_ids"][0].tolist()),first=first,plain_sha=digest(plain),
                        reg=digest(reg),pid=reg["pid"])
            write(RESULT/f"q{q}-{i:02}.json",pair)
            durations.append(time.monotonic()-st)
            print("SEALED",i,q,first["first"],pair["on"]["bill"]["extra_Q"],flush=True)
    write(RESULT/"complete.json",dict(n=18,reg=digest(reg)))


if __name__=="__main__":
    main()
