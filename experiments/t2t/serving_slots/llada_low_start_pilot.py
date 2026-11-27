"""Independent CPU-only diagnostic, fixed Q4/Q8, frozen indices 0/10/20."""
import argparse
from functools import partial
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bdlm_serving.evaluation import scoring
from bdlm_serving.evaluation.frozen90 import load_tasks
from . import llada_fixed_family_eval as ev
from .llada_all_slots90 import require, read, save, sha
from .fixed_cold_exposure import exposure

INDICES = (0, 10, 20)
ARMS = ("resident", "credit", "eager")


def compare_gate(row, ref):
    """Both physical trajectories must be identical up to the first vetoed forced write."""
    found=None
    for n,(new,old) in enumerate(zip(row["trace"],ref["trace"])):
        comparable={k:v for k,v in new.items() if k not in ("deferred_fallback","cold_max_conf")}
        if new.get("deferred_fallback"):
            require(old["admitted"]==new["deferred_fallback"] and not new["written_absolute"] and
                    all(new[k]==old[k] for k in ("rows","sampled","dirty_before","visible_kv")),
                    "First veto was not the original forced admission")
            found=n
            break
        require(comparable==old,"Different trace before first veto")
    if found is None:
        require(row["output_ids"]==ref["output_ids"] and row["F"]==ref["F"] and row["Q"]==ref["Q"],
                "No treatment, but a different output or bill")
    for e in row["trace"]:
        if "deferred_fallback" in e:
            require(e["visible_kv"] < min(row["prefill_Q"]+ev.reg.CAP,(1+e["block"])*32)
                    and e["sampled"]==e["cold"] and not e["prefix"] and
                    (not e["deferred_fallback"] or (e["cold_max_conf"]<=.9 and not e["admitted"])),
                    "Selective fallback applied outside an incomplete cold block")
    return dict(first_veto=found,partial_calls=sum("deferred_fallback" in e for e in row["trace"]),
                vetoed_calls=sum(bool(e.get("deferred_fallback")) for e in row["trace"]),
                confident_early_writes=sum(len(e["admitted"]) for e in row["trace"] if "deferred_fallback" in e))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument("--confident-only",action="store_true",help="Use the sealed eager arm as paired control")
    mode.add_argument("--added60-pair",action="store_true",help="Run the original 1/8 family and the gate on indices 30:90")
    args=ap.parse_args()
    gate, expanded=args.confident_only,args.added60_pair
    arms=("eager","confident") if expanded else ("confident",) if gate else ARMS
    # Interleave the three datasets; the initial 0/10/20 never enter this cohort.
    indices=tuple(30+20*d+t for t in range(20) for d in range(3)) if expanded else INDICES
    law=ev.Law(int(expanded),8)
    import torch
    import transformers, transformers.utils as u
    if not hasattr(u, "LossKwargs"):
        from typing import TypedDict
        u.LossKwargs = TypedDict("LossKwargs", {"num_items_in_batch":int}, total=False)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from bdlm_serving.methods.t2t import llada_budget_request as base
    from bdlm_serving.methods.t2t.llada_low_start import LowStartRequest
    from bdlm_serving.runtime.serial import install_llada_rectangular_mask
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available() and
            os.sched_getaffinity(0) == set(range(40,44)) and
            str(torch.__version__) == "2.9.0+cu128" and transformers.__version__ == "4.57.1" and
            sha(Path(scoring.__file__)) == ev.SCORER_SHA, "Unregistered pilot environment")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    os.umask(0o077)
    samples = ev.reg.DATA/"samples.json"
    require(sha(samples) == ev.reg.SAMPLES_SHA, "Frozen suite changed")
    tasks = load_tasks(samples)
    ref = ev.resolve_reference(SimpleNamespace(samples=samples))
    original_root=ev.reg.DATA/"llada21-low-start-cpu3-v1"
    root = (ev.reg.DATA/"llada21-low-start-gate-added60-v1" if expanded else
            ev.reg.DATA/"llada21-low-start-gate-cpu3-v1" if gate else original_root)
    predecessor=None
    if gate:
        predecessor=read(original_root/"complete.json")
        require(predecessor["registration"]==sha(original_root/"run.json") and len(predecessor["files"])==18,
                "Original cohort is not sealed")
    root.mkdir(exist_ok=False)
    sources = {p:sha(ev.REPO/p) for p in (
        "bdlm_serving/methods/t2t/llada_budget_request.py","bdlm_serving/methods/t2t/llada_low_start.py",
        "bdlm_serving/methods/t2t/llada_sparse.py","bdlm_serving/methods/t2t/budget_family.py",
        "bdlm_serving/methods/optimus/upstream.py","bdlm_serving/runtime/serial.py",
        "experiments/t2t/serving_slots/llada_low_start_pilot.py",
        "experiments/t2t/serving_slots/llada_fixed_family_eval.py",
        "experiments/t2t/serving_slots/fixed_cold_exposure.py")}
    save(root/"run.json",dict(indices=indices, arms=arms, budgets=[4,8], sources=sources,
         samples=sha(samples), scoring=ev.SCORER_SHA, device="cpu",cores=[40,41,42,43],
         **({"law":[1,8],"control":"unmodified fixed family","exclude_pilot":True} if expanded else {}),
         **({"eager_seal":sha(original_root/"complete.json")} if gate else {}),
         checkpoint=str(ev.CKPT),m2t=.9,t2t=.5,cap=ev.reg.CAP))
    print("Validating checkpoints and prompts", flush=True)
    ev.verify_checkpoint(ev.CKPT)
    opts = dict(local_files_only=True, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(ev.CKPT, **opts)
    ev.validate_prompts(tok, [tasks[i] for i in indices], ref)
    model = AutoModelForCausalLM.from_pretrained(ev.CKPT, **opts, dtype=torch.bfloat16,
                         attn_implementation="sdpa",device_map={"":"cpu"}).eval()
    install_llada_rectangular_mask(model,allow_selected_cache=True)
    receipts = {}
    for i in indices:
        prompt = read(ref/ev.reg.filename(i,"native32"))["prompt_ids"]
        for q in (4,8):
            original = None if expanded else read(ev.OUT/"cpu/law-1-8"/f"{i:02}-q{q}.json")
            if not expanded:
                require(original["row_visits"]["prefix"] == 0, "K0 comparison requires zero original exposure")
            for arm in arms:
                require(all(sha(ev.REPO/p)==h for p,h in sources.items()), "Pilot code changed")
                print(f"Running {i:02} q={q} {arm}", flush=True)
                # The frozen evaluator imports its constructor inside fixed(). Substitute only in
                # this private, single-threaded driver, never in the serving worker or on disk.
                if expanded and arm=="eager":
                    row=ev.fixed(model,tok,tasks[i],prompt,q,law)
                    original=row
                else:
                    with patch.object(base,"LLaDABudgetRequest",partial(LowStartRequest,arm=arm)):
                        row = ev.fixed(model,tok,tasks[i],prompt,q,law)
                exp = exposure(row)
                require(original["input_sha256"] == row["input_sha256"] and
                        (arm != "resident" or (row["phase_F"].get("build",0)>0 and
                         not exp["first_before_full"] and not row["row_visits"]["prefix"])) and
                        (arm != "eager" or all(row[k]==original[k] for k in
                             ("output_ids","score","F","Q","trace"))), "Intervention/control audit failed")
                row.update(arm=arm, exposure=exp, registration_sha256=sha(root/"run.json"))
                if gate:
                    path=original_root/f"{i:02}-q{q}-eager.json"
                    require(predecessor["files"][path.name]==sha(path),"Changed eager control")
                    row["treatment"]=compare_gate(row,read(path))
                if expanded and arm=="confident":
                    require(row["law"]==original["law"]==[1,8] and
                            all(row[k]==original[k] for k in ("input_sha256","index","task_id","dataset","seed","q")),
                            "Mismatched expanded control")
                    row["treatment"]=compare_gate(row,original)
                dst = root/f"{i:02}-q{q}-{arm}.json"
                save(dst,row)
                receipts[dst.name]=sha(dst)
                print(f"Sealed {i:02} Q{q} {arm}: correct={row['correct']}, N/F/Q="
                      f"{row['N']}/{row['F']}/{row['Q']}, prefix={row['row_visits']['prefix']}", flush=True)
    save(root/"complete.json",dict(files=receipts,registration=sha(root/"run.json")))


if __name__ == "__main__":
    main()
