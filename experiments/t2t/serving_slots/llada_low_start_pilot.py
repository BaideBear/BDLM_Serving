"""Independent CPU-only diagnostic, fixed Q4/Q8, frozen indices 0/10/20."""
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


def main():
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
    root = ev.reg.DATA/"llada21-low-start-cpu3-v1"
    root.mkdir(exist_ok=False)
    sources = {p:sha(ev.REPO/p) for p in (
        "bdlm_serving/methods/t2t/llada_budget_request.py","bdlm_serving/methods/t2t/llada_low_start.py",
        "bdlm_serving/methods/t2t/llada_sparse.py","bdlm_serving/methods/t2t/budget_family.py",
        "bdlm_serving/methods/optimus/upstream.py","bdlm_serving/runtime/serial.py",
        "experiments/t2t/serving_slots/llada_low_start_pilot.py",
        "experiments/t2t/serving_slots/llada_fixed_family_eval.py",
        "experiments/t2t/serving_slots/fixed_cold_exposure.py")}
    save(root/"run.json",dict(indices=INDICES, arms=ARMS, budgets=[4,8], sources=sources,
         samples=sha(samples), scoring=ev.SCORER_SHA, device="cpu",cores=[40,41,42,43],
         checkpoint=str(ev.CKPT),m2t=.9,t2t=.5,cap=ev.reg.CAP))
    print("Validating checkpoints and prompts", flush=True)
    ev.verify_checkpoint(ev.CKPT)
    opts = dict(local_files_only=True, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(ev.CKPT, **opts)
    ev.validate_prompts(tok, [tasks[i] for i in INDICES], ref)
    model = AutoModelForCausalLM.from_pretrained(ev.CKPT, **opts, dtype=torch.bfloat16,
                         attn_implementation="sdpa",device_map={"":"cpu"}).eval()
    install_llada_rectangular_mask(model,allow_selected_cache=True)
    receipts = {}
    for i in INDICES:
        prompt = read(ref/ev.reg.filename(i,"native32"))["prompt_ids"]
        for q in (4,8):
            original = read(ev.OUT/"cpu/law-1-8"/f"{i:02}-q{q}.json")
            require(original["row_visits"]["prefix"] == 0, "K0 comparison requires zero original exposure")
            for arm in ARMS:
                require(all(sha(ev.REPO/p)==h for p,h in sources.items()), "Pilot code changed")
                print(f"Running {i:02} q={q} {arm}", flush=True)
                # The frozen evaluator imports its constructor inside fixed(). Substitute only in
                # this private, single-threaded driver, never in the serving worker or on disk.
                with patch.object(base,"LLaDABudgetRequest",partial(LowStartRequest,arm=arm)):
                    row = ev.fixed(model,tok,tasks[i],prompt,q,ev.Law(0,8))
                exp = exposure(row)
                require(original["input_sha256"] == row["input_sha256"] and
                        (arm != "resident" or (row["phase_F"].get("build",0)>0 and
                         not exp["first_before_full"] and not row["row_visits"]["prefix"])) and
                        (arm != "eager" or all(row[k]==original[k] for k in
                             ("output_ids","score","F","Q","trace"))), "Intervention/control audit failed")
                row.update(arm=arm, exposure=exp, registration_sha256=sha(root/"run.json"))
                dst = root/f"{i:02}-q{q}-{arm}.json"
                save(dst,row)
                receipts[dst.name]=sha(dst)
                print(f"Sealed {i:02} Q{q} {arm}: correct={row['correct']}, N/F/Q="
                      f"{row['N']}/{row['F']}/{row['Q']}, prefix={row['row_visits']['prefix']}", flush=True)
    save(root/"complete.json",dict(files=receipts,registration=sha(root/"run.json")))


if __name__ == "__main__":
    main()
