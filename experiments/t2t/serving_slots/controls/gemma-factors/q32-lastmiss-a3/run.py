"""A3 Q32 bare/no-op recorder. Reuse the sealed A3 Live runner's hooks."""
import importlib.util
import os
from pathlib import Path
import sys

from recorder import B, Q, check, exposures, install

CASES = (0, 10, 20)
PARENT = Path("/gs/work/gemma-live-q128-a3-7iDh2EqM")
sys.path.insert(1, str(PARENT))
spec = importlib.util.spec_from_file_location("frozen_live_runner", PARENT/"run.py")
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)
live.Q = Q  # Set the reused runner's fixed offer, not the profile implementation.


def main():
    live.guard(False)
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import EntropyBoundSampler
    from bdlm_serving.evaluation import frozen90, scoring
    from bdlm_serving.methods.t2t.serving_profiles import open_strict_gemma
    from bdlm_serving.runtime.diffusiongemma import observe
    from experiments.t2t.serving_slots.gemma_strict_eval import encoder_tape
    from experiments.t2t.serving_slots.gemma_unified_audit import projectors

    check(torch.cuda.device_count() == 1 and str(torch.__version__) == "2.6.0+cu124"
          and transformers.__version__ == "5.12.0", "Runtime drift")
    check(live.sha(live.SAMPLES.read_bytes()) == "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed"
          and live.sha(Path(scoring.__file__).read_bytes()) ==
          "a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692", "Input/scorer pin")
    proc = AutoProcessor.from_pretrained(live.MODEL, local_files_only=True)
    tasks = frozen90.load_tasks(live.SAMPLES)
    live.guard(False)
    torch.zeros(1, device="cuda:0")
    live.guard(True)
    live.emit("start", pid=os.getpid(), gpu=live.GPU, q=Q, profile=live.PROFILE, cases=CASES,
              torch=str(torch.__version__), transformers=transformers.__version__)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        live.MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
        experts_implementation="grouped_mm", device_map={"": "cuda:0"}).eval()
    check(model.config.canvas_length == B and
          all(p.device == torch.device("cuda:0") and p.dtype == torch.bfloat16 for p in model.parameters()),
          "Checkpoint placement")
    special = set(proc.tokenizer.all_special_ids) | set(model.generation_config.eos_token_id)
    special.add(model.generation_config.pad_token_id)
    for i in CASES:
        live.guard(True)
        task = tasks[i]
        check(task["index"] == i, "Wrong fixed index")
        data = live.prompt(proc, task).to("cuda:0")
        args = model, data, task, torch, observe, projectors, encoder_tape
        tape = []

        def open_noop(*a, **kw):
            h = open_strict_gemma(*a, **kw)
            install(h._request, tape, EntropyBoundSampler, torch)
            return h

        bare = live.run(*args, open_strict_gemma, True)
        noop = live.run(*args, open_noop, True)
        check(not bare["error"] and not noop["error"] and bare == noop,
              (i, "bare/no-op drift", bare["error"], noop["error"]))
        check(len(tape) == bare["D"], "Recorder missed physical tick")
        for t, (row, event) in enumerate(zip(tape, noop["events"])):
            check(row["tick"] == t and row["width"] == event["decoder_Q"]
                  and row["rows"] == event["positions"] and row["sampled"] == event["sampled_absolute"]
                  and row["written_absolute"] == event["written_absolute"], (i, t, "row receipt drift"))
            # The reused runner independently verifies every one of the 85 widths.
            row["layer_physical_q"] = [row["width"]]*noop["projection_widths_verified"]
        counts, swaps = exposures(tape, special)
        live.emit("case", index=i, seed=task["seed"], task_id=task["task_id"], counts=counts, swaps=swaps,
                  controls_identical=True, bare=bare, noop=noop, tape=tape, special=sorted(special))
    live.emit("complete", requests=len(CASES))


if __name__ == "__main__":
    main()
