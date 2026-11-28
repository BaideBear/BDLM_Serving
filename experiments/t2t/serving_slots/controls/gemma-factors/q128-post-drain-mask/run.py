"""A3-only matched six, using the prior A3 runner and its physical hooks."""
import importlib.util
import json
import os
from pathlib import Path
import sys

from seat import B, Q, check, install

PARENT = Path("/gs/work/gemma-live-q128-a3-7iDh2EqM")
sys.path.insert(1, str(PARENT))
spec = importlib.util.spec_from_file_location("frozen_live_runner", PARENT/"run.py")
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)

MATCH = ("ids", "D", "E", "Q", "F", "events", "route_F", "terminal_eos")
MEASURED = ("physical", "physical_positions", "encoder", "projection_widths_verified")


def match(a, b, keys, message):
    check(not a["error"] and not b["error"], (message, a["error"], b["error"]))
    different = [k for k in keys if live.digest(a[k]) != live.digest(b[k])]
    check(not different, (message, different))


def main(path):
    live.guard(False)
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from bdlm_serving.evaluation import frozen90, scoring
    from bdlm_serving.methods.t2t.serving_profiles import open_strict_gemma
    from bdlm_serving.runtime.diffusiongemma import observe
    from experiments.t2t.serving_slots.gemma_strict_eval import encoder_tape
    from experiments.t2t.serving_slots.gemma_unified_audit import projectors
    from experiments.t2t.serving_slots.mask_pair_gemma import input_digest

    check(torch.cuda.device_count() == 1 and str(torch.__version__) == "2.6.0+cu124"
          and transformers.__version__ == "5.12.0", "Runtime drift")
    check(live.sha(live.SAMPLES.read_bytes()) == "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed"
          and live.sha(Path(scoring.__file__).read_bytes()) ==
          "a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692", "Input/scorer pin")
    refs = json.loads(Path(path).read_bytes())
    check([r["case"]["index"] for r in refs] == list(live.CASES) == [7, 13, 25, 33, 55, 71],
          "Unexpected reference cohort")
    proc = AutoProcessor.from_pretrained(live.MODEL, local_files_only=True)
    tasks = frozen90.load_tasks(live.SAMPLES)
    live.guard(False)
    torch.zeros(1, device="cuda:0")
    live.guard(True)
    live.emit("start", pid=os.getpid(), gpu=live.GPU, q=Q, profile=live.PROFILE, cases=list(live.CASES),
              reference_sha256=live.sha(Path(path).read_bytes()), torch=str(torch.__version__),
              transformers=transformers.__version__, checkpoint=str(live.MODEL),
              dtype="bf16", attention="sdpa", experts="grouped_mm")
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        live.MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
        experts_implementation="grouped_mm", device_map={"": "cuda:0"}).eval()
    check(model.config.canvas_length == B and
          all(p.device == torch.device("cuda:0") and p.dtype == torch.bfloat16 for p in model.parameters()),
          "Checkpoint placement")
    special = set(proc.tokenizer.all_special_ids) | set(model.generation_config.eos_token_id)
    special.add(model.generation_config.pad_token_id)

    for ref in refs:
        live.guard(True)
        old, prompt_ref = ref["case"], ref["prompt"]
        i = old["index"]
        task = tasks[i]
        data = live.prompt(proc, task)
        check(all(task[k] == old[k] == prompt_ref[k] for k in ("index", "task_id", "seed", "dataset"))
              and data["input_ids"][0].tolist() == prompt_ref["prompt_ids"]
              and input_digest(data) == prompt_ref["historical_live_input_sha256"], (i, "prompt"))
        data = data.to("cuda:0")
        args = model, data, task, torch, observe, projectors, encoder_tape

        def arm(mode):
            notes = []

            def open_arm(*a, **kw):
                handle = open_strict_gemma(*a, **kw)
                if mode != "bare":
                    install(handle._request, mode == "treat", notes)
                return handle

            result = live.run(*args, open_arm, True)
            check(not result["error"], (i, mode, result["error"]))
            result["N"] = sum(t not in special for t in result["ids"])
            result["scoring"] = scoring.score(proc.tokenizer.decode(result["ids"], skip_special_tokens=True), task)
            result["correct"] = bool(result["scoring"]["correct"])
            return result, notes

        bare, _ = arm("bare")
        noop, opportunities = arm("noop")
        match(bare, old["live"], MATCH+MEASURED+("N", "correct"), (i, "historical A3"))
        match(bare, noop, MATCH+MEASURED+("N", "correct"), (i, "bare/no-op"))
        live.emit("controls", index=i, bare=bare, noop=noop, opportunities=opportunities)
        treat, swaps = arm("treat")
        check(bool(swaps) == bool(opportunities), (i, "exposure parity"))
        if swaps:
            first = swaps[0]
            t = first["tick"]
            check(first == opportunities[0] and noop["events"][:t] == treat["events"][:t]
                  and noop["physical_positions"][:t] == treat["physical_positions"][:t],
                  (i, "first-fork predecessor"))
            a, b = (set(r["events"][t]["parts"][0]["rows"]) for r in (noop, treat))
            check(a-b == {first["drop"]} and b-a == {first["take"]}
                  and len({s["head"] for s in swaps}) == len(swaps), (i, "measured exchange"))
        else:
            match(noop, treat, MATCH+MEASURED+("N", "correct"), (i, "unexposed"))
        live.emit("case", index=i, treat=treat, swaps=swaps)
    live.emit("complete", requests=len(refs))


if __name__ == "__main__":
    main(sys.argv[1])
