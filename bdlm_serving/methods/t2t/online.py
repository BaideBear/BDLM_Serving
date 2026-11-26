"""Run one frozen request at a time and derive metrics from observed model work."""
from collections import Counter
import json

from bdlm_serving.evaluation.frozen import ROOT, sha
from bdlm_serving.methods.t2t.context_gate import ADMISSION, CONTEXT


def package(tok, ids, work, eos, special, prefill, **extra):
    end = next((i+1 for i, x in enumerate(ids) if x in eos), None)
    seq = list(ids[:end])
    n = sum(x not in special for x in seq)
    stages, queries = Counter(), Counter()
    for stage, rows in work:
        if rows <= 0:
            raise ValueError((stage, rows))
        stages[stage] += 1
        queries[stage] += rows
    f, q = sum(stages.values()), sum(queries.values())
    return dict(status="success", output_ids=seq, output_text=tok.decode(seq, skip_special_tokens=True),
                output_tokens=n, generation_forwards=f, generation_query_rows=q,
                generation_physical_padded_rows=q, max_q=max(rows for _, rows in work),
                tpf=n/f, token_efficiency=n/q, eos=end is not None, truncated=end is None,
                prefill_rows=prefill, forwards_by_phase=dict(stages), q_by_phase=dict(queries),
                q_histogram=dict(Counter(str(rows) for _, rows in work)), **extra)


def llada_backend(args, prepare, *, context_ranker=None, admission_fit=None):
    import torch
    import transformers.utils as utils
    if not hasattr(utils, "LossKwargs"):
        from typing import TypedDict
        class LossKwargs(TypedDict, total=False):
            num_items_in_batch: int
        utils.LossKwargs = LossKwargs
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bdlm_serving.runtime.serial import install_llada_rectangular_mask
    from .llada_sparse import LLaDASparseKernel, qualify_full_budget

    for entry in json.loads((ROOT / "benchmarks/checkpoints/llada21.json").read_text()):
        path = args.checkpoint / entry["name"]
        if path.stat().st_size != entry["bytes"] or ("sha256" in entry and sha(path) != entry["sha256"]):
            raise ValueError(f"Checkpoint mismatch: {path.name}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, trust_remote_code=True,
        local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    install_llada_rectangular_mask(model, allow_selected_cache=True)
    if prepare:
        for p in model.parameters():
            p.data = p.data.clone()
        prepare(model)
    model.to("cuda")
    eos = [tok.eos_token_id]
    mask = tok.convert_tokens_to_ids("<|mask|>")
    special = set(tok.all_special_ids) | set(eos) | {mask}
    cfg = dict(confidence_threshold=.5, editing_threshold=.5, max_post_steps=16)
    weights = None
    rates = None
    if getattr(args, "exchange_rates", None):
        from .exchange import read_rates
        rates = read_rates(args.exchange_rates)
    if getattr(args, "hazard_weights", None):
        from .hazard import read_weights
        weights = read_weights(args.hazard_weights)

    @torch.inference_mode()
    def run(task, arm, *, next_budget=None, prompt_reference=None):
        prompt = tok.apply_chat_template([dict(role="user", content=task["prompt"])],
                                         tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        ref = (dict(prompt_ids=prompt_reference) if prompt_reference is not None and arm is not None else
               json.loads((args.reference / f"{task['index']:02d}-native32.json").read_text()))
        if ids[0].tolist() != ref["prompt_ids"]:
            raise ValueError("Changed prompt")
        parameters = (ids, cfg, 2048, task["seed"], mask, eos)
        if arm is None and task["index"] == 0:
            # Additional differential check of per-step logits/RNG/KV, not study work.
            qualify_full_budget(model, *parameters, reference=ref)
        opts = {} if arm is None else dict(fresh_slots=arm.fresh, mask_slots=arm.mask, policy=arm.selection)
        kernel = LLaDASparseKernel(model, 32 if arm is None else arm.budget, hazard_weights=weights,
                                   exchange_rates=rates,
                                   admission_fit=admission_fit if arm and arm.selection in ADMISSION else None,
                                   context_ranker=context_ranker if arm and arm.selection in CONTEXT else None,
                                   **opts)
        state = kernel.create(*parameters,next_budget=next_budget)
        kernel.run(state)
        path = state.events
        if arm is None:
            if len(path) != len(ref["events"]) or any(any(e.get(k) != v for k, v in old.items())
                    for e, old in zip(path, ref["events"])):
                raise ValueError(f"Full-width per-step native event mismatch on {task['index']}")
        if not all(e["F"] == 1 and e["actual_q"] == e["physical_q"]
                   and all(n == e["actual_q"] for n in e["layer_physical_q"]) for e in path):
            raise ValueError("Mismatch between requested and executed transformer rows")
        work = [(e["stage"], e["physical_q"]) for e in path if e["phase"] != "prefill"]
        return package(tok, kernel.output(state)[0].tolist(), work, eos, special,
                       sum(e["physical_q"] for e in path if e["phase"] == "prefill"),
                       t2t_edits=sum(e.get("edited", 0) for e in path), events=path)
    return run


def load_backend(args, prepare, *, context_ranker=None, admission_fit=None):
    if args.model == "llada21":
        return llada_backend(args, prepare, context_ranker=context_ranker, admission_fit=admission_fit)
    if admission_fit is not None:
        raise ValueError('Admission study is LLaDA B32 only')
    return gemma_backend(args, prepare)


def gemma_backend(args, prepare):
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from bdlm_serving.runtime.diffusiongemma import STUDY, check_snapshot
    from .gemma_sparse import sparse_gemma
    plan = json.loads((STUDY / "config.json").read_text())
    if transformers.__version__ != plan["transformers"]:
        raise ValueError(f"Requires transformers {plan['transformers']}")
    check_snapshot(args.checkpoint)
    proc = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(args.checkpoint, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation=plan["attention"],
        experts_implementation=plan["experts"], device_map={"": "cpu" if prepare else "cuda"}).eval()
    if model.config.canvas_length != plan["canvas"] or model.generation_config.max_denoising_steps != 48:
        raise ValueError("Unexpected canvas or denoising configuration")
    if prepare:
        prepare(model)
    model.to("cuda")
    tok, cfg = proc.tokenizer, model.generation_config
    eos = set(cfg.eos_token_id)
    special = set(tok.all_special_ids) | eos | {cfg.pad_token_id}

    @torch.inference_mode()
    def run(task, arm, *, next_budget=None):
        data = proc.apply_chat_template([dict(role="user", content=task["prompt"])],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
        torch.manual_seed(task["seed"])
        opts = ({} if arm is None else dict(fresh_slots=arm.fresh, mask_slots=arm.mask,
                                            policy=arm.selection, selection_seed=task["seed"]))
        with sparse_gemma(model, plan["canvas"] if arm is None else arm.budget, next_budget=next_budget, **opts) as ledger:
            out = model.generate(**data, max_new_tokens=plan["cap"])
        prompt = data["input_ids"]
        plen = prompt.shape[1]
        if not torch.equal(out.sequences[:, :plen], prompt) or ledger.events[:1] != [("encoder", plen)]:
            raise ValueError("Changed prompt or prefill rows")
        suffix = out.sequences[0, plen:].tolist()
        enc = sum(phase == "encoder" for phase, _ in ledger.events)
        if not 0 < len(suffix) <= plan["cap"] or len(suffix) != enc * plan["canvas"]:
            raise ValueError("Generated canvas count disagrees with physical encoder calls")
        if len(suffix) != plan["cap"] and not any(t in eos for t in suffix):
            raise ValueError("Generation ended without EOS or length cap")
        if arm is None:
            ref = json.loads((args.reference / f"{task['index']:02d}-native256.json").read_text())
            if [list(event) for event in ledger.events] != ref["events"]:
                raise ValueError(f"Full-width native event mismatch on {task['index']}")
        return package(tok, suffix, ledger.work, eos, special, plen, events=ledger.trace,
                       physical_events=ledger.events, t2t_edits=sum(e["edited"] for e in ledger.trace))
    return run
