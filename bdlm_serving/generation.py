"""Model execution for the three archived experiments; never manages devices."""
import collections
import gc
import json
import time
import types


def run(args, tasks, plan, before_cuda=None, *, cohort_done=None):
    import torch
    from bdlm_serving.methods.optimus.boundary import BoundaryModel
    import transformers.utils as utils
    if not hasattr(utils, "LossKwargs"):
        from typing import TypedDict
        class LossKwargs(TypedDict, total=False):
            num_items_in_batch: int
        utils.LossKwargs = LossKwargs
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bdlm_serving.runtime.serial import SerialKernel, install_llada_rectangular_mask
    from bdlm_serving.methods.flowblock.control import FlowBlockControl
    from bdlm_serving.methods.flowblock.prefix_cache import FlowBlockPrefix
    from bdlm_serving.runtime.phase_batch import PhaseBatcher
    from bdlm_serving.runtime.audited_batch import AuditedBatcher
    from bdlm_serving.methods.b3.llada import NativeB3Kernel
    from bdlm_serving.methods.optimus.fixed import FixedOptimus
    from bdlm_serving.methods.b3.sdar import SDARB3
    from bdlm_serving.runtime.batch import ForwardBatcher, ModelFacade
    from bdlm_serving.evaluation.outcomes import capture_optimus_progress_failure
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_path = args.checkpoint.resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,
                                              local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    is_llada = args.model in ("llada20", "llada21")
    checkpoint_forward = model.model.forward if args.model == "llada20" else None
    if is_llada:
        install_llada_rectangular_mask(model, allow_selected_cache=True)
        mask_id, eos = tokenizer.convert_tokens_to_ids("<|mask|>"), [tokenizer.eos_token_id]
    else:
        from bdlm_serving.runtime.sdar_numerics import configure_sdar_rmsnorm
        configure_sdar_rmsnorm(4)
        mask_id = tokenizer.convert_tokens_to_ids("<|MASK|>")
        eos = json.loads((model_path/"generation_config.json").read_text())["eos_token_id"]
        if isinstance(eos, int):
            eos = [eos]

    if before_cuda:
        before_cuda(model)
    else:
        model.to("cuda")
    model._erqw_execution_binding = args.model + "/fixed-baselines"
    llada_config = {"confidence_threshold": .5, "editing_threshold": .5, "max_post_steps": 16}
    if args.model == "llada20":
        llada_config = {"confidence_threshold": .95, "editing_threshold": 1., "max_post_steps": 0}
    sdar_config = {"threshold": .9, "temperature": .6, "top_k": 20, "top_p": .95}
    special = set(tokenizer.all_special_ids) | {mask_id} | set(eos)
    encoded = {}
    for task in tasks:
        text = tokenizer.apply_chat_template([{"role": "user", "content": task["prompt"]}],
                                             tokenize=False, add_generation_prompt=True)
        encoded[task["index"]] = tokenizer(text, return_tensors="pt",
            add_special_tokens=False).input_ids.to("cuda")

    def invoke(batcher, task, policy, cap, phase):
        with torch.inference_mode():
            index, seed = task["index"], task["seed"]
            request_id = f"{phase}/{index:02d}/{policy}"
            facade = ModelFacade(batcher, request_id, seed)
            prompt = encoded[index]
            sampler = types.SimpleNamespace(
                sample_with_temperature_topk_topp=lambda logits, **kw: facade.sdar_sample(logits))
            start = time.perf_counter()
            request_failure = None
            if policy.startswith("optimus_bobs_c"):
                engine = FixedOptimus(BoundaryModel(facade), args.model, int(policy.split("_")[2][1:]), "obs")
                try:
                    raw, events, extra = engine.run(prompt, cap, seed, mask_id, eos)
                except RuntimeError as error:
                    raw, events, extra, request_failure = capture_optimus_progress_failure(error, engine)
                extra["native_b32_decode_mask"] = True
            elif policy == "native32_cached" and args.model == "llada20":
                from bdlm_serving.runtime.llada20_cached import cached_generate
                raw, events, extra = cached_generate(model, facade, prompt, cap, seed, mask_id, eos[0])
            elif policy == "native32" and args.model == "llada20":
                from bdlm_serving.runtime.llada20 import checkpoint_generate
                raw, events, extra = checkpoint_generate(model, checkpoint_forward, facade, prompt,
                                                         cap, seed, mask_id, eos[0])
            elif policy == "native32":
                engine = SerialKernel(facade, args.model, sampler)
                state = engine.create(prompt, llada_config if args.model == "llada21" else sdar_config,
                                      cap, seed, mask_id, eos, cached=True)
                engine.run(state)
                raw, events, extra = state.x, state.events, {"native_serial": True}
            elif policy.startswith("optcap_"):
                _, mode, *fields = policy.split("_")
                caps = tuple(map(int,fields))
                engine = FixedOptimus(facade, args.model, max(caps), mode, physical_caps=caps)
                try:
                    raw, events, extra = engine.run(prompt, cap, seed, mask_id, eos)
                except RuntimeError as error:
                    raw, events, extra, request_failure = capture_optimus_progress_failure(error, engine)
            elif policy.startswith("optimus"):
                _, mode, chunk = policy.split("_")
                engine = FixedOptimus(facade, args.model, int(chunk[1:]), mode)
                try:
                    raw, events, extra = engine.run(prompt, cap, seed, mask_id, eos)
                except RuntimeError as error:
                    raw, events, extra, request_failure = capture_optimus_progress_failure(error, engine)
            elif policy.startswith(("flow", "prefix", "disabled")):
                config = llada_config if is_llada else {
                    "confidence_threshold": .9, "editing_threshold": 2., "max_post_steps": 0}
                klass = FlowBlockControl if policy.startswith("flow") else FlowBlockPrefix
                flags = {"enabled": not policy.startswith("disabled")} if klass is FlowBlockPrefix else {}
                engine = klass(facade, int(policy.split("_w")[-1]), mask_id, eos[0], config,
                               block_size=plan["flowblock_block_size"],
                               ignore_prompt_eos=args.model == "sdar", **flags)
                raw, events, extra = engine.run(prompt, cap)
                extra["model_sampler"] = "greedy" if is_llada else "SDAR .6/top20/p.95"
            elif policy == "b3_native32" and args.model == "llada21":
                engine = NativeB3Kernel(facade, promotion_post="carry", entropy_promotion="reset")
                state = engine.create(prompt, llada_config, cap, seed, mask_id, eos, cached=True)
                engine.run(state)
                raw, events, extra = state.x, state.events, {"paper_derived": True, "block_size": 32}
            elif policy == "b3_native32":
                if args.model == "llada20":
                    engine = SDARB3(facade, facade.llada20_sample, threshold=.95, kind="llada20")
                else:
                    engine = SDARB3(facade, facade.sdar_sample)
                raw, events, extra = engine.run(prompt, cap, seed, mask_id, eos)
            else:
                raise ValueError(f"Unknown policy: {policy}")
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter()-start)*1000
            assert torch.equal(raw[:, :prompt.shape[1]], prompt)
            for event in events:
                if event["phase"] == "flowblock_prefix":
                    assert not set(event["absolute_positions"]) & set(event["removed_positions"]), (
                        "A retired position still has a computed query replica", index, policy)
            physical = [e for e in events if e.get("physical_forwards", 1)]
            assert [e["q_tokens"] for e in physical] == [c["rows"] for c in facade.calls], policy
            generation = [(e, c) for e, c in zip(physical, facade.calls) if e["phase"] != "prefill"]
            sequence = raw[0, prompt.shape[1]:prompt.shape[1]+cap].tolist()
            if request_failure:
                # A failed request returned no completed response. Preserve
                # all attempted work in the charged counters.
                sequence = []
            stops = [i for i, token in enumerate(sequence) if token in eos]
            if stops:
                sequence = sequence[:stops[0]+1]
            n = sum(t not in special for t in sequence)
            q = sum(c["rows"] for e, c in generation)
            qp = sum(c["physical_rows"] for e, c in generation)
            result = {
                "status": "generation_failed" if request_failure else "success",
                "model": args.model, "policy": policy, "task_id": task["task_id"],
                "index": index, "dataset": task["dataset"], "seed": seed, "phase": phase,
                "generation_cap": cap, "prompt_ids": prompt[0].tolist(), "output_ids": sequence,
                "output_text": tokenizer.decode(sequence, skip_special_tokens=True),
                "extra": extra,
                "output_tokens": n, "generation_forwards": len(generation),
                "generation_query_rows": q, "generation_physical_padded_rows": qp,
                "prefill_rows": sum(c["rows"] for e, c in zip(physical, facade.calls) if e["phase"] == "prefill"),
                "max_q": max(c["rows"] for e, c in generation),
                "q_histogram": dict(collections.Counter(str(c["rows"]) for e, c in generation)),
                "tpf": n/len(generation), "token_efficiency": n/q,
                "physical_token_efficiency": n/qp, "eos": bool(stops),
                "truncated": not bool(stops) and request_failure is None,
                "remaining_masks": sequence.count(mask_id), "e2e_shared_gpu_wall_ms": wall_ms,
                "t2t_edits": sum(e.get("edited", 0) for e in events),
                "logical_iterations": len([e for e in events if e["phase"] != "prefill"]),
                "removed_q": sum(e.get("skipped_q", 0) for e in events),
                "dense_equivalent_q": sum(e.get("dense_q", e["q_tokens"]) for e in events if e["phase"] != "prefill"),
                "request_failure": request_failure,
            }
            # Opt-in raw ledgers for native32 only.  The default archived format
            # is unchanged; sparse policies never set capture_traces.
            if cohort_done or (getattr(args, "capture_traces", False)
                    and policy == "native32" and args.model == "llada21"):
                result["events"] = events
                result["calls"] = facade.calls

            # Stored only in the user's ignored output directory for offline scoring.
            path = args.output / f"{index:02d}-{policy}.json"
            path.write_text(json.dumps(result, ensure_ascii=False))
            return path

    cls = {"baselines": ForwardBatcher, "flowblock-prefix": AuditedBatcher,
           "optimus-masked-obs": PhaseBatcher, "llada20-frozen30": AuditedBatcher,
           "llada20-cached-b32": AuditedBatcher}[args.experiment]
    batcher = cls(model, args.model, args.batch_size or plan["batch_size"])
    arms = plan["policies"]
    groups = [arms] if plan["mix_policies"] else [[p] for p in arms]
    try:
        for group in groups:
            print("starting", group, flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            if cohort_done and (len(group) != 1 or len(tasks) != 90):
                raise ValueError("A qualification gate requires one policy and exactly 90 requests")
            for cohort in ((tasks[:30], tasks[30:]) if cohort_done else (tasks,)):
                jobs = [lambda t=t, p=p: invoke(batcher, t, p, 2048, "formal")
                        for p in group for t in cohort]
                if args.model == "llada20" and group in (["native32"], ["native32_cached"]):
                    for job in jobs:
                        job()
                else:
                    batcher.run(jobs)
                if cohort_done:
                    cohort_done(group[0],cohort)
            print("finished", group, flush=True)
    finally:
        if hasattr(batcher, "close"):
            batcher.close()
