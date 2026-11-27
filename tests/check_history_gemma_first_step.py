"""Real-checkpoint first-step attribution. This script cannot approve a gate."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import platform

import torch

from bdlm_serving.evaluation.frozen import sha
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.methods.t2t.history_gemma import Rows, _numerics, _replace, history_gemma
from bdlm_serving.methods.t2t.seat_audit import digest
from bdlm_serving.runtime.diffusiongemma import STUDY, check_snapshot
from tools.run_history_gemma import per_example_rng, provenance, read, write


class Captured(Exception):
    pass


class Replayed(Exception):
    pass


def snapshot(model, inputs):
    snap = {}

    def capture(**kw):
        snap.update(kw=kw, cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(model.device))
        raise Captured

    try:
        with ExitStack() as stack:
            _replace(stack, model, "_denoising_step", capture)
            model.generate(**inputs, max_new_tokens=model.config.canvas_length)
    except Captured:
        return snap
    raise RuntimeError("Native generate did not reach the first denoising step")


def fingerprint(kw, device):
    cache = kw["past_key_values"]
    return dict(
        encoder_kv=digest([(x.keys, x.values, getattr(x, "cumulative_length", None))
                           for x in cache.layers]),
        inputs=digest(kw["input_ids"], kw["current_canvas"], kw["argmax_canvas"],
                      kw["decoder_position_ids"], kw["self_conditioning_logits"],
                      kw["finished_denoising"], kw["cur_step"], sorted(kw["mask_mapping"])),
        self_conditioning_none=kw["self_conditioning_logits"] is None,
        cpu_rng=digest(torch.get_rng_state()),
        cuda_rng=digest(torch.cuda.get_rng_state(device)),
    )


def attention_truth(q, k, v, scale):
    """Independent dense FP64 oracle on the exact projected BF16 tensors."""
    group = q.shape[1] // k.shape[1]
    k = k.double().repeat_interleave(group, dim=1)
    v = v.double().repeat_interleave(group, dim=1)
    return (((q.double() @ k.transpose(-1, -2)) * scale).softmax(-1) @ v).transpose(1, 2)


def compare(a, b):
    return dict(_numerics(a, b), equal=torch.equal(a, b))


def native_two_step_qkv(model, inputs):
    from copy import copy
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import ALL_ATTENTION_FUNCTIONS as impl

    layers = model.model.decoder.layers
    first_global = next(i for i, x in enumerate(layers) if not x.self_attn.is_sliding)
    targets = (0, first_global, len(layers) - 1)
    records = []
    current = None
    alias, native = "bdlm_gemma_capture_native_qkv", impl["sdpa"]
    step = model._denoising_step

    def attention(module, q, k, v, mask, **opts):
        i = module.layer_idx
        if current is None or i in current["layers"] or mask is not None or opts["is_causal"]:
            raise RuntimeError("Unexpected native attention scope")
        h = k.shape[-2] - q.shape[-2]
        layer = current["cache"].layers[i]
        if (h <= 0 or not torch.equal(layer.keys, k[:, :, :h])
                or not torch.equal(layer.values, v[:, :, :h])):
            raise RuntimeError("Native K/V does not start with the exact encoder cache")
        current["layers"][i] = (tuple(x.contiguous().clone() for x in (q, k, v)),
                                h, opts["scaling"])
        return native(module, q, k, v, mask, **opts)

    def capture(**kw):
        nonlocal current
        current = dict(cache=kw["past_key_values"], fingerprint=fingerprint(kw, model.device), layers={})
        records.append(current)
        if len(records) == 3:
            records.pop()  # Stop before a third decoder forward.
            raise Captured
        return step(**kw)

    try:
        with ExitStack() as stack:
            if alias in impl:
                raise RuntimeError("Attention capture already active")
            impl[alias] = attention
            stack.callback(impl.__delitem__, alias)
            for i in targets:
                module = layers[i].self_attn
                cfg = copy(module.config)
                cfg._attn_implementation = alias
                _replace(stack, module, "config", cfg)
            _replace(stack, model, "_denoising_step", capture)
            model.generate(**inputs, max_new_tokens=model.config.canvas_length)
    except Captured:
        pass
    if (len(records) != 2 or any(set(r["layers"]) != set(targets) for r in records)
            or records[0]["cache"] is not records[1]["cache"]
            or records[0]["fingerprint"]["encoder_kv"] != records[1]["fingerprint"]["encoder_kv"]
            or not records[0]["fingerprint"]["self_conditioning_none"]
            or records[1]["fingerprint"]["self_conditioning_none"]):
        raise RuntimeError("Expected two native steps on one encoder version, the second self-conditioned")
    return records


def fp64_partition(q, k, v, scale):
    group = q.shape[1] // k.shape[1]
    keys = k.double().repeat_interleave(group, 1)
    scores = (q.double() @ keys.transpose(-2, -1)) * scale
    values = v.double().repeat_interleave(group, 1)
    return scores.logsumexp(-1), scores.softmax(-1) @ values


def fp64_merge(local, history):
    block_lse, block_out = local
    hist_lse, hist_out = history
    lse = torch.logaddexp(block_lse, hist_lse)
    return ((block_lse - lse).exp().unsqueeze(-1) * block_out
            + (hist_lse - lse).exp().unsqueeze(-1) * hist_out)


def packed_control(model, processor, task):
    from bdlm_serving.methods.t2t.history_attention import SlotPlan, block_history, state_like

    data = processor.apply_chat_template([{"role": "user", "content": task["prompt"]}],
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    with per_example_rng(task["seed"], model.device):
        records = native_two_step_qkv(model, data)

    width = model.config.canvas_length
    report, empty = {}, {}
    # These guardrails qualify the packed-vs-FP64 diagnostic only.
    rel_limit, lse_limit = .004, 1e-4
    for i in records[0]["layers"]:
        frames = [r["layers"][i] for r in records]
        h, scale = frames[0][1:]
        if frames[1][1:] != (h, scale):
            raise RuntimeError("Native history geometry changed within the canvas")
        local, historical = zip(*[(fp64_partition(q, k[:, :, h:], v[:, :, h:], scale),
                                   fp64_partition(q, k[:, :, :h], v[:, :, :h], scale))
                                  for (q, k, v), _, _ in frames])
        q, k, v = frames[0][0]
        empty_state = state_like(q)
        local_only = block_history(q, k[:, :, h:].contiguous(), v[:, :, h:].contiguous(),
                                   k[:, :, :0].contiguous(), v[:, :, :0].contiguous(), None, None,
                                   SlotPlan.make([()], width, q.device), empty_state, scale)
        empty[i] = compare(local[0][1], local_only)
        if (empty[i]["rel_rmse"] > rel_limit or not torch.isneginf(empty_state[0]).all()
                or any(t.count_nonzero() for t in empty_state[1:])):
            raise RuntimeError(f"Empty-history FP64 check failed at layer={i}")
        for budget in (32, 64, 128):
            rows = Rows(width)
            state = state_like(frames[0][0][0])
            hist = (torch.full_like(historical[0][0], -torch.inf), torch.zeros_like(historical[0][1]))
            steps = []
            for j, ((q, k, v), _, _) in enumerate(frames):
                selected = rows.select(budget, [False] * width, True)
                unseen = set(range(width)) - rows.valid - set(selected)
                reused = rows.valid - set(selected)
                reread = rows.valid & set(selected)
                keep = sorted(set(range(width)) - set(selected))
                old = tuple(t[:, :, keep].clone() for t in state)
                got = block_history(q, k[:, :, h:].contiguous(), v[:, :, h:].contiguous(),
                                    k[:, :, :h].contiguous(), v[:, :, :h].contiguous(),
                                    None, None, SlotPlan.make([selected], width, q.device), state, scale)
                if not all(torch.equal(t[:, :, keep], a) for t, a in zip(state, old, strict=True)):
                    raise RuntimeError("An unselected historical row changed")
                if unseen:
                    cold = sorted(unseen)
                    if (not torch.isneginf(state[0][:, :, cold]).all()
                            or any(t[:, :, cold].count_nonzero() for t in state[1:])):
                        raise RuntimeError("A cold history row is not the empty partition")
                idx = list(selected)
                for cache, fresh in zip(hist, historical[j], strict=True):
                    cache[:, :, idx] = fresh[:, :, idx]
                want = fp64_merge(local[j], hist)
                truth_lse = historical[j][0][:, :, idx]
                got_lse = (state[0][:, :, idx] + state[1][:, :, idx].log()).double()
                lse_error = (truth_lse - got_lse).abs().amax().item()
                groups = {"cold": unseen, "reused": reused, "reread": reread,
                          "new": set(selected) - reread}
                errors = {name: {**compare(want[:, :, sorted(ids)], got[:, :, sorted(ids)]),
                                 "rows": len(ids)}
                          for name, ids in groups.items() if ids}
                if (len(selected) != budget or lse_error > lse_limit
                        or any(x["rel_rmse"] > rel_limit for x in errors.values())):
                    raise RuntimeError(f"Packed FP64 check failed at layer={i}, C={budget}, step={j}")
                note = dict(step=j, decoder_q=width, history_q=len(selected), lse_max_abs=lse_error,
                            unselected_cache_bitwise_equal=True, errors=errors)
                if reused:
                    ix = sorted(reused)
                    all_fresh = fp64_merge(local[j], historical[j])
                    note["reused_vs_wrong_fresh"] = compare(all_fresh[:, :, ix], got[:, :, ix])
                    if note["reused_vs_wrong_fresh"]["rel_rmse"] <= errors["reused"]["rel_rmse"]:
                        raise RuntimeError("Reuse oracle does not distinguish cached history from fresh history")
                steps.append(note)
                rows.refresh(selected, j)
                if j == 0:
                    rows.write(selected[:budget // 2], j)
            if not {"cold", "reused", "reread", "new"} <= steps[1]["errors"].keys():
                raise RuntimeError("Packed diagnostic missed one of the four row classes")
            report[f"layer{i}-C{budget}"] = dict(head_dim=q.shape[-1], encoder_rows=h, rounds=steps)
            worst = max(x["rel_rmse"] for s in steps for x in s["errors"].values())
            print(f"layer={i} C={budget} rel_rmse_max={worst:.6g} "
                  f"wrong_fresh_rel_rmse={steps[1]['reused_vs_wrong_fresh']['rel_rmse']:.6g}", flush=True)
    return dict(input=[r["fingerprint"] for r in records], cases=report, empty_history=empty,
                relative_rmse_limit=rel_limit, lse_abs_limit=lse_limit)


def replay(model, snap, arm, reference=None, *, single=None, freeze=None):
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import ALL_ATTENTION_FUNCTIONS as impl

    alias = "bdlm_gemma_block_history"
    sdpa = impl["sdpa"]
    layers = model.model.decoder.layers
    kw = snap["kw"].copy()
    for key in ("sampler", "diffusion_stopping_criteria", "logits_processor", "finished_denoising"):
        kw[key] = deepcopy(kw[key])
    torch.set_rng_state(snap["cpu_rng"])
    torch.cuda.set_rng_state(snap["cuda_rng"], model.device)
    start = fingerprint(kw, model.device)
    capture, stats, interposed = {}, {}, {}
    trace = dict(input=start, attention=interposed)
    forward = kw["decoder_forward"]

    def measured(**args):
        out = forward(**args)
        capture["logits"] = out.logits
        return out

    kw["decoder_forward"] = measured
    if arm == "native":
        expected_uses = 0
    else:
        expected_uses = len(layers)

    def stash(layer_id, name, tensor):
        if reference is None:
            capture.setdefault(layer_id, {})[name] = tensor.detach().clone()
        else:
            stats.setdefault(layer_id, {})[name] = compare(reference[layer_id][name], tensor)

    def router_hook(i):
        def hook(module, args, out):
            probs, weights, indices = out
            stash(i, "router_input", args[0])
            if reference is None:
                kth = torch.topk(probs, indices.shape[-1] + 1, dim=-1).values
                capture[i].update(indices=indices.clone(), weights=weights.clone(),
                                  margin=(kth[..., -2] - kth[..., -1]).clone())
                return None
            base = reference[i]
            changed = (base["indices"].sort().values != indices.sort().values).any(-1)
            margin = base["margin"]
            stats[i].update(router_set_flips=int(changed.sum()), rows=indices.shape[0],
                            router_order_flips=int((base["indices"] != indices).any(-1).sum()),
                            changed_margin_p50=float(margin[changed].median()) if changed.any() else None,
                            stable_margin_p50=float(margin[~changed].median()) if (~changed).any() else None)
            if freeze == "choices":
                keep = base["indices"]
                p = probs.gather(-1, keep)
                p = p / p.sum(dim=-1, keepdim=True) * module.per_expert_scale[keep]
                return probs, p, keep
            if freeze == "choices_weights":
                return probs, base["weights"], base["indices"]
            return None
        return hook

    def observe_layer(i, name, which):
        def hook(_m, _a, out):
            stash(i, name, out[0] if which == "tuple" else out)
        return hook

    def observed_attention(production):
        def hook(module, q, k, v, mask, **opts):
            i = module.layer_idx
            if i in interposed:
                raise RuntimeError("Repeated decoder attention")
            split = arm == "split" and (single is None or i == single)
            oracle = arm == "oracle" and (single is None or i == single)
            cache = kw["past_key_values"].layers[i]
            h = cache.keys.shape[-2]
            if (mask is not None or opts["is_causal"] or opts.get("dropout") or h == 0 or
                    not torch.equal(k[:, :, :h], cache.keys) or
                    not torch.equal(v[:, :, :h], cache.values) or k.shape[2] != h + q.shape[2]):
                raise RuntimeError("Projected KV does not match native encoder/canvas geometry")
            candidate, extra = production(module, q, k, v, mask, **opts)
            note = interposed[i] = dict(head_dim=q.shape[-1], h=h, b=q.shape[2],
                                        split_output=split, oracle_output=oracle)
            if oracle:
                truth = attention_truth(q, k, v, opts["scaling"])
                rounded = truth.to(candidate.dtype).contiguous()
                note.update(sdpa_vs_oracle=compare(candidate, rounded),
                            sdpa_diff_elements=int((candidate != rounded).sum()),
                            elements=rounded.numel(),
                            sdpa_vs_fp64=compare(truth, candidate),
                            rounded_vs_fp64=compare(truth, rounded))
                return rounded, extra
            if arm == "oracle":
                return candidate, extra
            if arm == "passthrough":
                return candidate, extra
            # In a one-layer intervention only the designated layer's result
            # is returned; other calls run the production split for audit only.
            if single is not None and not split:
                return sdpa(module, q, k, v, mask, **opts)
            native, _ = sdpa(module, q, k, v, mask, **opts)
            note["split_vs_sdpa"] = compare(native, candidate)
            if freeze is None and (single is not None or i in (0, len(layers) // 2, len(layers) - 1)):
                truth = attention_truth(q, k, v, opts["scaling"])
                note.update(native_vs_fp64=compare(truth, native), split_vs_fp64=compare(truth, candidate))
            return candidate, extra
        return hook

    with ExitStack() as stack:
        for i, layer in enumerate(layers):
            stack.callback(layer.router.register_forward_hook(router_hook(i)).remove)
            for name, module, kind in (
                ("attention_projected", layer.self_attn, "tuple"),
                ("dense", layer.post_feedforward_layernorm_1, "tensor"),
                ("moe", layer.post_feedforward_layernorm_2, "tensor"),
                ("layer_output", layer, "tensor"),
            ):
                stack.callback(module.register_forward_hook(observe_layer(i, name, kind)).remove)

        def run_step():
            current, draft, sc, finished = model._denoising_step(**kw)
            trace["sampler"] = dict(canvas=digest(current), draft=digest(draft),
                                    accepted=digest(kw["sampler"].accepted_token_mask),
                                    self_conditioning=digest(sc), finished=digest(finished))

        if arm == "native":
            run_step()
        else:
            try:
                with history_gemma(model, model.config.canvas_length if arm == "split" else None) as ledger:
                    production = impl[alias]
                    impl[alias] = observed_attention(production)
                    run_step()
                    if len(ledger.trace) != 1 or ledger.events != [("decoder", model.config.canvas_length)]:
                        raise RuntimeError("Diagnostic replay did not run one native decoder step")
                    raise Replayed  # Exit without committing or simulating a terminal clock.
            except Replayed:
                pass
    if len(interposed) != expected_uses:
        raise RuntimeError("Attention instrumentation missed layers")
    after = fingerprint(kw, model.device)
    if after["encoder_kv"] != start["encoder_kv"]:
        raise RuntimeError("Decoder mutated the captured encoder cache")
    trace["post_rng"] = {k: after[k] for k in ("cpu_rng", "cuda_rng")}
    trace["logits"] = digest(capture["logits"])
    if reference is not None:
        trace["logit_error"] = compare(reference["logits"], capture["logits"])
        trace["argmax_changed_rows"] = int((reference["logits"].argmax(-1) != capture["logits"].argmax(-1)).sum())
        trace["layers"] = stats
    return trace, capture


def diagnose(model, processor, task, oracle_control=False):
    data = processor.apply_chat_template([{"role": "user", "content": task["prompt"]}],
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    with per_example_rng(task["seed"], model.device):
        snap = snapshot(model, data)
        base, reference = replay(model, snap, "native")
        report = {"native": base}
        decoder = model.model.decoder.layers
        global_layer = next(i for i, x in enumerate(decoder) if not x.self_attn.is_sliding)
        arms = [
            ("native_replay", "native", None, None),
            ("sdpa_passthrough", "passthrough", None, None),
            ("split_first_layer", "split", 0, None),
            ("split_first_global", "split", global_layer, None),
            ("split_last_layer", "split", len(decoder) - 1, None),
            ("split_all", "split", None, None),
            ("split_all_freeze_choices", "split", None, "choices"),
            ("split_all_freeze_choices_weights", "split", None, "choices_weights"),
        ]
        if oracle_control:
            arms = [arms[i] for i in (0, 1, 5)] + [
                ("fp64_first_layer", "oracle", 0, None),
                ("fp64_all", "oracle", None, None),
                ("fp64_all_freeze_choices", "oracle", None, "choices"),
            ]
        for name, arm, single, freeze in arms:
            trace, _ = replay(model, snap, arm, reference, single=single, freeze=freeze)
            report[name] = trace
            if trace["input"] != base["input"] or trace["post_rng"] != base["post_rng"]:
                raise RuntimeError(f"Replay input or RNG diverged: {name}")
            if name in ("native_replay", "sdpa_passthrough") and trace["sampler"] != base["sampler"]:
                raise RuntimeError(f"Unexpected native control sampler drift: {name}")
            errors = trace["logit_error"]
            flips = sum(x["router_set_flips"] for x in trace["layers"].values())
            print(f"{name}: logit_rel_rmse={errors['rel_rmse']:.6g} "
                  f"argmax_rows={trace['argmax_changed_rows']} expert_set_flips={flips}", flush=True)
            del _
    return report


def main():
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion, __version__

    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "samples", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--oracle-control", action="store_true")
    group.add_argument("--packed-control", action="store_true")
    args = parser.parse_args()
    plan = read(STUDY / "config.json")
    if not 0 <= args.index < 30 or __version__ != plan["transformers"] or torch.cuda.device_count() != 1:
        raise ValueError("Requires pinned transformers, one visible CUDA device and an original30 case")
    args.output.mkdir(parents=True, exist_ok=False)
    print("Checking pinned checkpoint bytes", flush=True)
    check_snapshot(args.checkpoint)
    tasks = load_tasks(args.samples)
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(args.checkpoint, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation=plan["attention"], experts_implementation=plan["experts"],
        device_map={"": "cuda:0"}).eval()
    with torch.inference_mode():
        report = (packed_control(model, processor, tasks[args.index]) if args.packed_control else
                  diagnose(model, processor, tasks[args.index], args.oracle_control))
    filename = "packed-control.json" if args.packed_control else "first-step.json"
    write(args.output / filename, dict(index=args.index, host=platform.node(),
          gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), transformers=__version__,
          source_sha256=provenance(), diagnostic_sha256=sha(Path(__file__)),
          **({"packed": report} if args.packed_control else {"arms": report}),
          gate_changed=False, diagnostic_only=True))


if __name__ == "__main__":
    main()
