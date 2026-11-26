"""Real-weight CPU mixed-offer mask diagnosis; deliberately not a quality run."""
import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path

import torch
from bdlm_serving.evaluation.original30 import save_json
from bdlm_serving.runtime.serial import install_llada_rectangular_mask
from experiments.t2t.serving_slots.llada_all_slots import AllSlotKernel
from experiments.t2t.serving_slots.llada_top_transition import TopTransitionKernel
from experiments.t2t.serving_slots.mask_probe_llada import verify_checkpoint
from experiments.t2t.serving_slots.mask_pair_llada import head_to_tail_mask

OFFERS = (32, 64, 4, 8, 16, 32, 64, 4)


@torch.inference_mode()
def measure(kernel, ids, config, length, seed, mask_id, eos, series, *, opened=False, price=False):
    digest = sha256()
    opportunity_input = None
    first = None
    observations = []
    quotes = []
    state = kernel.create(ids, config, length, seed, mask_id, eos)
    def change(_, args, kw):
        nonlocal first, opportunity_input
        positions = kw["position_ids"]
        native = kw["attention_mask"]
        mapping, edge = head_to_tail_mask(native, positions, ids.shape[1], state.block)
        # Compare input tensors on both arms up to and including the first opportunity.
        if first is None:
            for tensor in (positions, native, kw["inputs_embeds"]):
                digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
                digest.update(tensor.detach().contiguous().view(torch.uint8).numpy())
            if edge is not None:
                first = len(observations)
                opportunity_input = digest.hexdigest()
        observations.append(dict(q=positions.numel(), edges=edge["opened_edges"] if edge else 0))
        if opened and edge is not None:
            return args, dict(kw, attention_mask=mapping)
    with ExitStack() as stack:
        stack.callback(kernel.model.register_forward_pre_hook(change, with_kwargs=True).remove)
        def supply(info):
            return series[info.tick % len(series)]
        if price:
            from experiments.t2t.serving_slots.llada_quote import with_quotes
            def select(info, options):
                offer = supply(info)
                quotes.append(options[offer])
                return offer
            kernel.run(state, with_quotes(kernel, state, select))
        else:
            kernel.run(state, supply)
    if [r["q"] for r in observations] != [e["physical_q"] for e in state.events]:
        raise ValueError("Input hooks disagree with runtime")
    gen = [e for e in state.events if e["phase"] != "prefill"]
    if any(e["layer_physical_q"] != [e["physical_q"]]*len(kernel.projections) for e in gen):
        raise ValueError("Hidden padded projection")
    if price:
        for e, opt in zip(gen, quotes, strict=True):
            if (e["physical_q"], e["top_action"], e["refreshed_pending"]) != (
                    opt["physical_q"], opt["action"], opt["mandatory_positions"]):
                raise ValueError("Quoted Q, phase or mandatory rows differ from execution")
        if sum(opt["prefill_q"] for opt in quotes) != sum(e["physical_q"] for e in state.events if e["phase"] == "prefill"):
            raise ValueError("Quoted prompt prefill differs from execution")
    output = kernel.output(state)[0].tolist()
    return dict(first_physical_call=first, pretrigger_input_sha256=opportunity_input or digest.hexdigest(),
                charged_F=len(gen), physical_Q=sum(e["physical_q"] for e in gen),
                edges_presented=sum(r["edges"] for r in observations), output_ids=output,
                projected_layers=len(kernel.projections), **({"quoted_steps":len(quotes)} if price else {}),
                steps=[dict(
                    action=e["top_action"], offer=e["nominal_q"], q=e["physical_q"],
                    rows=e["absolute_positions"], parked=e["refresh_only_positions"], edited=e.get("edited", 0))
                    for e in gen])


@torch.inference_mode()
def handle_pair(model, cases, cfg, mask_id, eos):
    from .request_handle import open_llada
    kernel = AllSlotKernel(model)
    series = [OFFERS, (4, 64, 8, 16, 32)]
    args = [(torch.tensor([case["prompt_ids"]]), cfg, 38, case["seed"], mask_id, eos)
            for case in cases]
    expected = [measure(kernel, *params, pattern, price=True)
                for params,pattern in zip(args,series,strict=True)]
    handles = [open_llada(kernel, *params) for params in args]
    ticks = [0,0]
    init = [0,0]
    observed = []
    layers = [[] for _ in kernel.projections]
    with ExitStack() as stack:
        def observe(_, __, kw):
            observed.append(kw["position_ids"][0].tolist())
        stack.callback(model.register_forward_pre_hook(observe, with_kwargs=True).remove)
        for idx, projection in enumerate(kernel.projections):
            def count_rows(_, inp, j=idx):
                layers[j].append(inp[0].shape[-2])
            stack.callback(projection.register_forward_pre_hook(count_rows).remove)
        while any(h.status == "ready" for h in handles):
            for i,h in enumerate(handles):
                if h.status != "ready":
                    continue
                j = ticks[i]
                exp = expected[i]["steps"][j]
                before = len(observed)
                menu = h.menu()
                if set(menu) != {4,8,16,32,64} or before != len(observed):
                    raise ValueError("Invalid menu or model work during quote")
                s = h.advance(menu[series[i][j%len(series[i])]])
                calls = observed[before:]
                if (len(calls) != s.total_f or calls[-1] != exp["rows"]
                    or (s.main_q,s.offer,s.action) != (exp["q"],exp["offer"],exp["action"])
                    or s.prefill_q != sum(map(len,calls[:-1]))
                    or any(widths[before:] != list(map(len,calls)) for widths in layers)):
                    raise ValueError(f"Physical work changed by interleaving: {i}/{j}")
                init[i] += s.prefill_q
                ticks[i] += 1
    out = []
    for i,h in enumerate(handles):
        ref = expected[i]
        if (h.status != "done" or list(h.result()) != ref["output_ids"]
                or ticks[i] != ref["charged_F"] or init[i] != len(cases[i]["prompt_ids"])//32*32):
            raise ValueError("Interleaved output, forwards or prefill differs from solo")
        out.append(dict(task_id=cases[i]["task_id"], seed=cases[i]["seed"],
                        prompt_rows=len(cases[i]["prompt_ids"]), prefill_Q=init[i],
                        offers=series[i], paid_F=ticks[i], paid_Q=ref["physical_Q"],
                        output_tokens_including_special=len(h.result()),
                        output_ids_sha256=sha256(json.dumps(h.result()).encode()).hexdigest(),
                        positions_and_outputs_match_solo=True,
                        maximum_head_or_tail_refresh=max(len(s["parked"]) for s in ref["steps"])))
    return dict(device="cpu", cap=38, quality_evaluated=False, mode="serial_interleaved",
                model="llada21", physical_layers=len(layers), requests=out)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ("checkpoint", "original", "output"):
        parser.add_argument("--"+k, required=True, type=Path)
    parser.add_argument("--quote", action="store_true", help="Check all chosen offers against projections")
    parser.add_argument("--interleave-with", type=Path, help="Second native prompt for the request-handle check")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Fresh CPU-only diagnostic required")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    verify_checkpoint(args.checkpoint)
    opts = dict(trust_remote_code=True, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(args.checkpoint, **opts)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, **opts, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": "cpu"}).eval()
    install_llada_rectangular_mask(model, allow_selected_cache=True)
    row = json.loads(args.original.read_text())
    params = (torch.tensor([row["prompt_ids"]]), dict(confidence_threshold=.5, editing_threshold=.5,
              max_post_steps=16), 38, row["seed"], tok.convert_tokens_to_ids("<|mask|>"), [tok.eos_token_id])
    if args.interleave_with:
        peer = json.loads(args.interleave_with.read_text())
        if (row["index"], peer["index"]) != (0,10) or row["seed"] == peer["seed"]:
            raise ValueError("Expected the registered pair, original indices 0 and 10")
        result = handle_pair(model, [row, peer], params[1], params[4], params[5])
        save_json(args.output, result)
        print(json.dumps(result), flush=True)
        return
    controls = []
    for offer in (32, 64):
        old = measure(TopTransitionKernel(model), *params, (offer,))
        new = measure(AllSlotKernel(model), *params, (offer,))
        if new != old:
            raise ValueError(f"Fixed {offer} differs from existing top-tier route")
        controls.append(dict(offer=offer, F=new["charged_F"], Q=new["physical_Q"], exact=True))
    on, off = [measure(AllSlotKernel(model), *params, OFFERS, opened=opened, price=args.quote)
               for opened in (False, True)]
    if (on["first_physical_call"] is None or on["first_physical_call"] != off["first_physical_call"]
            or on["pretrigger_input_sha256"] != off["pretrigger_input_sha256"]
            or not any(e["offer"] < 32 and e["parked"] for e in on["steps"])
            or not set(OFFERS) <= {e["offer"] for e in on["steps"]}):
        raise ValueError("No matched first opportunity or dirty-tail downgrade")
    save_json(args.output, dict(status="route_only", quality_evaluated=False, device="cpu", cap=38,
        task_id=row["task_id"], seed=row["seed"], series=OFFERS, controls=controls,
        output_ids_equal=on["output_ids"] == off["output_ids"], on=on, off=off))
    for k, r in (("on", on), ("off", off)):
        print(k, r["charged_F"], r["physical_Q"], r["first_physical_call"], flush=True)


if __name__ == "__main__":
    main()
