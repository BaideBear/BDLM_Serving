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


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ("checkpoint", "original", "output"):
        parser.add_argument("--"+k, required=True, type=Path)
    parser.add_argument("--quote", action="store_true", help="Check all chosen offers against projections")
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
