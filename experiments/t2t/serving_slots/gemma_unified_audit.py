"""Observe true projections on the unified Gemma route; CPU qualification only."""
import argparse
from contextlib import ExitStack, contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path

import torch

from bdlm_serving.evaluation.frozen import ROOT, sha
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.evaluation.original30 import save_json
from bdlm_serving.runtime import diffusiongemma_wave as wave
from bdlm_serving.runtime.diffusiongemma import STUDY, observe
from experiments.t2t.serving_slots.expanded_handoff import SAMPLES


@contextmanager
def projectors(model):
    calls, positions = {}, []
    with ExitStack() as stack:
        def see_input(_, args, kw):
            pos = kw["decoder_position_ids"]
            if pos.ndim != 2 or pos.shape[0] != 1:
                raise ValueError("Expected one physical request")
            positions.append(pos[0].tolist())
        stack.callback(model.model.decoder.register_forward_pre_hook(see_input, with_kwargs=True).remove)
        for i, layer in enumerate(model.model.decoder.layers):
            for name in ("q_proj", "k_proj", "v_proj"):
                proj = getattr(layer.self_attn, name)
                if proj is None and name == "v_proj":
                    continue  # Pinned checkpoint can share the key/value projection.
                tape = calls[i, name] = []

                def see(_, args, tape=tape):
                    x, = args
                    if x.ndim != 3 or x.shape[0] != 1:
                        raise ValueError("Projection must receive selected batch-one hidden rows")
                    tape.append(x.shape[1])
                stack.callback(proj.register_forward_pre_hook(see).remove)
        yield calls, positions


def measure(model, data, plan, series, seed, causal=True):
    from experiments.t2t.serving_slots.gemma_unified import generate
    b = plan["block_size"]
    origin = data["input_ids"].shape[1]
    offers = []

    def supply(state):
        if state.tick != len(offers):
            raise ValueError("Budget clock is not once per decoder invocation")
        q = series[state.tick % len(series)]
        offers.append(q)
        return q

    with observe(model) as events, projectors(model) as (proj, positions):
        ids, trace = generate(model, data, plan, next_budget=supply, seed=seed, causal=causal)
    selected = [e["selected"] for e in trace]
    widths = [e["q"] for e in trace]
    if (not trace or offers != [e["offer"] for e in trace] or positions != selected
            or not proj or any(tape != widths for tape in proj.values())):
        raise ValueError("Offers/selected positions disagree with physical projections")
    expected = [("encoder", origin)]
    for e in trace:
        head = e["head"]
        base = origin + b * head
        selected, sampled = set(e["selected"]), set(e["sampled"])
        if (e["q"] != len(selected) or e["selected"] != sorted(selected)
                or not sampled <= selected or set(e["refresh_only"]) != selected - sampled):
            raise ValueError("Inconsistent physical or parked rows")
        if not all(base + b*i + u in selected for i,dirty in enumerate(e["dirty_before"]) for u in dirty):
            raise ValueError("Dropped pending cache row")
        if e["offer"] < 2*b and any(p >= base+b for p in sampled):
            raise ValueError("Parked tail sampled under a narrow offer")
        if any(p < base or p >= base+b for p in e["prefix_selected"]):
            raise ValueError("Optional prefix must stay within the current physical head")
        expected.append(("decoder", e["q"]))
        expected.extend(("encoder", b) for _ in e["encoded"])
    if expected != events:
        raise ValueError("Physical decoder, re-encode or prefill ledger differs")
    return ids, trace, dict(
        decoder_layers=len(model.model.decoder.layers), projection_modules=len(proj),
        all_projected_widths_match=True, prefill=origin, F=len(events)-1,
        Q=sum(q for _,q in events[1:]), reencodes=sum(tag=="encoder" for tag,_ in events[1:]),
        sampled_tail_steps=sum(any(p >= origin+(e["head"]+1)*b for p in e["sampled"]) for e in trace),
        parked_refresh_rows=sum(len(e["refresh_only"]) for e in trace),
        offered=offers, actual_decoder_q=widths)


def main():
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("samples", "checkpoint", "output"):
        p.add_argument("--"+k, type=Path, required=True)
    p.add_argument("--index", type=int, default=0, choices=range(90))
    p.add_argument("--threads", type=int, default=8, choices=range(1, 13))
    p.add_argument("--mask-pair", action="store_true",
                   help="Repeat the same input and offers with only the block-causal mask disabled")
    a = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or sha(a.samples) != SAMPLES or a.output.exists():
        raise ValueError("Only a new CPU diagnostic on the registered frozen90 input is supported")
    task = load_tasks(a.samples)[a.index]
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    native = json.loads((STUDY/"config.json").read_text())
    # This diagnostic deliberately forces eligibility and shortens the output cap.
    # Do not score it or mix it into the registered 90-item suite.
    plan = dict(wave.resolve_plan(ROOT/"archive/diffusiongemma-wave/config.json"),
                cap=512, flowblock_spawn_threshold=0.0)
    from bdlm_serving.runtime.diffusiongemma import check_snapshot
    check_snapshot(a.checkpoint)
    proc = AutoProcessor.from_pretrained(a.checkpoint, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        a.checkpoint, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation=native["attention"], experts_implementation=native["experts"],
        device_map={"":"cpu"}).eval()
    data = proc.apply_chat_template([{"role":"user","content":task["prompt"]}],
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    offers = (256,512,32,64,128,256,512,32)
    with torch.inference_mode():
        ids, trace, obs = measure(model, data, plan, offers, task["seed"])
    result = dict(status="physical_route_only", quality_evaluated=False, causal=True,
                  task_id=task["task_id"], seed=task["seed"], device="cpu",
                  cap=512, spawn_threshold=0, returned_ids=len(ids), trace=trace, physical_work=obs,
                  sources={name:sha(Path(__file__).with_name(name)) for name in
                           ("gemma_unified.py","gemma_wave_kv.py")})
    if a.mask_pair:
        with torch.inference_mode():
            off_ids, off_trace, off_obs = measure(model, data, plan, offers, task["seed"], causal=False)
        origin = data["input_ids"].shape[1]
        def first_opportunity(steps):
            for i,e in enumerate(steps):
                boundary = origin + (1+e["head"]) * plan["block_size"]
                if any(p < boundary for p in e["selected"]) and any(p >= boundary for p in e["visible"]):
                    return i
        first = first_opportunity(trace)
        if first is None or first != first_opportunity(off_trace) or trace[:first] != off_trace[:first]:
            raise ValueError("Missing matched causal intervention")
        for key in ("head","offer","visible","selected","sampled","ages_before","dirty_before"):
            if trace[first][key] != off_trace[first][key]:
                raise ValueError(f"Input geometry differed at first intervention: {key}")
        result.update(first_opportunity_tick=first, pretrigger_trace_equal=True,
                      output_ids_equal=ids==off_ids, opened_returned_ids=len(off_ids),
                      output_sha256={k:sha256(json.dumps(v).encode()).hexdigest()
                                     for k,v in (("on",ids),("off",off_ids))},
                      opened_physical_work=off_obs)
    save_json(a.output, result)
    print(json.dumps(obs), flush=True)


if __name__ == "__main__":
    main()
