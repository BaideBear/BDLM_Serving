"""Batch-one shard API; caller owns device selection, deadlines and persistence.

Q is the actual history-reading decoder Q, F includes post-prefill encodes.
Only exhaustion of the 48-step sampler before head coverage is a valid
algorithm failure: no complete answer is delivered (N=0, incorrect), but all
paid decoder/encoder work is retained. Infra, scoring and invariant errors
propagate and must not be counted as scored requests. Use serially per process.
"""
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

from bdlm_serving.evaluation import scoring
from bdlm_serving.methods.t2t.unified_rows import Params
from . import unified_gemma as adapter


@dataclass(frozen=True)
class Handle:
    model: object
    processor: object
    device: object
    special_ids: frozenset


def load(checkpoint: Path) -> Handle:
    """Load the pinned local checkpoint once onto the caller's current CUDA device."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("shared Gemma evaluation requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion, __version__
    from bdlm_serving.runtime.diffusiongemma import check_snapshot
    if __version__ != "5.12.0" or version("math-verify") != "0.8.0":
        raise RuntimeError("Expected transformers 5.12.0 and frozen math-verify 0.8.0 scorer")
    check_snapshot(checkpoint)
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
        experts_implementation="grouped_mm", device_map={"": str(device)}).eval()
    if (model.config.canvas_length != adapter.B or model.device != device or
            any(p.device != device for p in model.parameters()) or
            any(b.device != device for b in model.buffers())):
        raise RuntimeError("Expected pinned B256 model wholly on the selected CUDA device")
    cfg = model.generation_config
    eos = (cfg.eos_token_id,) if isinstance(cfg.eos_token_id, int) else cfg.eos_token_id or ()
    special = frozenset(processor.tokenizer.all_special_ids).union(eos, (cfg.pad_token_id,))
    return Handle(model, processor, device, special)


def _account(roles, event, offer):
    positions = event["physical_decoder_positions"]
    width = len(positions)
    if not (positions == event["positions"] and 0 < width <= offer and
            width == event["decoder_Q"] and event["decoder_F"] == 1 and
            event["projection_streams"] == 85 and event["offer"] == offer):
        raise RuntimeError("Invalid measured decoder receipt")
    plan = event["shared_plan"]
    route = event["route"]
    if route == "ordinary":
        if list(plan["rows"]) != positions:
            raise RuntimeError("Planner/physical positions differ")
        for name in ("dirty", "mask", "prefix", "cold"):
            roles[f"{name}_Q"] += len(plan[name])
        route = "drain" if plan["drain_only"] else "sample"
    elif route not in ("rebuild", "verify") or plan is not None:
        raise RuntimeError("Unrecognized mandatory projection")
    roles[route+"_Q"] += width
    roles[route+"_F"] += 1
    roles["peak_Q"] = max(roles["peak_Q"], width)


def _coverage_expired(req, err, previous_attempt):
    rows, sampler = req.rows, req.sampler
    return (type(err) is RuntimeError and str(err) == "Gemma clock closed before complete sampler coverage"
            and req.status == "failed" and req.last_attempt is previous_attempt and req.cursor == 0
            and rows.status == "incomplete" and rows.age == sampler.age == rows.max_steps == 48
            and not rows.dirty and (req.tail is None or not req.tail.rows.dirty)
            and frozenset(sampler.sc) == rows.initialized and len(rows.initialized) < adapter.B)


def evaluate(handle: Handle, task, offer: int, params: Params) -> dict:
    """Execute and bill one fixed-offer frozen task, never retry or change an offer."""
    if type(offer) is not int or offer not in adapter.OFFERS or not isinstance(params, Params):
        raise ValueError("Expected a registered fixed Gemma offer and shared Params")
    if handle.device.type != "cuda" or handle.model.device != handle.device or handle.model.training:
        raise RuntimeError("Cannot evaluate on CPU, a moved model or a training model")
    if task["dataset"] not in ("gsm8k", "math500", "humaneval"):
        raise ValueError("Unknown frozen scorer")
    prompt = task["prompt"]
    data = handle.processor.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to(handle.device)
    meter = adapter.RequestMeter()
    req = adapter.create_request(handle.model, data, q=offer, seed=task["seed"],
                                 max_new_tokens=2048, params=params, meter=meter)
    roles = Counter()
    reason = None
    while req.status == "ready":
        before = req.last_attempt
        try:
            event = req.step(offer)
        except RuntimeError as err:
            if not _coverage_expired(req, err, before):
                raise
            reason = "gemma_clock_exhausted_before_full_head_coverage"
            break
        _account(roles, event, offer)
    failed = reason is not None
    if ((meter.started, meter.completed, meter.failed, meter.in_flight) !=
            (1, int(not failed), int(failed), 0)):
        raise RuntimeError("Wrong request denominator")
    receipts = req.encoder_receipts
    if (not receipts or receipts[0] != dict(prefill=True, F=1, Q=req.origin) or
            any(e != dict(prefill=False, F=1, Q=adapter.B) for e in receipts[1:])):
        raise RuntimeError("Incorrect encoder receipts")
    enc_f = len(receipts)-1
    f, q = req.decoder_F+enc_f, req.decoder_Q
    routes = ("sample", "drain", "rebuild", "verify")
    if (sum(roles[r+"_F"] for r in routes) != req.decoder_F or
            sum(roles[r+"_Q"] for r in routes) != q or
            sum(roles[k+"_Q"] for k in ("dirty", "mask", "prefix")) != roles["sample_Q"]+roles["drain_Q"]):
        raise RuntimeError("Unbalanced decoder work")
    metric = output_sha = None
    if failed:
        n, correct = 0, False
    else:
        bill = req.completed_bill(handle.special_ids)
        if (bill["F"], bill["decoder_Q"], bill["encoder_Q"]) != (f, q, adapter.B*enc_f):
            raise RuntimeError("Complete request bill mismatch")
        n = bill["N"]
        text = handle.processor.tokenizer.decode(req.tokens, skip_special_tokens=True)
        score = scoring.score(text, task)
        correct, metric = score["correct"], score["metric"]
        if type(correct) is not bool:
            raise RuntimeError("Scorer did not produce a Boolean")
        output_sha = sha256(json.dumps(req.tokens, separators=(",", ":")).encode()).hexdigest()
    return dict(status="valid_algorithm_failure" if failed else "complete", failure_reason=reason,
                requests=1, model="gemma", offer=offer, index=task["index"], seed=task["seed"],
                params={"prefix_share": str(params.prefix_share), "open_at": str(params.open_at)},
                dataset=task["dataset"], task_id=task["task_id"],
                prompt_sha256=sha256(prompt.encode()).hexdigest(), output_ids_sha256=output_sha,
                correct=correct, metric=metric, N=n, F=f, Q=q, decoder_F=req.decoder_F, decoder_Q=q,
                encoder_F=enc_f, encoder_Q=adapter.B*enc_f, prefill_Q=req.origin,
                TPF=n/f, TE=n/q, max_decoder_Q=roles["peak_Q"], roles=dict(roles))
