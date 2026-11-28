"""A800 physical A0: Q64 paid-IBS six-case prefix-seat production pair.

Arms per request, alternating order: ``stock`` (certified paid IBS only),
``observer`` (no-change paid-IBS repeat, must equal stock bit for bit) and
``seat`` (the same paid IBS planner plus at most one warm-MASK -> omitted paid
prefix exchange per physical block).  The seat row is budget neutral and is
audited with the same all-layer Q/K/V projection hooks as the controls.  The
six fixed indices are a screening pair; no expansion to 90 is authorized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import ExitStack
from functools import lru_cache
from types import SimpleNamespace

from .gemma_ibs_certified_eos import B, Stop, require, generate as certified_generate
from .gemma_paid_agreement import Agreement, generate as agreement_generate
from .gemma_prefix_seat import PrefixSeat
from tools.run_gemma_ibs_certified_eos import reconcile

Q64 = 64
SIX = (7, 13, 25, 33, 55, 71)
ARMS = ("stock", "observer", "seat")
A0_DEVICE_INDEX = 0
A0_UUID = "GPU-d3c3fc9c-b15d-9776-9094-7d3b85b705c1"

_PLAN = "            n = len(local)\n"
_AFTER = "            if qualify and len(trace)==2:\n"
_UNMASK = """                updated = unmask.low_confidence_dynamic_sparse(raw, proposal,
                    torch.as_tensor(prior.copy(),device=device),sel,row_width=width)
"""
_SIG = "def _generate(bank, model, data, controls, unmask, mode, c, qualify, caps):"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def guard_a0(stage: str, gpu_uuid: str, device_index: int = A0_DEVICE_INDEX) -> None:
    if device_index != A0_DEVICE_INDEX:
        raise RuntimeError(f"Only A0 is allowed; got device index {device_index}")
    apps = subprocess.check_output(
        ["nvidia-smi", "-i", str(device_index),
         "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
        text=True).strip()
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", str(device_index), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True).strip()
    if uuid != gpu_uuid or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise RuntimeError(f"A0 UUID mismatch: nvidia={uuid!r} arg={gpu_uuid!r} "
                           f"visible={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    if stage == "before":
        mem, util = subprocess.check_output(
            ["nvidia-smi", "-i", str(device_index),
             "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True).strip().split(",")
        if mem.strip() != "0" or util.strip() != "0":
            raise RuntimeError(f"A0 is not idle: memory={mem.strip()} util={util.strip()}")
    expected = "" if stage == "before" else f"{os.getpid()}, {gpu_uuid}"
    if apps != expected:
        raise RuntimeError(f"Unexpected A0 state at {stage}: {apps!r}")


def observe(model, unmask, stack, torch, hold):
    """Physical tracer identical to tools.observe, gated by ``hold['suppress']``."""
    decoder, encoder, projections, events = [], [], {}, []

    def cuda(x):
        require(x.device.type == "cuda" and x.device.index == 0, "Non-A0 model input")
        return x

    def rng():
        return hashlib.sha256(torch.cuda.get_rng_state(0).numpy().tobytes()).hexdigest()

    def dec(_, args, kw):
        if hold['suppress']:
            return
        ids, pos = (cuda(kw[k]) for k in ("decoder_input_ids", "decoder_position_ids"))
        require(ids.ndim == 2 and ids.shape[0] == 1 and ids.shape == pos.shape, "Decoder shape")
        events.append(["D", ids.shape[1]])
        decoder.append(dict(ids=ids[0].tolist(), pos=pos[0].tolist(), rng=rng(), sample=None))

    def enc(_, args, kw):
        if hold['suppress']:
            return
        ids, pos = (cuda(kw[k]) for k in ("input_ids", "position_ids"))
        require(ids.ndim == 2 and ids.shape[0] == 1 and ids.shape == pos.shape, "Encoder shape")
        events.append(["E", ids.shape[1]])
        encoder.append([ids[0].tolist(), pos[0].tolist()])

    real_sampler = unmask.low_confidence_dynamic_sparse

    def sample(raw, proposal, prior, sel, **kw):
        if hold['suppress']:
            return real_sampler(raw, proposal, prior, sel, **kw)
        require(decoder and decoder[-1]["sample"] is None, "Multiple samplings in one decoder tick")
        updated = real_sampler(raw, proposal, prior, sel, **kw)
        decoder[-1]["sample"] = dict(rows=cuda(sel).tolist(), proposal=cuda(proposal).tolist(),
                                     take=cuda(updated[sel]).tolist())
        return updated

    stack.enter_context(model.model.decoder.register_forward_pre_hook(dec, with_kwargs=True))
    stack.enter_context(model.model.encoder.register_forward_pre_hook(enc, with_kwargs=True))
    from unittest.mock import patch
    stack.enter_context(patch.object(unmask, "low_confidence_dynamic_sparse", sample))
    for i, layer in enumerate(model.model.decoder.layers):
        for key in ("q_proj", "k_proj", "v_proj"):
            module = getattr(layer.self_attn, key)
            if key == "v_proj" and module is None:
                continue
            tape = projections[f"{i}.{key}"] = []

            def project(_, args, tape=tape):
                if hold['suppress']:
                    return
                x, = args
                require(cuda(x).ndim == 3 and x.shape[0] == 1, "Nonphysical batch or projection")
                tape.append(x.shape[1])

            stack.enter_context(module.register_forward_pre_hook(project))
    return decoder, encoder, projections, events, rng


@lru_cache(maxsize=1)
def _seat_loop(rt):
    """Instrument only the pinned runtime; planning changes stay before bank.select."""
    path = Path(rt.__file__)
    raw = path.read_bytes()
    require(sha(raw).hexdigest() ==
            "a09d04477361cb2a260dac85ce2195c0bfee9218306b7cfb96d7ddf4f1dba9af",
            "Unqualified IBS runtime source")
    src = raw.decode().split(_SIG)
    require(len(src) == 2, "Missing IBS loop")
    body = src[1].split("\n\ndef ", 1)[0]
    anchors = (_PLAN, _AFTER, _UNMASK)
    require(all(body.count(a) == 1 for a in anchors), "Ambiguous pinned-loop hook anchors")
    require("_ibs_seat" not in body, "seat hook name collides with stock locals")
    body = body.replace(_PLAN, "            local = _ibs_eos.rows(local, prior, dirty, start, offer)\n" + _PLAN)
    body = body.replace(_AFTER, """\
            delivery = _ibs_eos.after_update(ids, resolved, dirty, start, ctl.cached_prefix_len, eos, trace)
            if delivery is not None:
                return delivery, trace
""" + _AFTER)
    body = body.replace(_UNMASK, """\
                updated = _ibs_agreement.observe(unmask, raw, proposal, prior, sel, width, start, tick)
""")
    seat_hook = ("            local = _ibs_seat.rows(local=local, prior=prior, width=width, "
                 "dirty=dirty, start=start, offer=offer, ready=ctl.cached_prefix_len, "
                 "bank=bank, tick=tick)\n")
    require(body.count(_PLAN) == 1, "Lost EOS planning anchor")
    body = body.replace(_PLAN, seat_hook + _PLAN)
    namespace = {}
    source = ("\n" * src[0].count("\n") + _SIG[:-2] +
              ", _ibs_eos, _ibs_agreement, _ibs_seat):" + body)
    exec(compile(source, f"<prefix-seat {path}>", "exec"), vars(rt), namespace)
    return namespace["_generate"]


def prefix_seat_generate(rt, model, data, controls, unmask, stop, agreement, seat):
    loop = _seat_loop(rt)
    bank = rt.Resident()
    bank.bypass = False
    with rt.resident_attention(model, bank):
        return loop(bank, model, data, controls, unmask, "ibs", stop.offer, False,
                    (stop.offer,), stop, agreement, seat)


def run_arm(rt, torch, model, data, proc, controls, unmask, task, mode, scoring,
            special, seat_cls=PrefixSeat):
    hold = {'suppress': False}
    stop = Stop("paid", Q64)
    agreement = None if mode == "stock" else Agreement(
        apply=False, excluded_ids=special, floor=0.5)
    seat = seat_cls() if mode == "seat" else None
    prompt = data["input_ids"][0].tolist()
    eos = set(model.generation_config.eos_token_id)
    extras = set(proc.tokenizer.all_special_ids) | eos | {model.generation_config.pad_token_id}
    torch.manual_seed(task["seed"])
    with torch.inference_mode(), ExitStack() as stack:
        decoder, encoder, projections, events, rng = observe(model, unmask, stack, torch, hold)
        if agreement is not None:
            agreement.record_take = lambda take: decoder[-1]["sample"].update(take=take)
        if mode == "stock":
            ids, trace = certified_generate(rt, model, data, controls, unmask, stop)
        elif mode == "observer":
            ids, trace = agreement_generate(rt, model, data, controls, unmask, stop, agreement)
        else:
            ids, trace = prefix_seat_generate(rt, model, data, controls, unmask, stop,
                                              agreement, seat)
        after = rng()
    widths = [t["q"] for t in trace]
    require(len(projections) == 85 and all(v == widths for v in projections.values()),
            "Real Q/K/V widths")
    certs = {k: getattr(stop, k, None) for k in ("first_clean", "first_paid", "trigger", "payment")}
    bill, states = reconcile(trace, decoder, encoder, events, ids, prompt, eos,
                             certs["trigger"], certs["payment"], Q64)
    score = scoring.score(proc.tokenizer.decode(ids, skip_special_tokens=True), task)
    return dict(ids=ids, trace=trace, decoder=decoder, encoder=encoder, events=events,
                states=states, projections=sorted(projections), rng_after=after, bill=bill,
                **certs, correct=bool(score["correct"]), scoring=score["metric"],
                N=sum(i not in extras for i in ids), agreement=agreement, seat=seat)


def stable(a):
    return {k: v for k, v in a.items() if k not in ("agreement", "seat")}


def output_digest(ids):
    return hashlib.sha256(json.dumps(ids.tolist(), separators=(",", ":")).encode()).hexdigest()


def first_fork(reference, treatment):
    for tick, (old, new) in enumerate(zip(reference["trace"], treatment["trace"])):
        if old != new:
            return dict(kind="trace", tick=tick, reference=old, treatment=new)
    for offset, (x, y) in enumerate(zip(reference["ids"].tolist(),
                                        treatment["ids"].tolist(), strict=True)):
        if x != y:
            return dict(kind="output_id", offset=offset, reference=x, treatment=y)
    if len(reference["trace"]) != len(treatment["trace"]):
        return dict(kind="trace_length", reference=len(reference["trace"]),
                    treatment=len(treatment["trace"]))
    return None


def run_one(handle, task, position):
    data = handle["proc"].apply_chat_template(
        [{"role": "user", "content": task["prompt"]}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda:0")
    order = list(ARMS) if position % 2 == 0 else list(reversed(ARMS))
    out = {}
    for mode in order:
        out[mode] = run_arm(handle["rt"], handle["torch"], handle["model"], data,
                            handle["proc"], handle["controls"], handle["unmask"], task,
                            mode, handle["scoring"], handle["special"])
    require(stable(out["stock"]) == stable(out["observer"]),
            "Stock and observer-only paid-close diverged")
    seat = out["seat"]["seat"].record if out["seat"]["seat"] is not None else None
    if seat is None or not seat["exchanges"]:
        require(stable(out["observer"]) == stable(out["seat"]),
                "Seat arm with no planned exchange diverged")
    fork = first_fork(out["observer"], out["seat"])
    if seat and seat["exchanges"]:
        require(fork is not None and fork["kind"] == "trace" and
                fork["tick"] == seat["exchanges"][0]["tick"],
                "First trace fork is not the audited seat exchange")
    arms = {}
    for mode, v in out.items():
        record = v["seat"].record if v["seat"] is not None else None
        arms[mode] = dict(
            bill=v["bill"], trigger=v["trigger"], payment=v["payment"], correct=v["correct"],
            N=v["N"], scoring=v["scoring"], rng_after=v["rng_after"],
            output_sha256=output_digest(v["ids"]),
            agreement=None if v["agreement"] is None else dict(
                first_extra=v["agreement"]["first_extra"], applied=v["agreement"]["applied"],
                extra_events=[o for o in v["agreement"]["observations"] if o["extra"]]),
            seat=record)
    prompt_sha = hashlib.sha256(json.dumps(
        data["input_ids"][0].tolist(), separators=(",", ":")).encode()).hexdigest()
    return dict(index=task["index"], task_id=task["task_id"], dataset=task["dataset"],
                seed=task["seed"], order=order, prompt_sha256=prompt_sha,
                stock_observer_identical=True, first_fork=fork, arms=arms)


def load_runtime():
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from bdlm_serving.evaluation import frozen, frozen90, scoring
    from bdlm_serving.methods.optimus import upstream
    from bdlm_serving.runtime import diffusiongemma_optimus as rt, optimus_limits
    from bdlm_serving.runtime.diffusiongemma import check_snapshot
    from . import gemma_paid_agreement_eval as a3

    require(transformers.__version__ == "5.12.0", "Wrong transformers version")
    controls, strategy, hashes = upstream.load_controls()
    require(hashes == a3.OPTIMUS_FILES and upstream.COMMIT == a3.OPTIMUS, "Wrong Optimus source")
    require(sha(rt.__file__) == a3.BASELINE_SHA, "Wrong IBS runtime source")
    require(sha(scoring.__file__) ==
            "a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692"
            and sha(upstream.__file__) ==
            "f21a4704697cace53b19dfbbbf94bbabc877241b9c979e15b0d40e9a2ea0d223"
            and sha(optimus_limits.__file__) ==
            "842f078e06b5f79b336165b060d1212e947e01182e167379a1d1dbb9f62b5376",
            "Wrong scorer/upstream/limits source")
    require(sha(a3.SAMPLES) == "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed",
            "Wrong frozen samples")
    check_snapshot(a3.MODEL)
    require(sha(frozen.ROOT / "benchmarks/checkpoints/diffusiongemma.json") == a3.MODEL_MANIFEST,
            "Wrong model manifest")
    proc = AutoProcessor.from_pretrained(a3.MODEL, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        a3.MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
        experts_implementation="grouped_mm", device_map={"": "cuda:0"}).eval()
    require(model.config.canvas_length == B, "Wrong canvas length")
    unmask = controls.UnmaskingProcessor(SimpleNamespace(
        confidence_threshold=.9, block_length=B, denoising_steps=B,
        unmasking_strategy=strategy.LOW_CONFIDENCE_DYNAMIC))
    tasks = frozen90.load_tasks(a3.SAMPLES)
    eos = set(model.generation_config.eos_token_id)
    special = set(proc.tokenizer.all_special_ids) | eos | {model.generation_config.pad_token_id}
    return dict(torch=torch, model=model, proc=proc, controls=controls, unmask=unmask,
                tasks=tasks, scoring=scoring, rt=rt, special=special, a3=a3,
                transformers=str(transformers.__version__))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--deadline", type=int, required=True)
    ap.add_argument("--gpu-uuid", default=A0_UUID)
    a = ap.parse_args()
    if not str(a.root).startswith("/dev/shm/"):
        print("A0 prefix-seat root must be under /dev/shm", file=sys.stderr)
        return 2
    if a.gpu_uuid != A0_UUID:
        print("A0 prefix-seat refuses a non-A0 UUID", file=sys.stderr)
        return 2
    try:
        guard_a0("before", a.gpu_uuid)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"A0 PREFIX SEAT NOT STARTED: A0 guard failed: {exc}", file=sys.stderr)
        return 4
    a.root.mkdir(parents=True, exist_ok=True)
    handle = load_runtime()
    cases = SIX
    rows, gate_rows = [], []
    for pos, index in enumerate(cases):
        require(time.time() < a.deadline, "Deadline")
        guard_a0("request", a.gpu_uuid)
        task = next(t for t in handle["tasks"] if t["index"] == index)
        error = None
        try:
            row = run_one(handle, task, pos)
        except Exception as exc:  # keep each case independent
            import traceback
            traceback.print_exc()
            error = repr(exc)
            row = dict(index=index, task_id=task["task_id"], dataset=task["dataset"],
                       seed=task["seed"], order=list(ARMS),
                       stock_observer_identical=False, arms={}, error=error)
        row["error"] = error
        rows.append(row)
        gate_rows.append(error is None and row.get("stock_observer_identical") is True)
        (a.root / f"case-{index:02}.json").write_text(
            json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        print("CASE " + json.dumps(dict(
            index=index, error=error, order=row["order"],
            identical=row.get("stock_observer_identical"),
            first_fork=row.get("first_fork"), arms=row.get("arms", {})),
            separators=(",", ":"), allow_nan=False), flush=True)
    gate = "passed" if all(gate_rows) else "failed"
    summary = dict(schema="gemma-paid-ibs-q64-a0-prefix-seat-six-v1",
                   gpu=a.gpu_uuid, q=Q64, cases=list(cases), arms=list(ARMS),
                   gate=gate, failed=[rows[i]["index"] for i, ok in enumerate(gate_rows) if not ok],
                   per_case=[dict(index=r["index"], error=r.get("error"),
                                  first_fork=r.get("first_fork"),
                                  seat=(r.get("arms", {}).get("seat", {}) or {}).get("seat"))
                             for r in rows],
                   transformers=handle["transformers"])
    (a.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY " + json.dumps(dict(gate=gate, failed=summary["failed"]),
                                  separators=(",", ":")), flush=True)
    return 0 if gate == "passed" else 5


if __name__ == "__main__":
    raise SystemExit(main())
