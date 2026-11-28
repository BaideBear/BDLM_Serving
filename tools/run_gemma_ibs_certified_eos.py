"""Fixed-Q frozen90: stream sealed raw case receipts, with a six-case launch gate."""
import argparse
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

from experiments.t2t.serving_slots.gemma_ibs_certified_eos import (
    BASELINE_SHA, B, CAP, OFFERS, Stop, generate, require,
)

SIX = (0, 10, 20, 30, 40, 50)
ARMS = ("stock", "control", "strict", "paid")
GPU = "GPU-d3c3fc9c-b15d-9776-9094-7d3b85b705c1"
MODEL = Path("/gs/models/diffusiongemma-cd40195")
SAMPLES = Path("/gs/code/samples.json")
MODEL_MANIFEST = "ceb951734eb354728d7ba984093739119addbeda981b554af46d5dc105071f53"
OPTIMUS = "7343a37b2a69337caa452f17b59727dc6665f039"
OPTIMUS_FILES = {
    "sequence.py": "99c80047d517113579e04fd57443471a759a738dc30b206ba1342e0d4f9b3da2",
    "unmasking.py": "bbc3c3328e5497aa08aed6e04150ab0383f95dc06b4ff570a69adaed4a769746",
}


def emit(kind, **fields):
    print("RECEIPT " + json.dumps(dict(kind=kind, **fields), separators=(",", ":"), allow_nan=False), flush=True)


def sha(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def gpu_query(flag):
    return subprocess.check_output(
        ["nvidia-smi", "-i", "0", flag, "--format=csv,noheader,nounits"], text=True).strip()


def guard(stage):
    apps = gpu_query("--query-compute-apps=pid,gpu_uuid")
    require(gpu_query("--query-gpu=uuid") == os.environ["CUDA_VISIBLE_DEVICES"] == GPU
            and (not apps if stage == "before" else apps == f"{os.getpid()}, {GPU}"),
            f"Unexpected A0 state at {stage}: {apps}")


def observe(model, unmask, stack, torch):
    decoder, encoder, projections, events = [], [], {}, []

    def cuda(x):
        require(x.device.type == "cuda" and x.device.index == 0, "Non-A0 model input")
        return x

    def rng():
        return sha256(torch.cuda.get_rng_state(0).numpy().tobytes()).hexdigest()

    def dec(_, args, kw):
        ids, pos = (cuda(kw[k]) for k in ("decoder_input_ids", "decoder_position_ids"))
        require(ids.ndim == 2 and ids.shape[0] == 1 and ids.shape == pos.shape, "Decoder shape")
        events.append(["D", ids.shape[1]])
        decoder.append(dict(ids=ids[0].tolist(), pos=pos[0].tolist(), rng=rng(), sample=None))

    def enc(_, args, kw):
        ids, pos = (cuda(kw[k]) for k in ("input_ids", "position_ids"))
        require(ids.ndim == 2 and ids.shape[0] == 1 and ids.shape == pos.shape, "Encoder shape")
        events.append(["E", ids.shape[1]])
        encoder.append([ids[0].tolist(), pos[0].tolist()])

    real_sampler = unmask.low_confidence_dynamic_sparse

    def sample(raw, proposal, prior, sel, **kw):
        require(decoder[-1]["sample"] is None, "Multiple samplings in one decoder tick")
        updated = real_sampler(raw, proposal, prior, sel, **kw)
        decoder[-1]["sample"] = dict(rows=cuda(sel).tolist(), proposal=cuda(proposal).tolist(),
                                     take=cuda(updated[sel]).tolist())
        return updated

    stack.enter_context(model.model.decoder.register_forward_pre_hook(dec, with_kwargs=True))
    stack.enter_context(model.model.encoder.register_forward_pre_hook(enc, with_kwargs=True))
    stack.enter_context(patch.object(unmask, "low_confidence_dynamic_sparse", sample))
    for i, layer in enumerate(model.model.decoder.layers):
        for key in ("q_proj", "k_proj", "v_proj"):
            module = getattr(layer.self_attn, key)
            if key == "v_proj" and module is None:
                continue
            tape = projections[f"{i}.{key}"] = []

            def project(_, args, tape=tape):
                x, = args
                require(cuda(x).ndim == 3 and x.shape[0] == 1, "Nonphysical batch or projection")
                tape.append(x.shape[1])

            stack.enter_context(module.register_forward_pre_hook(project))
    return decoder, encoder, projections, events, rng


def reconcile(trace, decoder, encoder, events, output, prompt, eos, trigger, payment, q):
    origin = len(prompt)
    expected_enc = [[prompt, list(range(origin))]]
    expected_events = [["E", origin]]
    delivery, states = [], []
    head, ready, tokens, dirty, seen = -1, 0, {}, set(), set()
    admissions = paid_rows = 0
    require(trace and len(trace) == len(decoder) and trace[0]["q"] == q,
            "Missing true opening projection at registered Q")
    for i, (t, d) in enumerate(zip(trace, decoder, strict=True)):
        if t["block"] != head:
            require(t["block"] == head + 1 and (head < 0 or ready == B and not dirty),
                    "Bad block transition")
            head, ready, tokens, seen = t["block"], 0, {}, set()
        start = head * B
        rows = t["positions"]
        picked = {x-start for x in rows}
        width = len(rows)
        require(t["tick"] == i and t["offer"] == q and t["cross"] == 0
                and 0 < width == t["q"] == len(picked) <= q
                and rows == sorted(rows) and 0 <= min(picked) <= max(picked) < B
                and d["pos"] == [origin+x for x in rows] and t["visible_end"] == max(rows)+1,
                f"Physical row mismatch at {i}")
        expected_events.append(["D", width])
        seen |= picked
        require(set(range(max(picked)+1)) <= seen, "Visible KV hole")
        require(all(p not in tokens or tokens[p] == token for p, token in
                    zip((x-start for x in rows), d["ids"], strict=True)), "Immutable ID changed")
        mask = sorted(picked - tokens.keys())
        remaining = dirty - picked
        sample = d["sample"]
        decision = {}
        require(bool(mask) == (sample is not None) and not (remaining and mask), "Sampling before debt")
        if sample is not None:
            require(sample["rows"] == mask, "Sampler did not receive all selected unresolved rows")
            decision = {p: x for p, x, keep in zip(mask, sample["proposal"], sample["take"], strict=True) if keep}
            require(decision, "No sampling progress")
        require(t["admitted"] == [start+p for p in decision] and t["accepted"] == len(decision),
                "Admission/sampler mismatch")
        admissions += len(decision)
        paid_rows += len(dirty & picked)
        tokens.update(decision)
        dirty = remaining | decision.keys()
        require(ready <= t["ready"] <= B
                and set(range(t["ready"])) <= tokens.keys() - dirty, "Frozen prefix contains debt or holes")
        ready = t["ready"]
        first = next((p for p in range(ready) if tokens[p] in eos), None)
        states.append(dict(ready=ready, eos_index=first, eos_id=tokens.get(first),
                           debt=[start+x for x in sorted(dirty)]))
        if ready == B and i+1 < len(trace):
            require(not dirty and first is None, "Stock crossed a terminal head")
            block = [tokens[p] for p in range(B)]
            delivery += block
            expected_enc.append([block, list(range(origin+start, origin+start+B))])
            expected_events.append(["E", B])
    if trigger is None:
        require(ready == B and (first is not None or head == CAP//B-1), "Uncertified termination")
        end = B if first is None else first+1
    else:
        end = trigger["eos_index"]+1
        require(end <= ready and first == end-1 and tokens[first] == trigger["eos_id"]
                and head == trigger["block"], "Bad terminal certificate")
        if payment is not None:
            require(payment == dict(tick=len(trace)-1, positions=trace[-1]["positions"], Q=trace[-1]["q"])
                    and payment["tick"] == trigger["tick"]+1 and payment["positions"] == trigger["debt"]
                    and decoder[-1]["sample"] is None, "Payment was not exactly one real suffix-only forward")
        else:
            require(trigger["kind"] == "clean" and trigger["tick"] == len(trace)-1, "Payment missing")
    require(not dirty and admissions == paid_rows and
            output == delivery+[tokens[p] for p in range(end)] and
            encoder == expected_enc and events == expected_events, "Final IDs, debt, or physical ledger")
    d, e = len(decoder), len(encoder)-1
    return dict(D=d, E=e, F=d+e, Q=sum(t["q"] for t in trace), encoder_Q=B*e,
                peak_decoder_Q=max(t["q"] for t in trace), admissions=admissions, paid_rows=paid_rows), states


def run_arm(rt, torch, model, data, proc, controls, unmask, task, mode, scoring, q):
    stop = None if mode == "stock" else Stop(mode, q)
    prompt = data["input_ids"][0].tolist()
    eos = set(model.generation_config.eos_token_id)
    extras = set(proc.tokenizer.all_special_ids) | eos | {model.generation_config.pad_token_id}
    torch.manual_seed(task["seed"])
    with torch.inference_mode(), ExitStack() as stack:
        decoder, encoder, projections, events, rng = observe(model, unmask, stack, torch)
        if stop is None:
            ids, trace = rt.generate(model, data, controls, unmask, "ibs", q, caps=(q,))
        else:
            ids, trace = generate(rt, model, data, controls, unmask, stop)
        after = rng()
    widths = [t["q"] for t in trace]
    require(len(projections) == 85 and all(v == widths for v in projections.values()), "Real Q/K/V widths")
    certs = {k: getattr(stop, k, None) for k in ("first_clean", "first_paid", "trigger", "payment")}
    bill, states = reconcile(trace, decoder, encoder, events, ids, prompt, eos, certs["trigger"], certs["payment"], q)
    score = scoring.score(proc.tokenizer.decode(ids, skip_special_tokens=True), task)
    return dict(ids=ids, trace=trace, decoder=decoder, encoder=encoder, events=events, states=states,
                projections=sorted(projections), rng_after=after, bill=bill, **certs,
                correct=bool(score["correct"]), scoring=score["metric"], N=sum(i not in extras for i in ids))


def match(arms, q):
    stock, control = (arms[k] for k in ("stock", "control"))
    cert_keys = {"first_clean", "first_paid", "trigger", "payment"}
    def stable(a):
        return {k: v for k, v in a.items() if k not in cert_keys}
    require(stable(stock) == stable(control), "Stock and predicate-only control diverged")
    first_clean = first_paid = None
    for t, s in zip(stock["trace"], stock["states"], strict=True):
        if s["eos_index"] is None or s["ready"] == B:
            continue
        start = t["block"]*B
        debt = s["debt"]
        if not debt or (len(debt) <= q and all(start+s["eos_index"] < x < start+B for x in debt)):
            cert = dict(kind="suffix_debt" if debt else "clean", tick=t["tick"], block=t["block"], **s)
            first_paid = first_paid or cert
            if not debt:
                first_clean = first_clean or cert
    require(control["first_clean"] == first_clean and control["first_paid"] == first_paid,
            "Control certificate disagrees with independent stock ledger")
    for mode, expected in (("strict", first_clean), ("paid", first_paid)):
        arm = arms[mode]
        hit = arm["trigger"]
        require(hit == expected, f"{mode} did not stop at its first eligible tick")
        if hit is None:
            require(stable(arm) == stable(stock), f"Unexposed {mode} diverged")
            continue
        end = hit["tick"]+1
        require(all(arm[k][:end] == stock[k][:end] for k in ("trace", "decoder", "states"))
                and arm["encoder"] == stock["encoder"] and arm["ids"] == stock["ids"]
                and arm["correct"] == stock["correct"] and arm["N"] == stock["N"]
                and end < len(stock["decoder"])
                and arm["rng_after"] == stock["decoder"][end]["rng"], f"{mode} diverged by trigger or delivery")
        payment = arm["payment"]
        require((payment is not None) == (hit["kind"] == "suffix_debt")
                and len(arm["trace"]) == end+int(payment is not None), f"Wrong {mode} retirement")
        if payment is not None:
            require(arm["decoder"][-1]["rng"] == arm["rng_after"], "Payment modified RNG")


def main(q):
    guard("before")
    import torch
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from bdlm_serving.evaluation import frozen, frozen90, scoring
    from bdlm_serving.methods.optimus import upstream
    from bdlm_serving.runtime import diffusiongemma_optimus as rt, optimus_limits

    reg = Path(os.environ["IBS_SOURCE"])
    model_manifest = frozen.ROOT/"benchmarks/checkpoints/diffusiongemma.json"
    snapshot = json.loads(Path("/gs/runtime/gemma-ibs-prefix-bridge-a1-1128/checkpoint.json").read_bytes())
    pins = {
        rt.__file__: BASELINE_SHA,
        optimus_limits.__file__: "842f078e06b5f79b336165b060d1212e947e01182e167379a1d1dbb9f62b5376",
        scoring.__file__: "a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692",
        str(SAMPLES): "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed",
        str(model_manifest): MODEL_MANIFEST,
        upstream.__file__: "f21a4704697cace53b19dfbbbf94bbabc877241b9c979e15b0d40e9a2ea0d223",
    }
    versions = dict(torch=str(torch.__version__), transformers=version("transformers"), math_verify=version("math-verify"))
    require(versions == dict(torch="2.10.0+cu128", transformers="5.12.0", math_verify="0.8.0")
            and all(sha(k) == v for k, v in pins.items()) and snapshot["checkpoint"] == str(MODEL)
            and snapshot["manifest"] == MODEL_MANIFEST and snapshot["entries"] == 19, "Pin mismatch")
    controls, strategy, hashes = upstream.load_controls()
    require(hashes == OPTIMUS_FILES and upstream.COMMIT == OPTIMUS, "Wrong Optimus source")
    manifest = (reg/"sources.sha256").read_text()
    subprocess.run(["sha256sum", "-c", "sources.sha256"], cwd=reg, check=True)
    emit("registration", pid=os.getpid(), gpu=GPU, q=q, arms=ARMS, six=SIX, snapshot=snapshot,
         local_head=os.environ["IBS_LOCAL_HEAD"], sources_sha256=manifest, versions=versions, pins=pins,
         optimus=OPTIMUS, optimus_files=hashes, boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
         start_ticks=Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19])
    require(torch.cuda.device_count() == 1, "Expected exactly one visible GPU")
    torch.cuda.init()
    torch.zeros(1, device="cuda:0")
    guard("initialized")
    proc = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
        experts_implementation="grouped_mm", device_map={"": "cuda:0"}).eval()
    require(model.config.canvas_length == B and
            all(p.device == torch.device("cuda:0") and p.dtype == torch.bfloat16 for p in model.parameters()),
            "Not the pinned GPU bf16 model")
    unmask = controls.UnmaskingProcessor(SimpleNamespace(
        confidence_threshold=.9, block_length=B, denoising_steps=B, unmasking_strategy=strategy.LOW_CONFIDENCE_DYNAMIC))
    tasks = frozen90.load_tasks(SAMPLES)
    totals = {arm: Counter() for arm in ARMS}
    for j, i in enumerate((*SIX, *(i for i in range(90) if i not in SIX))):
        guard("request")
        t = tasks[i]
        require(t["index"] == i, "Wrong frozen task index")
        data = proc.apply_chat_template([{"role": "user", "content": t["prompt"]}], tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda:0")
        order = ARMS if j % 2 == 0 else ARMS[::-1]
        arms = {m: run_arm(rt, torch, model, data, proc, controls, unmask, t, m, scoring, q) for m in order}
        try:
            match(arms, q)
        except Exception:
            emit("failed_pair", index=i, arms=arms)
            raise
        for m, a in arms.items():
            payment = a["payment"]
            totals[m].update({**a["bill"], "peak_decoder_Q": 0, "N": a["N"], "correct": int(a["correct"]),
                              "hits": int(a["trigger"] is not None), "payments": int(payment is not None),
                              "payment_Q": payment["Q"] if payment else 0})
            totals[m]["peak_decoder_Q"] = max(totals[m]["peak_decoder_Q"], a["bill"]["peak_decoder_Q"])
        case = {k: t[k] for k in ("index", "task_id", "seed", "dataset")}
        menu = {m: dict(bill=a["bill"], trigger=a["trigger"], payment=a["payment"]) for m, a in arms.items()}
        emit("case", **case, order=order, arms=arms)
        print(f"SEALED {i:02d} {j+1}/90 " + json.dumps(menu, separators=(",", ":")), flush=True)
        if j == len(SIX)-1:
            emit("gate_passed", indices=SIX, totals=totals)
            print("GATE_PASSED 6/6", flush=True)
    emit("complete", requests=90, totals=totals)
    print("COMPLETE 90/90", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", required=True, type=int, choices=OFFERS)
    main(parser.parse_args().q)
