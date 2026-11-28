"""Three-arm A3-only frozen-six with independent decoder/encoder reconciliation."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
from types import SimpleNamespace

from experiments.t2t.serving_slots import gemma_paid_agreement_eval as ev
from experiments.t2t.serving_slots.gemma_provisional_recheck import Recheck, RecheckStop, generate
from experiments.t2t.serving_slots.gemma_ibs_certified_eos import B, CAP, require


def digest(obj):
    return ev.sha256(json.dumps(obj, separators=(",", ":")).encode()).hexdigest()


def emit(kind, **fields):
    print("RECEIPT " + json.dumps(dict(kind=kind, **fields), separators=(",", ":"), allow_nan=False), flush=True)


def replay(trace, decoder, encoder, physical, ids, data, eos, stop, recheck):
    prompt = data["input_ids"][0].tolist()
    enc = [[prompt, list(range(len(prompt)))]]
    events = [["E", len(prompt)]]
    dirty, seen, tokens, delivery, states = set(), set(), {}, [], []
    edits = {}
    for e in recheck.corrections:
        edits.setdefault(e["payment"], []).append(e)
    paid = admissions = repaid = 0
    head, ready = -1, 0
    for i, (t, d) in enumerate(zip(trace, decoder, strict=True)):
        if head != t["block"]:
            require(t["block"] == head+1 and (head < 0 or ready == B and not dirty), "Bad new block")
            head, ready, seen, tokens = t["block"], 0, set(), {}
        start = head * B
        rows = t["positions"]
        selected = set(rows)
        require(t["tick"] == i and t["offer"] == ev.Q and t["q"] == len(rows) == len(selected)
                and 0 < len(rows) <= ev.Q and rows == sorted(rows)
                and start <= rows[0] <= rows[-1] < start+B and t["cross"] == 0
                and d["pos"] == [len(prompt)+p for p in rows], "Physical Q or positions")
        require(all(p not in tokens or tokens[p] == v for p, v in zip(rows, d["ids"], strict=True)),
                "Decoder input ID differs from published ID")
        events.append(["D", len(rows)])
        seen.update(rows)
        require(set(range(start, rows[-1]+1)) <= seen and t["visible_end"] == rows[-1]+1, "KV hole")
        paid += len(dirty & selected)
        for e in recheck.corrections:
            if e["repayment"] == i:
                p = e["pos"]
                require(p in (dirty & selected) and d["ids"][rows.index(p)] == e["top"],
                        "Missing replacement projection")
                repaid += 1
        first_paid = dirty & selected
        dirty -= selected
        mask = [p for p in rows if p not in tokens]
        require(bool(d["sample"]) == bool(mask) and not (mask and dirty), "Sampling ahead of debt")
        writes = {}
        if mask:
            s = d["sample"]
            require([start+p for p in s["rows"]] == mask, "Wrong repeat sampler rows")
            writes = {p: v for p, v, keep in zip(mask, s["proposal"], s["take"], strict=True) if keep}
        require(t["admitted"] == list(writes) and t["accepted"] == len(writes), "Ordinary write mismatch")
        tokens.update(writes)
        admissions += len(writes)
        dirty.update(writes)
        for e in edits.get(i, ()):
            p = e["pos"]
            require(p >= start+ready and p in first_paid and p not in writes and tokens[p] == e["old"]
                    and e["repayment"] is not None and e["repayment"] > i, "Invalid correction write")
            tokens[p] = e["top"]
            dirty.add(p)
        require(ready <= t["ready"] <= B and set(range(start, start+t["ready"])) <= tokens.keys()-dirty,
                "Debt crossed the frozen prefix")
        ready = t["ready"]
        first = next((p for p in range(start, start+ready) if tokens[p] in eos), None)
        states.append(dict(ready=ready, debt=sorted(dirty), eos=first))
        if ready == B and i+1 < len(trace) and trace[i+1]["block"] != head:
            require(not dirty and first is None, "Encoder crossed unpaid/terminal boundary")
            block = [tokens[p] for p in range(start, start+B)]
            enc.append([block, list(range(len(prompt)+start, len(prompt)+start+B))])
            events.append(["E", B])
            delivery.extend(block)
    if stop.trigger:
        require(first == head*B+stop.trigger["eos_index"] and first is not None and
                (stop.trigger["kind"] == "clean" or stop.payment), "Uncertified EOS")
    else:
        require(ready == B and (first is not None or head == CAP//B-1), "Uncertified block end")
    end = head*B+B if first is None else first+1
    require(not dirty and not recheck.pending and not recheck.replacements and trace[0]["q"] == ev.Q
            and len(recheck.measurements) == admissions and paid == admissions + len(recheck.corrections)
            and repaid == len(recheck.corrections) and ids == delivery+[tokens[p] for p in range(head*B, end)]
            and encoder == enc and physical == events, "Final output, payment, encoder or decoder bill")
    d, e = len(trace), len(enc)-1
    return dict(D=d, E=e, F=d+e, Q=sum(t["q"] for t in trace), encoder_Q=B*e,
                peak_decoder_Q=max(t["q"] for t in trace), admissions=admissions, paid_rows=paid), states


def run_extra(rt, torch, model, data, proc, controls, unmask, task, scoring, special, correct):
    stop = RecheckStop(ev.Q)
    torch.manual_seed(task["seed"])
    with torch.inference_mode(), ExitStack() as stack:
        decoder, encoder, projections, physical, rng = ev.observe(model, unmask, stack, torch)
        recheck = Recheck(correct=correct, excluded_ids=special, torch=torch, rng=rng)
        recheck.record_take = lambda take: decoder[-1]["sample"].update(take=take)
        ids, trace = generate(rt, model, data, controls, unmask, stop, recheck)
        after = rng()
    require(len(projections) == 85 and all(w == [t["q"] for t in trace] for w in projections.values()),
            "Wrong count of real Q/K/V projections")
    bill, states = replay(trace, decoder, encoder, physical, ids, data,
                          set(model.generation_config.eos_token_id), stop, recheck)
    score = scoring.score(proc.tokenizer.decode(ids, skip_special_tokens=True), task)
    return dict(ids=ids, trace=trace, decoder=decoder, encoder=encoder, rng_after=after,
                bill=bill, states=states, observations=recheck.observations, first_extra=recheck.first_extra,
                measurements=recheck.measurements, corrections=recheck.corrections, same_step=recheck.same_step,
                terminal_payments=stop.terminal_payments, trigger=stop.trigger, payment=stop.payment,
                correct=bool(score["correct"]), scoring=score["metric"], N=sum(p not in special for p in ids))


def compare(repeat, observer, corrected, old):
    hist = old["arms"]["apply"]
    for k in ("bill", "trigger", "payment", "correct", "scoring", "N", "rng_after"):
        require(repeat[k] == hist[k], f"Historical repeat drift: {k}")
    require(repeat["agreement"]["first_extra"] == hist["agreement"]["first_extra"] and
            [r for r in repeat["agreement"]["observations"] if r["extra"]] == hist["agreement"]["extra_events"],
            "Historical extra admissions differ")
    require(observer["measurements"] == old["payment_observer"]["events"], "Historical first-payment tape differs")
    keys = ("ids", "trace", "decoder", "encoder", "rng_after", "bill", "trigger", "payment", "correct", "scoring", "N")
    require(all(repeat[k] == observer[k] for k in keys) and
            observer["observations"] == repeat["agreement"]["observations"], "No-op observer altered repeat")
    first = min((r["payment"] for r in corrected["corrections"]), default=None)
    if first is None:
        require(corrected == observer, "Nonexposed treatment must exactly reproduce observer")
        return None
    require(corrected["trace"][:first] == observer["trace"][:first] and
            corrected["decoder"][:first+1] == observer["decoder"][:first+1] and
            corrected["states"][:first] == observer["states"][:first] and
            corrected["same_step"][:first+1] == observer["same_step"][:first+1] and
            [o for o in corrected["observations"] if o["tick"] <= first] ==
            [o for o in observer["observations"] if o["tick"] <= first],
            "Base sample or model input differed by the first correction")
    return dict(tick=first, q=corrected["trace"][first]["q"],
                positions=[e["pos"] for e in corrected["corrections"] if e["payment"] == first],
                stock_sample=observer["decoder"][first]["sample"],
                post_sample=observer["same_step"][first])


def main():
    ev.guard("before")
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from bdlm_serving.evaluation import frozen, frozen90, scoring
    from bdlm_serving.methods.optimus import upstream
    from bdlm_serving.runtime import diffusiongemma_optimus as rt, optimus_limits
    require(torch.cuda.device_count() == 1 and str(torch.__version__) == "2.6.0+cu124"
            and transformers.__version__ == "5.12.0", "Environment pin")
    controls, strategy, hashes = upstream.load_controls()
    require(hashes == ev.OPTIMUS_FILES and upstream.COMMIT == ev.OPTIMUS, "Optimus pin")
    pins = {rt.__file__: ev.BASELINE_SHA, scoring.__file__: "a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692",
            optimus_limits.__file__: "842f078e06b5f79b336165b060d1212e947e01182e167379a1d1dbb9f62b5376",
            upstream.__file__: "f21a4704697cace53b19dfbbbf94bbabc877241b9c979e15b0d40e9a2ea0d223",
            ev.SAMPLES: "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed",
            frozen.ROOT / "benchmarks/checkpoints/diffusiongemma.json": ev.MODEL_MANIFEST}
    require(all(ev.sha(p) == h for p, h in pins.items()), "Source, task or model pin")
    reference = Path(os.environ["RECHECK_REFERENCE"])
    refs = {i: json.loads((reference / f"case-{i:02}.json").read_bytes()) for i in ev.CASES}
    emit("start", pid=os.getpid(), gpu=ev.GPU, q=ev.Q, cases=ev.CASES, pins={str(k): v for k, v in pins.items()},
         references={i: ev.sha(reference / f"case-{i:02}.json") for i in ev.CASES})
    proc = AutoProcessor.from_pretrained(ev.MODEL, local_files_only=True)
    ev.guard("before")
    torch.zeros(1, device="cuda:0")
    ev.guard("load")
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        ev.MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
        experts_implementation="grouped_mm", device_map={"": "cuda:0"}).eval()
    require(model.config.canvas_length == B and all(p.device == torch.device("cuda:0") and
            p.dtype == torch.bfloat16 for p in model.parameters()), "Model placement/dtype")
    unmask = controls.UnmaskingProcessor(SimpleNamespace(
        confidence_threshold=.9, block_length=B, denoising_steps=B,
        unmasking_strategy=strategy.LOW_CONFIDENCE_DYNAMIC))
    tasks = frozen90.load_tasks(ev.SAMPLES)
    special = set(proc.tokenizer.all_special_ids) | set(model.generation_config.eos_token_id) | {model.generation_config.pad_token_id}
    for j, i in enumerate(ev.CASES):
        ev.guard("request")
        task = tasks[i]
        require(all(task[k] == refs[i][k] for k in ("index", "task_id", "dataset", "seed")), "Wrong task")
        data = proc.apply_chat_template([{"role": "user", "content": task["prompt"]}], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda:0")
        args = (rt, torch, model, data, proc, controls, unmask, task)
        order = ("repeat", "observer", "recheck")
        if j % 2:
            order = order[::-1]
        arms = {}
        for name in order:
            emit("stage", index=i, arm=name)
            arms[name] = (ev.run_arm(*args, "apply", scoring, special) if name == "repeat" else
                          run_extra(*args, scoring, special, name == "recheck"))
        try:
            first = compare(arms["repeat"], arms["observer"], arms["recheck"], refs[i])
        except Exception:
            emit("unverified", index=i, order=order, arms=arms)
            raise
        emit("case", index=i, order=order, first=first, arms=arms)
    emit("complete", cases=ev.CASES)


if __name__ == "__main__":
    main()
