"""A800 physical A3: six-case Q128 certified paid-EOS + paid-agreement pairing.

Arms: stock paid-close, observer-only paid-close, apply paid-close. All use the
same model load, seed and certified paid-EOS Stop. The observer arm runs the
ledger without applying it; the apply arm writes the extra accepted rows into
the stock IBS unmask result before the stock take/noise write.
"""
import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from experiments.t2t.serving_slots.gemma_ibs_certified_eos import (
    BASELINE_SHA, B, OFFERS, Stop, generate as certified_generate, require)
from experiments.t2t.serving_slots.gemma_paid_agreement import (
    Agreement, generate as agreement_generate)
from tools.run_gemma_ibs_certified_eos import observe, reconcile

GPU = "GPU-94c61e79-d80b-cae2-d85f-6bd2a2c99a1c"
Q = 128
CASES = (7, 13, 25, 33, 55, 71)
ARMS = ("stock", "observer", "apply")
MODEL = Path("/gs/models/diffusiongemma-cd40195")
SAMPLES = Path("/gs/code/samples.json")
MODEL_MANIFEST = "ceb951734eb354728d7ba984093739119addbeda981b554af46d5dc105071f53"
OPTIMUS = "7343a37b2a69337caa452f17b59727dc6665f039"
OPTIMUS_FILES = {
    "sequence.py": "99c80047d517113579e04fd57443471a759a738dc30b206ba1342e0d4f9b3da2",
    "unmasking.py": "bbc3c3328e5497aa08aed6e04150ab0383f95dc06b4ff570a69adaed4a769746",
}


def sha(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def guard(stage):
    apps = subprocess.check_output(
        ["nvidia-smi", "-i", "3", "--query-compute-apps=pid,gpu_uuid",
         "--format=csv,noheader,nounits"], text=True).strip()
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", "3", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True).strip()
    require(uuid == os.environ.get("CUDA_VISIBLE_DEVICES") == GPU
            and (not apps if stage == "before" else apps == f"{os.getpid()}, {GPU}"),
            f"Unexpected A3 state at {stage}: {apps}")


def run_arm(rt, torch, model, data, proc, controls, unmask, task, mode, scoring, special):
    stop = Stop("paid", Q)
    agreement = None if mode == "stock" else Agreement(
        apply=(mode == "apply"), excluded_ids=special, floor=0.5)
    prompt = data["input_ids"][0].tolist()
    eos = set(model.generation_config.eos_token_id)
    extras = set(proc.tokenizer.all_special_ids) | eos | {model.generation_config.pad_token_id}
    torch.manual_seed(task["seed"])
    with torch.inference_mode(), ExitStack() as stack:
        decoder, encoder, projections, events, rng = observe(model, unmask, stack, torch)
        if agreement is not None:
            agreement.record_take = lambda take: decoder[-1]["sample"].update(take=take)
        if agreement is None:
            ids, trace = certified_generate(rt, model, data, controls, unmask, stop)
        else:
            ids, trace = agreement_generate(rt, model, data, controls, unmask, stop, agreement)
        after = rng()
    widths = [t["q"] for t in trace]
    require(len(projections) == 85 and all(v == widths for v in projections.values()),
            "Real Q/K/V widths")
    certs = {k: getattr(stop, k, None) for k in ("first_clean", "first_paid", "trigger", "payment")}
    bill, states = reconcile(trace, decoder, encoder, events, ids, prompt, eos,
                             certs["trigger"], certs["payment"], Q)
    score = scoring.score(proc.tokenizer.decode(ids, skip_special_tokens=True), task)
    return dict(ids=ids, trace=trace, decoder=decoder, encoder=encoder, events=events,
                states=states, projections=sorted(projections), rng_after=after, bill=bill,
                **certs, correct=bool(score["correct"]), scoring=score["metric"],
                N=sum(i not in extras for i in ids),
                agreement=None if agreement is None else dict(
                    observations=agreement.observations,
                    first_extra=agreement.first_extra, applied=agreement.apply))


def stable(a):
    return {k: v for k, v in a.items() if k != "agreement"}


def check_fork(first, a, b, da, db, oa, ob):
    """The sampler numbers rows inside a block; the ledger numbers generated positions."""
    require(a["block"] == b["block"] == first["block"] and oa == ob and
            oa["block"] == first["block"] and oa["tick"] == first["tick"] and
            list(oa["extra"]) == list(first["extra"]), "First paid observations drifted")
    require(a["positions"] == b["positions"] and a["offer"] == b["offer"] and
            a["q"] == b["q"] and all(da[k] == db[k] for k in ("ids","pos","rng")),
            "First extra changed model input or RNG")
    sa, sb = da["sample"], db["sample"]
    require(sa["rows"] == sb["rows"] and sa["proposal"] == sb["proposal"] == oa["proposals"] and
            list(map(bool, sb["take"])) == oa["native"], "First sampler input or base decision drifted")
    absolute = [first["block"] * B + row for row in sa["rows"]]
    extra = set(first["extra"])
    require(absolute == oa["positions"] and extra <= set(absolute) and
            not (extra & set(b["admitted"])) and set(a["admitted"]) == set(b["admitted"]) | extra
            and list(map(bool, sa["take"])) == [v or p in extra for p,v in zip(absolute,oa["native"],strict=True)],
            "First extra admission differs from the paid row set")


def match(arms):
    stock, observer, apply = (arms[k] for k in ("stock", "observer", "apply"))
    require(stable(stock) == stable(observer), "Stock and observer-only paid-close diverged")
    first = None if apply["agreement"] is None else apply["agreement"]["first_extra"]
    if first is None:
        require(stable(apply) == stable(stock), "Unexposed apply diverged")
        return None
    end = first["tick"]
    require(apply["trace"][:end] == observer["trace"][:end]
            and apply["decoder"][:end] == observer["decoder"][:end]
            and apply["states"][:end] == observer["states"][:end],
            "Apply diverged before the first extra admission")
    observations = [next(o for o in v["agreement"]["observations"] if o["tick"] == end)
                    for v in (apply, observer)]
    check_fork(first, apply["trace"][end], observer["trace"][end], apply["decoder"][end],
               observer["decoder"][end], *observations)
    return first


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--deadline", type=int, required=True)
    a = ap.parse_args()
    require(a.root.is_dir() and str(a.root).startswith("/dev/shm/"), "Use a /dev/shm root")
    guard("before")
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from bdlm_serving.evaluation import frozen, frozen90, scoring
    from bdlm_serving.methods.optimus import upstream
    from bdlm_serving.runtime import diffusiongemma_optimus as rt, optimus_limits
    from bdlm_serving.runtime.diffusiongemma import check_snapshot
    require(transformers.__version__ == "5.12.0", "Wrong transformers version")
    controls, strategy, hashes = upstream.load_controls()
    require(hashes == OPTIMUS_FILES and upstream.COMMIT == OPTIMUS, "Wrong Optimus source")
    require(sha(rt.__file__) == BASELINE_SHA, "Wrong IBS runtime source")
    require(sha(scoring.__file__) == "a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692"
            and sha(upstream.__file__) == "f21a4704697cace53b19dfbbbf94bbabc877241b9c979e15b0d40e9a2ea0d223"
            and sha(optimus_limits.__file__) == "842f078e06b5f79b336165b060d1212e947e01182e167379a1d1dbb9f62b5376",
            "Wrong scorer/upstream/limits source")
    require(sha(SAMPLES) == "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed",
            "Wrong frozen samples")
    check_snapshot(MODEL)
    require(sha(frozen.ROOT / "benchmarks/checkpoints/diffusiongemma.json") == MODEL_MANIFEST,
            "Wrong model manifest")
    proc = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
        experts_implementation="grouped_mm", device_map={"": "cuda:0"}).eval()
    require(model.config.canvas_length == B, "Wrong canvas length")
    unmask = controls.UnmaskingProcessor(SimpleNamespace(
        confidence_threshold=.9, block_length=B, denoising_steps=B,
        unmasking_strategy=strategy.LOW_CONFIDENCE_DYNAMIC))
    tasks = frozen90.load_tasks(SAMPLES)
    eos = set(model.generation_config.eos_token_id)
    special = set(proc.tokenizer.all_special_ids) | eos | {model.generation_config.pad_token_id}
    per_case = []
    for j, index in enumerate(CASES):
        guard("request")
        require(time.time() < a.deadline, "Deadline")
        task = next(t for t in tasks if t["index"] == index)
        data = proc.apply_chat_template(
            [{"role": "user", "content": task["prompt"]}], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda:0")
        order = ARMS if j % 2 == 0 else ARMS[::-1]
        arms = {m: run_arm(rt, torch, model, data, proc, controls, unmask, task, m, scoring, special)
                for m in order}
        first = match(arms)
        row = dict(
            index=index, task_id=task["task_id"], dataset=task["dataset"], seed=task["seed"],
            order=order,
            arms={m: dict(bill=v["bill"], trigger=v["trigger"], payment=v["payment"],
                          correct=v["correct"], N=v["N"], scoring=v["scoring"],
                          rng_after=v["rng_after"],
                          agreement=None if v["agreement"] is None else dict(
                              first_extra=v["agreement"]["first_extra"],
                              applied=v["agreement"]["applied"],
                              extra_events=[o for o in v["agreement"]["observations"] if o["extra"]]))
                  for m, v in arms.items()},
            first_extra=first,
            stock_observer_identical=stable(arms["stock"]) == stable(arms["observer"]))
        per_case.append(row)
        (a.root / f"case-{index:02}.json").write_text(json.dumps(row, separators=(",", ":")) + "\n")
        print(json.dumps(dict(index=index, first_extra=first,
                              identical=row["stock_observer_identical"]),
                         separators=(",", ":")), flush=True)
    gate = ("passed" if all(r["stock_observer_identical"] for r in per_case) else "failed")
    (a.root / "summary.json").write_text(json.dumps(dict(
        schema="gemma-paid-agreement-q128-v1", gpu=GPU, q=Q, cases=list(CASES),
        gate=gate, per_case=per_case, torch=str(torch.__version__),
        transformers=transformers.__version__, model=str(MODEL), samples=str(SAMPLES),
        baseline_sha=BASELINE_SHA, optimus=OPTIMUS), indent=2) + "\n")
    print(json.dumps(dict(gate=gate), separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
