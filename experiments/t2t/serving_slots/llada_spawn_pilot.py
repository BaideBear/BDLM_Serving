"""CPU BF16, fixed Q64: tail admission x cross-block mask, frozen indices 0/10/20.

From repo root, with PYTHONPATH=.:../fixed-baseline-deps and CUDA_VISIBLE_DEVICES='',
run `taskset -c 40-47 ../venv/bin/python -um experiments.t2t.serving_slots.llada_spawn_pilot`.
Private per-request artifacts and sealed aggregate are written only on the cluster.
"""
from collections import Counter
import os
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from experiments.t2t.serving_slots import llada_all_slots90 as reg
from experiments.t2t.serving_slots.llada_mask_pilot import CFG, CKPT, INDICES

OUT = reg.DATA / "llada21-q64-admission-mask-pilot3"
ARMS = {f"{gate}-{mask}": (threshold, mask == "open")
        for gate, threshold in (("eager", None), ("progress80", .8))
        for mask in ("causal", "open")}


def generate(model, tok, task, prompt, name):
    import torch
    from bdlm_serving.methods.t2t.online import package
    from experiments.t2t.serving_slots.llada_quote import quote
    from experiments.t2t.serving_slots.llada_spawn import SpawnGatedKernel
    from experiments.t2t.serving_slots.mask_pair_llada import head_to_tail_mask

    threshold, opened = ARMS[name]
    k = SpawnGatedKernel(model, spawn_threshold=threshold)
    eos, mask = tok.eos_token_id, tok.convert_tokens_to_ids("<|mask|>")
    s = k.create(torch.tensor([prompt]), CFG, reg.CAP, task["seed"], mask, [eos])
    receipts, witnesses, prices = [], [], []
    first = None

    def swap(_module, args, kw):
        nonlocal first
        ids, bias = kw["position_ids"], kw["attention_mask"]
        pos, end = ids[0].tolist(), bias.shape[-1]
        native = ids[0][:, None] // 32 >= torch.arange(end)[None, :] // 32
        reg.require(torch.equal(bias[0, 0], native), "Invalid incoming mask")
        changed, info = head_to_tail_mask(bias, ids, s.prompt_length, s.block)
        reg.require(info == reg.geometry(s.block, s.prompt_length, pos, end), "Wrong head/tail edges")
        if info and first is None:
            first = reg.digest(dict(earlier=s.events[:-1], canvas=s.x[0].tolist(), write=s.last_write,
                               query=s.last_query, context=s.context_version, committed=s.committed_end,
                               rng=s.rng_cpu.tolist(), positions=pos, end=end, block=s.block,
                               history=receipts, tick=s.offer_tick))
        receipts.append(dict(pos=pos, visible=end, eligible=info["opened_edges"] if info else 0))
        if opened and info:
            return args, dict(kw, attention_mask=changed)

    def witness(_module, args, kw):
        bias, pos = kw["attention_mask"][0, 0], kw["position_ids"][0]
        keys = torch.arange(bias.shape[-1])
        native = pos[:, None] // 32 >= keys[None, :] // 32
        boundary = (s.block + 1) * 32
        window = ((pos >= max(boundary-32, s.prompt_length)) & (pos < boundary))[:, None] & (
                  (keys >= boundary) & (keys < boundary+32))[None, :]
        expected = native | window if opened else native
        reg.require(torch.equal(bias, torch.zeros_like(bias).masked_fill(
            ~expected, torch.finfo(bias.dtype).min)), "Bad effective mask in attention")
        witnesses.append(int((~native & (bias == 0)).sum()))

    def offer(tick):
        prices.append(quote(k, s, 64))
        return 64

    with model.model.register_forward_pre_hook(swap, with_kwargs=True), \
            model.model.layers[0].attention.register_forward_pre_hook(witness, with_kwargs=True):
        k.run(s, offer)
    reg.require(len(receipts) == len(s.events) == len(witnesses), "Unobserved call")
    for e, r, w in zip(s.events, receipts, witnesses, strict=True):
        reg.require(e["absolute_positions"] == r["pos"] and e["visible_kv"] == r["visible"]
                    and w == (r["eligible"] if opened else 0), "Witness disagrees with execution")
    gen = [e for e in s.events if e["phase"] != "prefill"]
    for e, q in zip(gen, prices, strict=True):
        reg.require((e["q_budget"], e["top_action"], e["physical_q"], e["refreshed_pending"]) ==
                    (64, q["action"], q["physical_q"], q["mandatory_positions"]), "Wrong Q64 quote")
    ids = k.output(s)[0].tolist()
    return package(tok, ids, [(e["stage"],e["physical_q"]) for e in gen], [eos],
                   set(tok.all_special_ids) | {mask, eos}, sum(e["physical_q"] for e in s.events
                   if e["phase"] == "prefill"), first=first, events=s.events, prompt_ids=prompt,
                   opportunities=sum(bool(r["eligible"]) for r in receipts), opened_edges=sum(witnesses),
                   index=task["index"], policy=name, seed=task["seed"], task_id=task["task_id"])


def compare(on, off):
    keys = ("index", "task_id", "seed", "prompt_ids", "first")
    reg.require(all(on[k] == off[k] for k in keys), "Unmatched condition before first exposure")
    reg.require(on["opened_edges"] == 0 and bool(off["opened_edges"]) == (on["first"] is not None),
                "Exposure does not match intervention")
    def first_event(r):
        for i, e in enumerate(r["events"]):
            geom = reg.geometry(e["block"], len(r["prompt_ids"]), e["absolute_positions"], e["visible_kv"])
            if geom is not None:
                return i, geom, e["absolute_positions"]
        return None
    a, b = first_event(on), first_event(off)
    reg.require(a == b and (a is not None) == (on["first"] is not None),
                "First actual Q/K geometry differs")
    if a is not None:
        reg.require(on["events"][:a[0]] == off["events"][:a[0]], "Unequal pre-intervention events")
    if on["first"] is None:
        reg.require(all(on[k] == off[k] for k in ("events","output_ids")), "Unexposed pair changed")


def main():
    os.umask(0o077)
    cpus = sorted(os.sched_getaffinity(0))
    reg.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and cpus == list(range(40,48)),
                "Requires hidden CUDA and CPUs 40-47")
    OUT.mkdir(mode=0o700, exist_ok=False)
    (OUT / "tmp").mkdir()
    os.environ.update(TMPDIR=str(OUT / "tmp"), HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1")
    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    import transformers.utils as utils
    if not hasattr(utils, "LossKwargs"):
        from typing import TypedDict
        utils.LossKwargs = TypedDict("LossKwargs", {"num_items_in_batch": int}, total=False)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.evaluation.scoring import score
    from bdlm_serving.runtime.serial import install_llada_rectangular_mask
    from experiments.t2t.serving_slots.mask_probe_llada import verify_checkpoint
    from experiments.t2t.serving_slots.top_transition90 import resolve_reference, validate_prompts

    samples = reg.DATA/"samples.json"
    reg.require(reg.sha(samples) == reg.SAMPLES_SHA and reg.CAP == 2048, "Changed frozen protocol")
    tasks = [load_tasks(samples)[i] for i in INDICES]
    reference = resolve_reference(SimpleNamespace(samples=samples))
    verify_checkpoint(CKPT)
    opts = dict(trust_remote_code=True, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(CKPT, **opts)
    validate_prompts(tok, tasks, reference)
    reg.save(OUT/"run.json", dict(formal=False, device="cpu", cpus=cpus, indices=INDICES,
             arms=ARMS, offer=64, config=CFG, cap=reg.CAP, samples_sha256=reg.SAMPLES_SHA,
             source=reg.source_hashes() | {n: reg.sha(Path(__file__).with_name(n)) for n in
                 ("llada_spawn.py","llada_spawn_pilot.py","llada_quote.py")}))
    model = AutoModelForCausalLM.from_pretrained(CKPT, **opts, device_map={"":"cpu"},
                 dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    install_llada_rectangular_mask(model, allow_selected_cache=True)
    results = {name: [] for name in ARMS}
    for task in tasks:
        i = task["index"]
        prompt = reg.read(reference/f"{i:02d}-native32.json")["prompt_ids"]
        case = {}
        for arm in ARMS:
            with torch.inference_mode():
                row = generate(model, tok, task, prompt, arm)
            row["correct"] = bool(score(row["output_text"], task)["correct"])
            case[arm] = row
            reg.save(OUT/reg.filename(i, arm), row)
            admissions = Counter(e["tail_admission"] for e in row["events"] if "tail_admission" in e)
            results[arm].append(dict(index=i, N=row["output_tokens"], F=row["generation_forwards"],
                Q=row["generation_physical_padded_rows"], correct=row["correct"], tpf=row["tpf"],
                opened=admissions["open"], deferred=admissions["defer"], exposed=row["opportunities"]>0))
            print(i, arm, results[arm][-1], flush=True)
        for gate in ("eager","progress80"):
            compare(case[gate+"-causal"], case[gate+"-open"])
    out = {}
    for arm, rs in results.items():
        totals = {k:sum(r[k] for r in rs) for k in ("correct","N","F","Q","opened","deferred","exposed")}
        out[arm] = totals | dict(requests=len(rs), macro_tpf=mean(r["tpf"] for r in rs),
                                 token_efficiency=totals["N"]/totals["Q"])
    reg.save(OUT/"complete.json", dict(suite="cpu-pilot3", indices=INDICES, results=out,
                                      manifest_sha256=reg.sha(OUT/"run.json")))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["verify"]:
        sealed = reg.read(OUT/"complete.json")
        reg.require(sealed["indices"] == list(INDICES) and sealed["manifest_sha256"] == reg.sha(OUT/"run.json"),
                    "Unsealed pilot")
        for i in INDICES:
            for gate in ("eager", "progress80"):
                compare(*(reg.read(OUT/reg.filename(i, gate+"-"+m)) for m in ("causal", "open")))
        for arm in ARMS:
            rows = [reg.read(OUT/reg.filename(i,arm)) for i in INDICES]
            totals = sealed["results"][arm]
            gen = [[e for e in r["events"] if e["phase"] != "prefill"] for r in rows]
            qs = [sum(e["physical_q"] for e in events) for events in gen]
            ns = [r["output_tokens"] for r in rows]
            reg.require(all(r["policy"] == arm and r["index"] == i and
                            r["generation_forwards"] == len(events) and
                            r["generation_physical_padded_rows"] == q
                            for r,i,events,q in zip(rows, INDICES, gen, qs, strict=True)), "Invalid rows")
            reg.require((sum(ns), sum(map(len,gen)), sum(qs), sum(r["correct"] for r in rows),
                         mean(n/len(events) for n,events in zip(ns,gen,strict=True)), sum(ns)/sum(qs))
                        == tuple(totals[k] for k in ("N","F","Q","correct","macro_tpf","token_efficiency")),
                        "Aggregate does not match physical ledger")
        print("Verified six mask pairs: frozen indices, pre-intervention events, physical Q and scores")
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit("Expected no arguments or verify")
