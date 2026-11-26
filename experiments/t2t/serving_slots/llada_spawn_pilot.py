"""CPU BF16, fixed Q64: tail admission x cross-block mask, frozen indices 0/10/20.

From repo root, with PYTHONPATH=.:../fixed-baseline-deps and CUDA_VISIBLE_DEVICES='',
run `taskset -c 40-47 ../venv/bin/python -um experiments.t2t.serving_slots.llada_spawn_pilot`.
Private per-request artifacts and sealed aggregate are written only on the cluster.
Append original30 for all four arms on the original thirty; verify-original30 for readback.
new60 repeats all four arms only on indices 30..89. score-original30 / score-new60
verify each complete run and publish paired, CPU-only aggregates separately.
"""
from collections import Counter
import hashlib
import inspect
import os
from pathlib import Path
from shutil import copyfile
from statistics import mean
from types import SimpleNamespace

from experiments.t2t.serving_slots import llada_all_slots90 as reg
from experiments.t2t.serving_slots.llada_mask_pilot import CFG, CKPT, INDICES

OUT = reg.DATA / "llada21-q64-admission-mask-pilot3"
OUT30 = reg.DATA / "llada21-q64-admission-mask-cpu-original30-v1"
OUT60 = reg.DATA / "llada21-q64-admission-mask-cpu-new60-v1"
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
    s = k.create(torch.tensor([prompt], device=model.device, dtype=torch.long),
                 CFG, reg.CAP, task["seed"], mask, [eos])
    receipts, witnesses, prices = [], [], []
    first = None

    def swap(_module, args, kw):
        nonlocal first
        ids, bias = kw["position_ids"], kw["attention_mask"]
        pos, end = ids[0].tolist(), bias.shape[-1]
        native = ids[0][:, None] // 32 >= torch.arange(end, device=ids.device)[None, :] // 32
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
        keys = torch.arange(bias.shape[-1], device=pos.device)
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


def reused_rows():
    reg.require(reg.sha(OUT/"run.json") == "295cffb14ae3fa5bd7365ef0570c7f5ab6c6f4d92b5c685df68e8190f1d6757e"
                and reg.sha(OUT/"complete.json") == "2d001069c0d07676d4a947285461b5962e6b053dea2dd436b0260b3339e521ed",
                "Unknown pilot manifests")
    source = reg.read(OUT/"run.json")
    deps = reg.source_hashes() | {n: reg.sha(Path(__file__).with_name(n))
                                 for n in ("llada_spawn.py", "llada_quote.py")}
    # A later generator can still run the full cohort, but must not inherit
    # old pilot outputs as if they came from this implementation.
    if (not all(source["source"][n] == h for n, h in deps.items())
            or hashlib.sha256(inspect.getsource(generate).encode()).hexdigest() !=
            "8930ff55308a5a81073d05e80d78aa7845c08ba25ecd428e870b3ef7355f266a"):
        return {}
    names = sorted(reg.filename(i, a) for i in INDICES for a in ARMS)
    hashes = {name: reg.sha(OUT/name) for name in names}
    ledger = "".join(f"{h}  {name}\n" for name, h in hashes.items()).encode()
    reg.require(hashlib.sha256(ledger).hexdigest() ==
                "237d8b632df006d2f34bbf0e75237d8da43d94069ad0f097b4a5b5383ae4682e", "Pilot transcripts changed")
    return hashes


def main(expanded=False, new60=False):
    folder = OUT60 if new60 else OUT30 if expanded else OUT
    indices = tuple(range(30,90)) if new60 else tuple(range(30)) if expanded else INDICES
    if new60:
        reg.require(reg.sha(OUT30/"run.json") ==
                    "f125562240b437fbaa1f201fa2bb99c78a723438fb54d437e0481c0ad2c45f0c"
                    and reg.read(OUT30/"complete.json")["indices"] == list(range(30))
                    and hashlib.sha256(inspect.getsource(generate).encode()).hexdigest() ==
                    "19c44b59ebbb4676ca87617b7e883071eb1d41bb27dd02b4c760ad3a22f862b9",
                    "Expected sealed original30 and the device-explicit generator")
        old = reg.read(OUT30/"run.json")["source"]
        checkout = Path(__file__).resolve().parents[3]
        for name, digest in old.items():
            if name != "llada_spawn_pilot.py":
                path = Path(__file__).with_name(name)
                if not path.is_file():
                    path = checkout/"bdlm_serving/methods/t2t"/name
                reg.require(reg.sha(path) == digest, f"Changed inference dependency {name}")
    os.umask(0o077)
    cpus = sorted(os.sched_getaffinity(0))
    reg.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and cpus == list(range(40,48)),
                "Requires hidden CUDA and CPUs 40-47")
    folder.mkdir(mode=0o700, exist_ok=False)
    (folder / "tmp").mkdir()
    os.environ.update(TMPDIR=str(folder / "tmp"), HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
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
    frozen = load_tasks(samples)
    tasks = [frozen[i] for i in indices]
    reference = resolve_reference(SimpleNamespace(samples=samples))
    verify_checkpoint(CKPT)
    opts = dict(trust_remote_code=True, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(CKPT, **opts)
    validate_prompts(tok, tasks, reference)
    reuse = reused_rows() if expanded and not new60 else {}
    extra = (dict(reused_rows=reuse, checkpoint=str(CKPT), dtype="bfloat16", attention="sdpa",
             scorer_sha256=reg.sha(reg.REPO/"bdlm_serving/evaluation/scoring.py"),
             torch=str(torch.__version__), threads=torch.get_num_threads(), interop=torch.get_num_interop_threads())
             if expanded or new60 else {})
    reg.save(folder/"run.json", dict(formal=False, device="cpu", cpus=cpus, indices=indices, **extra,
             arms=ARMS, offer=64, config=CFG, cap=reg.CAP, samples_sha256=reg.SAMPLES_SHA,
             source=reg.source_hashes() | {n: reg.sha(Path(__file__).with_name(n)) for n in
                 ("llada_spawn.py","llada_spawn_pilot.py","llada_quote.py")}))
    model = AutoModelForCausalLM.from_pretrained(CKPT, **opts, device_map={"":"cpu"},
                 dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    install_llada_rectangular_mask(model, allow_selected_cache=True)
    if new60:
        # The shared GPU-capable generator now makes three device/dtype arguments
        # explicit. On CPU it must reproduce every original record field.
        task = frozen[0]
        prompt = reg.read(reference/reg.filename(0,"native32"))["prompt_ids"]
        for arm in ARMS:
            with torch.inference_mode():
                probe = generate(model,tok,task,prompt,arm)
            probe["correct"] = bool(score(probe["output_text"],task)["correct"])
            reg.require(reg.digest(probe) == reg.digest(reg.read(OUT30/reg.filename(0,arm))),
                        f"CPU sentinel does not reproduce {arm}")
        reg.save(folder/"control-replay.json",dict(index=0,matched=list(ARMS),scored=False))
        print("Original case 0 reproduced in all four arms; beginning new60",flush=True)
    results = {name: [] for name in ARMS}
    for task in tasks:
        i = task["index"]
        prompt = reg.read(reference/f"{i:02d}-native32.json")["prompt_ids"]
        case = {}
        for arm in ARMS:
            name = reg.filename(i, arm)
            if name in reuse:
                row = reg.read(OUT/name)
                reg.require((row["index"],row["task_id"],row["seed"],row["prompt_ids"],row["policy"])
                            == (i,task["task_id"],task["seed"],prompt,arm)
                            and tok.decode(row["output_ids"],skip_special_tokens=True) == row["output_text"]
                            and row["correct"] == bool(score(row["output_text"],task)["correct"]),
                            "Pilot row changed frozen identity, text or score")
                copyfile(OUT/name, folder/name)
                reg.require(reg.sha(folder/name) == reuse[name], "Copy differs from sealed pilot")
            else:
                with torch.inference_mode():
                    row = generate(model, tok, task, prompt, arm)
                row["correct"] = bool(score(row["output_text"], task)["correct"])
                reg.save(folder/name, row)
            case[arm] = row
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
    reg.save(folder/"complete.json", dict(suite="cpu-new60" if new60 else "cpu-original30" if expanded else "cpu-pilot3",
             indices=indices, results=out, manifest_sha256=reg.sha(folder/"run.json")))


def verify(folder, indices):
    sealed = reg.read(folder/"complete.json")
    reg.require(sealed["indices"] == list(indices) and sealed["manifest_sha256"] == reg.sha(folder/"run.json"),
                "Unsealed pilot")
    for i in indices:
        for gate in ("eager", "progress80"):
            compare(*(reg.read(folder/reg.filename(i, gate+"-"+m)) for m in ("causal", "open")))
    for arm in ARMS:
        rows = [reg.read(folder/reg.filename(i,arm)) for i in indices]
        totals = sealed["results"][arm]
        gen = [[e for e in r["events"] if e["phase"] != "prefill"] for r in rows]
        qs = [sum(e["physical_q"] for e in events) for events in gen]
        ns = [r["output_tokens"] for r in rows]
        reg.require(all(r["policy"] == arm and r["index"] == i and
                        r["generation_forwards"] == len(events) and
                        r["generation_physical_padded_rows"] == q
                        for r,i,events,q in zip(rows, indices, gen, qs, strict=True)), "Invalid rows")
        reg.require((sum(ns), sum(map(len,gen)), sum(qs), sum(r["correct"] for r in rows),
                     mean(n/len(events) for n,events in zip(ns,gen,strict=True)), sum(ns)/sum(qs))
                    == tuple(totals[k] for k in ("N","F","Q","correct","macro_tpf","token_efficiency")),
                    "Aggregate does not match physical ledger")
    print(f"Verified {2*len(indices)} mask pairs: indices, pre-intervention events, physical Q and scores")


def score_cohort(new60=False):
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.evaluation.scoring import score
    from experiments.t2t.serving_slots.mask_pair_report import efficiency_interval
    from tools.prefix_cohorts import _window_row
    from tools.t2t_results import table

    indices,folder,name = (range(30,90),OUT60,"new60") if new60 else (range(30),OUT30,"original30")
    frozen = load_tasks(reg.DATA/"samples.json")
    tasks = [frozen[i] for i in indices]
    m = reg.read(folder/"run.json")
    if new60:
        reg.require(reg.read(folder/"control-replay.json")==dict(index=0,matched=list(ARMS),scored=False),
                    "Missing four-arm control replay")
    reg.require(m["indices"] == list(indices) and m["device"] == "cpu" and not m["formal"]
                and m["samples_sha256"] == reg.SAMPLES_SHA and m["offer"] == 64
                and m["arms"] == {k:list(v) for k,v in ARMS.items()}
                and m["cap"] == reg.CAP and m["config"] == CFG and m["dtype"] == "bfloat16"
                and m["scorer_sha256"] == reg.sha(Path(inspect.getfile(score))),
                "Wrong protocol or changed scorer")
    verify(folder, indices)
    sides = {}
    for arm in ARMS:
        side = sides[arm] = {}
        for i, task in zip(indices,tasks,strict=True):
            r = reg.read(folder/reg.filename(i,arm))
            n, f, q = (r[k] for k in ("output_tokens","generation_forwards","generation_physical_padded_rows"))
            correct = bool(score(r["output_text"],task)["correct"])
            reg.require(r["status"] == "success" and correct == r["correct"] and
                        all(r[k] == task[k] for k in ("index","seed","task_id")) and
                        r["generation_query_rows"] == q and r["tpf"] == n/f and r["token_efficiency"] == n/q and
                        all(e["layer_physical_q"] == [e["physical_q"]]*reg.LAYERS for e in r["events"]),
                        "Frozen identity, score or layer work mismatch")
            side[i] = dict(index=i, dataset=task["dataset"], task_id=task["task_id"],
                           N=n, F=f, Q=q, physical_Q=q, tpf=n/f, correct=correct, failed=False,
                           truncated=r["truncated"], max_q=r["max_q"], exposed=bool(r["opportunities"]))
    out = []
    pairs = [(g+"-open",g+"-causal","mask") for g in ("eager","progress80")]
    pairs += [("eager-"+a,"progress80-"+a,"spawn") for a in ("open","causal")]
    for base, arm, factor in pairs:
        a, b = sides[base], sides[arm]
        for scope in ("all","gsm8k","math500","humaneval"):
            tasks_in_scope = [t for t in tasks if scope=="all" or t["dataset"] == scope]
            r = _window_row(tasks,a,b,len(indices),scope,"llada21",base,arm,64)
            lo,hi = efficiency_interval(tasks_in_scope,a,b)
            out.append(dict(device="cpu", factor=factor, **r, efficiency_ci_low=lo, efficiency_ci_high=hi,
                            baseline_exposed=sum(a[t["index"]]["exposed"] for t in tasks_in_scope),
                            arm_exposed=sum(b[t["index"]]["exposed"] for t in tasks_in_scope)))
    table(Path(__file__).with_name("controls")/f"llada21-q64-admission-mask-cpu-{name}.csv",out)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] in (["verify"], ["verify-original30"]):
        verify(OUT30, range(30)) if sys.argv[1] == "verify-original30" else verify(OUT, INDICES)
    elif sys.argv[1:] in (["score-original30"],["score-new60"]):
        score_cohort(new60=sys.argv[1]=="score-new60")
    elif sys.argv[1:] == ["new60"]:
        main(new60=True)
    elif sys.argv[1:] in ([], ["original30"]):
        main(expanded=bool(sys.argv[1:]))
    else:
        raise SystemExit("Expected original30, new60, verify, verify-original30, score-original30, score-new60, or no args")
