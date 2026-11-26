"""Single-model n164 CPU qualification, not a score or a queue entry.

Run with the native Gemma Python and t2t-experiments on PYTHONPATH:
  CUDA_VISIBLE_DEVICES='' taskset -c 48-55 python -B gemma_request_audit.py
The mask pilot used 8 intra-op threads and 1 inter-op thread. Its archive
does not record affinity. 48-55 is a suggested separate n164 allocation.
Reads pinned prefix90 anchors; writes only one small JSON report to stdout.
"""
import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import astuple, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import signal
import time


REPO = Path("/data1/project_data/gx-erqw-20261123/runtime/t2t-experiments")
DATA = REPO.parent / "t2t-native-1125/prefix90"
CKPT = REPO.parent.parent / "models/diffusiongemma-cd40195"
ANCHORS = {
    "gemma-unified-paired-route-cpu0.json":
        "cc3669ee9a3076ea1595cc2b4cb8826c91854e0de7abdb4a3954da189d98a65b",
    "gemma-unified-mask-pilot-v1/20-replay_open.json":
        "6dfd6063da0f4cb72d9d8d7015ca6e8554301572d6ad06ca799c446fded78001",
    "gemma-unified-mask-pilot-v1/run.json":
        "1afa6f95e16daf38cf4659d39494fb1175c2f65fe04340936c5b2f74a9caf395",
}
SLOTS = (32, 64, 128, 256, 512)
SERIES = {0: (256, 512, 32, 64, 128, 256, 512, 32), 20: (512, 512, 128, 32, 64, 256, 512, 32)}


class AuditFailure(AssertionError):
    pass


def check(ok, label):
    if not ok:
        raise AuditFailure(label)  # Do not confuse a gate violation with an expected API error.


def digest(x):
    return sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_anchor(name):
    raw = (DATA / name).read_bytes()
    check(sha256(raw).hexdigest() == ANCHORS[name], "archive SHA: " + name)
    return json.loads(raw)


def post(e):
    return [("decoder", e["q"])] + [("encoder", 256) for _ in e["encoded"]]


@contextmanager
def forward_gate(model):
    """Reject extra work *before* the module runs, even if generate loops."""
    pending, enc = [], []

    def hook(role, key):
        def record(_module, _args, kw):
            ids = kw[key]
            check(pending and tuple(ids.shape) == (1, pending[0][1]) and role == pending[0][0],
                  "unexpected physical " + role)
            pending.pop(0)
            if role == "encoder":
                # Width alone could hide a cross-request prefix or position swap.
                enc.append(digest([ids.tolist(), kw["position_ids"].tolist()]))
        return record

    @contextmanager
    def allow(ledger):
        check(not pending, "overlapping physical calls")
        pending.extend(ledger)
        try:
            yield
            check(not pending, "missing physical call")
        finally:
            pending.clear()

    with ExitStack() as stack:
        for role, key in (("encoder", "input_ids"), ("decoder", "decoder_input_ids")):
            module = getattr(model.model, role)
            stack.callback(module.register_forward_pre_hook(hook(role, key), with_kwargs=True).remove)
        yield allow, enc


def guarded_caller(torch, model, kv):
    configs = [layer.self_attn.config for layer in model.model.decoder.layers]
    sc = model.model.decoder.self_conditioning
    pre_hooks = tuple(sc._forward_pre_hooks)
    registry = kv._attention_functions()
    check(kv._ALIAS not in registry, "wave attention active before audit")

    def call(label, fn, *args, **kwargs):
        # Different ambient RNG state at each hand-off; do not reseed the model.
        torch.rand(1)
        random.random()
        cpu, py = torch.get_rng_state(), random.getstate()
        modes = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        try:
            return fn(*args, **kwargs)
        finally:
            check(torch.equal(cpu, torch.get_rng_state()) and py == random.getstate(), label + ": caller RNG")
            check(modes == (torch.is_grad_enabled(), torch.is_inference_mode_enabled()), label + ": grad mode")
            check(kv._ALIAS not in registry and tuple(sc._forward_pre_hooks) == pre_hooks
                  and all(layer.self_attn.config is cfg and cfg._attn_implementation == "sdpa"
                          for layer, cfg in zip(model.model.decoder.layers, configs, strict=True)),
                  label + ": attention config/registry or self-conditioning hook leaked")
    return call


@dataclass
class Case:
    task: dict
    data: object
    plan: dict
    causal: bool
    origin: int
    prompt_hash: str
    ids: list
    trace: list
    obs: dict
    enc: list


def compare_serial(i, archive, ids, trace, obs, task, proc, model, plan, prompt_hash, suite):
    label = f"serial vs archive [{i}]"
    check((task["task_id"], task["seed"]) == (archive["task_id"], archive["seed"]), label + ": identity")
    if i == 0:
        check(digest(trace) == digest(archive["trace"]) and obs == archive["physical_work"], label + ": full trace/work")
        # This older archive uses the default JSON separators for output IDs.
        check(len(ids) == archive["returned_ids"] and
              sha256(json.dumps(ids).encode()).hexdigest() == archive["output_sha256"]["on"], label + ": output")
    else:
        row = suite.work_row(task, "replay_open", ids, trace, obs, proc.tokenizer,
                             model.generation_config, plan["cap"], prompt_hash)
        check(row == archive, label + ": compact row/output/pretrace/first geometry")


def menu(req):
    if req.done or req.tick is None:
        return req.done, None, None
    return req.done, astuple(req.tick), tuple((k, astuple(v)) for k, v in req.options.items())


def rejects(fn, exception, *args):
    try:
        fn(*args)
    except exception:
        return
    raise RuntimeError("request accepted an invalid lifecycle operation")


def exercise(model, cases, start, audit, allow, enc, call):
    at = dict.fromkeys(cases, 0)
    active = {}
    with audit.projectors(model) as (proj, positions), ExitStack() as cleanup:
        check(len(proj) == 85 and {layer for layer, _ in proj} == set(range(30)), "QKV topology")

        def quiet(label, fn, *args):
            before = (len(positions), [len(t) for t in proj.values()])
            with allow([]):
                value = call(label, fn, *args)
            check(before == (len(positions), [len(t) for t in proj.values()]), label + ": projected rows")
            return value

        def open_one(i, *, priced=True):
            c, n, p = cases[i], len(enc), len(positions)
            with allow([("encoder", c.origin)]):
                r = call(f"start {i}", start, model, c.data, c.plan, seed=c.task["seed"],
                         causal=c.causal, quote_costs=priced)
            cleanup.callback(r.close)
            check(enc[n:] == c.enc[:1] and len(positions) == p and all(len(t) == p for t in proj.values()),
                  f"start {i}: prompt encode only")
            return r

        def advance(i, r, n, *, priced=True):
            c = cases[i]
            e = c.trace[n]
            q = SERIES[i][n % len(SERIES[i])]
            label = f"request {i} tick {n}"
            check(not r.done and (r.tick.tick, r.tick.head) == (n, e["head"]), label + ": clock")
            opened = len(e["ages_before"]) == 2 and e["ages_before"][1] == 0
            dirty = tuple(map(len, e["dirty_before"]))
            resident = len(dirty) - int(opened)
            check((r.tick.resident, r.tick.mandatory) == (resident, dirty[:resident]), label + ": debt")
            check((q == 512 and r.tick.may_open) == opened, label + ": spawn")
            if priced:
                check(tuple(sorted(r.options)) == SLOTS, label + ": offer menu")
                opt = r.options[q]
                edge = c.origin + (e["head"] + 1) * 256
                actual = (q, 1, e["q"], sum(p < edge for p in e["selected"]),
                          sum(p >= edge for p in e["sampled"]), len(e["refresh_only"]),
                          e["mandatory_q"], len(e["prefix_selected"]), e["phase"], e["verify_reason"], opened)
                check((opt.offer, opt.decoder_f, opt.decoder_q, opt.head_q, opt.tail_sample_q,
                       opt.parked_dirty_q, opt.mandatory_q, opt.prefix_q, opt.phase, opt.verify_reason,
                       opt.opens_tail) == actual and len(e["encoded"]) <= opt.post_reencode_f_max
                      and 256 * len(e["encoded"]) <= opt.post_reencode_q_max, label + ": quote")
            else:
                check(r.options is None, label + ": unpriced menu")
            p, j = len(positions), len(enc)
            with allow(post(e)):
                event = call(label, r.advance, q)
            check(event == e, label + ": full returned event")
            check(positions[p:] == [e["selected"]] and all(t[p:] == [e["q"]] for t in proj.values()),
                  label + ": actual decoder positions/QKV")
            k = 1 + sum(len(s["encoded"]) for s in c.trace[:n])
            check(enc[j:] == c.enc[k:k + len(e["encoded"])], label + ": committed encoder input")
            check(r.done == (n + 1 == len(c.trace)), label + ": completion flag")

        a = active[0] = open_one(0)
        quiet("pending result", rejects, a.result, RuntimeError)
        before = quiet("read menu", menu, a)
        quiet("invalid offer", rejects, a.advance, ValueError, 63)
        check(before == quiet("reread menu", menu, a), "invalid offer changed pending decision")
        for n in (0, 1):  # Leave A with a real sampled tail, just before the downgrade.
            advance(0, a, n)
        at[0] = 2
        parked = quiet("pause A", menu, a)
        cancel = open_one(20, priced=False)
        advance(20, cancel, 0, priced=False)
        quiet("cancel B", cancel.close)
        quiet("cancelled advance", rejects, cancel.advance, RuntimeError, 512)
        quiet("cancelled result", rejects, cancel.result, RuntimeError)
        active[20] = open_one(20)
        check(parked == quiet("resume A", menu, a), "B cancellation/restart changed A")

        while active:
            for i in (20, 0, 0):  # Keep B live across A's commit and promotion verification.
                if i not in active:
                    continue
                r = active[i]
                peer = active.get(20 - i)
                saved = quiet("pause peer", menu, peer) if peer is not None else None
                advance(i, r, at[i])
                at[i] += 1
                if r.done:
                    ids, trace = quiet("result", r.result)
                    check(ids == cases[i].ids and trace == cases[i].trace, f"request {i}: IDs/final tape")
                    quiet("finished advance", rejects, r.advance, RuntimeError, 32)
                    quiet("close completed", r.close)
                    del active[i]
                if peer is not None:
                    check(saved == quiet("resume peer", menu, peer), "other request changed paused peer")
        check(at == {i: len(c.trace) for i, c in cases.items()}, "incomplete cooperative work")
        return len(positions)


def run(threads):
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from bdlm_serving.evaluation.frozen import ROOT, sha
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.runtime import diffusiongemma_wave as wave
    from bdlm_serving.runtime.diffusiongemma import STUDY, check_snapshot
    from experiments.t2t.serving_slots import gemma_unified as engine, gemma_unified_audit as audit
    from experiments.t2t.serving_slots import gemma_wave_kv as kv, unified90 as suite
    from experiments.t2t.serving_slots.mask_pair_gemma import input_digest

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    check(Path(ROOT).resolve() == REPO.resolve() and callable(getattr(engine, "start_request", None)),
          "requires n164 t2t-experiments with cooperative API installed")
    first, second, meta = map(read_anchor, ANCHORS)
    samples = DATA / "samples.json"
    native = json.loads((STUDY / "config.json").read_text())
    plan = wave.resolve_plan(ROOT / "archive/diffusiongemma-wave/config.json")
    manifest = ROOT / "benchmarks/checkpoints/diffusiongemma.json"
    check(meta["samples_sha256"] == sha(samples) and meta["native_config"] == native and meta["wave_config"] == plan
          and meta["torch"] == str(torch.__version__) and meta["transformers"] == transformers.__version__
          and meta["sources"][manifest.name] == sha(manifest)
          and meta["sources"]["gemma_unified.py"] == first["sources"]["gemma_unified.py"], "frozen provenance")
    tasks = load_tasks(samples)
    check(len(tasks) == 90 and [tasks[i]["seed"] for i in SERIES] == [17000, 17020]
          and tuple(suite.ARMS["replay_open"]) == (SERIES[20], False, {}), "task seeds/trace")
    check_snapshot(CKPT)
    proc = AutoProcessor.from_pretrained(CKPT, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        CKPT, local_files_only=True, dtype=torch.bfloat16, device_map={"": "cpu"},
        attn_implementation=native["attention"], experts_implementation=native["experts"]).eval()
    check(not torch.cuda.is_initialized() and all(p.device.type == "cpu" for p in model.parameters())
          and len(model.model.decoder.layers) == 30 and model.config.canvas_length == 256, "model topology/device")
    call = guarded_caller(torch, model, kv)
    cases = {}
    with forward_gate(model) as (allow, enc):
        for i, old in ((0, first), (20, second)):
            task, causal = tasks[i], i == 0
            cfg = (dict(plan, cap=512, flowblock_spawn_threshold=0.0) if causal
                   else dict(plan, prefix_slots={}, prefix_mode="zero"))
            data = proc.apply_chat_template([{"role": "user", "content": task["prompt"]}], tokenize=True,
                                            add_generation_prompt=True, return_tensors="pt", return_dict=True)
            origin, h = data["input_ids"].shape[1], input_digest(data)
            check(task["index"] == i and origin == (48 if causal else 145), "prompt identity")
            j = len(enc)
            with allow([("encoder", origin)] + [item for e in old["trace"] for item in post(e)]):
                ids, trace, obs = call(f"generate {i}", audit.measure, model, data, cfg, SERIES[i], task["seed"], causal)
            compare_serial(i, old, ids, trace, obs, task, proc, model, cfg, h, suite)
            check(input_digest(data) == h, f"generate {i}: modified input")
            cases[i] = Case(task, data, cfg, causal, origin, h, ids, trace, obs, enc[j:])
        a = cases[0]
        down = sum(x["offer"] == x["q"] == 512 and y["offer"] < 256 and y["q"] < x["q"]
                   and bool(y["refresh_only"]) for x, y in zip(a.trace, a.trace[1:]))
        check(set(a.obs["offered"]) == set(SLOTS) and down > 0 and a.obs["sampled_tail_steps"] > 0
              and any(e["promoted"] for e in a.trace) and any(e["verify_reason"] == "promotion" for e in a.trace)
              and a.prompt_hash != cases[20].prompt_hash, "missing real transitions or distinct inputs")
        count = exercise(model, cases, engine.start_request, audit, allow, enc, call)
    check(count == 1 + sum(len(c.trace) for c in cases.values())
          and all(input_digest(c.data) == c.prompt_hash for c in cases.values()), "final work/input conservation")
    files = [Path(m.__file__) for m in (engine, audit, kv, suite)] + [Path(engine.__file__).with_name("gemma_quote.py")]
    return dict(status="pass", scope="CPU real weights, request parity only", threads=torch.get_num_threads(),
                interop=torch.get_num_interop_threads(), affinity=sorted(os.sched_getaffinity(0)),
                torch=str(torch.__version__), transformers=transformers.__version__,
                model_loads=1, decoder_layers=30, projection_modules=85, archived_engine_sha256=first["sources"]["gemma_unified.py"],
                source_sha256={p.name: sha(p) for p in [*files, Path(__file__)]},
                archived_comparison={"0": "full trace + output hash + physical work",
                                     "20": "full IDs + compact trace + pretrace/geometry hashes"},
                trace_comparison="full serial event at each advance, full IDs and trace at result",
                actual_wide_to_parked_narrow=down, cooperative_decoder_calls_including_cancel=count,
                cancellation_overhead={"prefills": 1, "decoder_calls": 1}, caller_rng_restored=True,
                cases=[dict(index=i, seed=c.task["seed"], causal=c.causal, prefill_q=c.origin,
                            output_length=len(c.ids), output_ids_canonical_sha256=digest(c.ids), full_trace_sha256=digest(c.trace),
                            decoders=len(c.trace), reencodes=c.obs["reencodes"], generation_q=c.obs["Q"],
                            parked_refresh_rows=c.obs["parked_refresh_rows"]) for i, c in cases.items()])


def physical_core(cpu):
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    return tuple(int((topology / key).read_text()) for key in ("physical_package_id", "core_id"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, choices=range(1, 9), default=8)
    p.add_argument("--minutes", type=int, choices=range(1, 181), default=180, metavar="1..180")
    args = p.parse_args()
    check(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and hasattr(os, "sched_getaffinity"),
          "requires Linux, explicitly hidden CUDA, and the native Gemma environment")
    cpus = os.sched_getaffinity(0)
    check(len(cpus) == args.threads, "explicitly taskset the requested number of independent cores, e.g. 48-55")
    cores = {physical_core(cpu) for cpu in cpus}
    check(len(cores) == args.threads and cores.isdisjoint(map(physical_core, range(32, 48))),
          "affinity shares a core internally or with the reserved LLaDA CPU lanes (32-47)")
    os.environ.update(OMP_NUM_THREADS=str(len(cpus)), MKL_NUM_THREADS=str(len(cpus)), OPENBLAS_NUM_THREADS="1",
                      OMP_THREAD_LIMIT=str(len(cpus)), OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE",
                      TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1",
                      TRANSFORMERS_VERBOSITY="error", HF_ENABLE_PARALLEL_LOADING="false")
    signal.alarm(args.minutes * 60)
    t = time.monotonic()
    report = run(len(cpus))
    signal.alarm(0)
    print(json.dumps(dict(report, elapsed_s=round(time.monotonic() - t, 1)), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except AuditFailure as exc:
        raise SystemExit(json.dumps(dict(status="failed", check=str(exc)))) from None
