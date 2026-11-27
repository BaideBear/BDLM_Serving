"""Frozen90 real-checkpoint DiffusionGemma block-history runner."""
import argparse
from contextlib import contextmanager
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import platform
import runpy
import sys
import time

from bdlm_serving.evaluation.frozen import ROOT, sha

WIDTH = 256
BUDGETS = (32, 64, 128, 256, 512)
KERNEL_COMMIT = "11bf64c"
DISPATCH = "dispatch-v1"
GATE_INDICES = (0, 10, 20)
ORACLE = ROOT / "tests/check_history_gemma_first_step.py"
LAUNCHER_SHA = "ba80a293773326950738fd0d4ca0a8a04e6df4b86c6e9a7265bd2ba3a92e6e59"
SOURCE = (Path(__file__), ROOT / "bdlm_serving/methods/t2t/history_gemma.py",
          ROOT / "bdlm_serving/methods/t2t/history_attention.py")
# BF16 qualification thresholds, measured against same-input SDPA. The
# trajectory gate below is independent of these numeric tolerances.
TOLERANCE = {"attention": {"max_abs": 0.05, "rel_rmse": 0.015},
             "logits": {"max_abs": 2.0, "rel_rmse": 0.04}}


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def provenance(dispatch=False):
    return {str(path.relative_to(ROOT)): sha(path) for path in (*SOURCE, ORACLE) if dispatch or path != ORACLE}


@contextmanager
def per_example_rng(seed, device):
    import torch

    index = device.index if device.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[index]):
        torch.random.default_generator.manual_seed(seed)
        torch.cuda.default_generators[index].manual_seed(seed)
        yield


def quantiles(values):
    ordered = sorted(values)
    return {name: ordered[math.ceil(p * len(ordered)) - 1] if ordered else 0
            for name, p in (("p50", .50), ("p95", .95), ("max", 1))}


def request(model, processor, task, budget, cap, *, audit=False, dispatch=False):
    import torch
    from bdlm_serving.methods.t2t.history_gemma import attention_backend, history_gemma
    from bdlm_serving.runtime.diffusiongemma import account

    data = processor.apply_chat_template([{"role": "user", "content": task["prompt"]}],
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    prompt = data["input_ids"]
    plen = prompt.shape[1]
    torch.cuda.synchronize(model.device)
    start = time.perf_counter()
    with per_example_rng(task["seed"], model.device), history_gemma(
            model, budget, audit=audit, saturating_sdpa=dispatch) as ledger:
        out = model.generate(**data, max_new_tokens=cap)
    torch.cuda.synchronize(model.device)
    elapsed = (time.perf_counter() - start) * 1000
    if not torch.equal(out.sequences[:, :plen], prompt):
        raise RuntimeError("Native generation changed the frozen prompt")
    cfg, tok = model.generation_config, processor.tokenizer
    eos = set(cfg.eos_token_id if isinstance(cfg.eos_token_id, (tuple, list)) else [cfg.eos_token_id])
    ids, counts = account(ledger.events, out.sequences[0, plen:].tolist(), plen, WIDTH, cap,
                          eos, set(tok.all_special_ids), cfg.pad_token_id, out.tokens_per_forward.item())
    trace = ledger.trace
    if not all(r["q"] == WIDTH and (r["history_q"] <= (budget or WIDTH) or r["close"])
               for r in trace):
        raise RuntimeError("Invalid physical Q or unaccounted history overrun")
    row = {k: task[k] for k in ("index", "dataset", "task_id", "seed")}
    row.update(counts, status="success", model="diffusiongemma",
               policy=f"history{budget}" if budget is not None else "native256", nominal_history_q=budget,
               attention_backend=attention_backend(budget, WIDTH, dispatch),
               events=ledger.events, trace=trace, probes=ledger.probes, output_ids=ids,
               output_text=tok.decode(ids, skip_special_tokens=True), cap=cap,
               valid_output_rows_in_last_canvas=len(ids) - trace[-1]["block"],
               history_query_rows=sum(r["history_q"] for r in trace),
               history_layer_query_rows=sum(r["history_layer_q"] for r in trace),
               history_q_quantiles=quantiles([r["history_q"] for r in trace]),
               dirty_wait_quantiles=quantiles([d for r in trace for d in r["dirty_wait"]]),
               history_edges=sum(r["history_edges"] for r in trace),
               block_edges=sum(r["block_edges"] for r in trace),
               close_decoder_forwards=sum(r["close"] for r in trace),
               close_decoder_q=sum(r["q"] for r in trace if r["close"]),
               close_history_q=sum(r["history_q"] for r in trace if r["close"]),
               close_history_excess=sum(r["close_excess"] for r in trace),
               elapsed_ms=None if audit else elapsed,
               ms_per_token=elapsed / counts["output_tokens"] if counts["output_tokens"] and not audit else None)
    return row


def endpoint_case(native, full, double):
    fields = ("block", "clock", "canvas", "draft", "stop_candidate", "closed")

    def trajectory(row):
        return [tuple(step[k] for k in fields) for step in row["trace"]]

    first_difference = next((i for i, (a, b) in enumerate(zip_longest(
        trajectory(native), trajectory(full))) if a != b), None)
    peaks = {}
    validated = bool(full["probes"]) and len(full["probes"]) == len(double["probes"])
    for key, limits in TOLERANCE.items():
        values = [value for row in (full, double) for p in row["probes"]
                  for value in (p[key].values() if key == "attention" else [p[key]])]
        peaks[key] = {metric: max((v[metric] for v in values), default=None) for metric in limits}
        validated &= bool(values) and all(math.isfinite(v[metric]) and v[metric] <= limit
                                         for v in values for metric, limit in limits.items())
    endpoints_equal = all(full[k] == double[k] for k in
                          ("trace", "events", "output_ids", "decoder_steps_per_canvas"))
    full_work = all(r["q"] == r["history_q"] and not r["reused"] and not r["close_excess"]
                    for row in (native, full, double) for r in row["trace"])
    return dict(index=native["index"], numerical_pass=validated, numeric_peaks=peaks,
                saturated_endpoints_equal=endpoints_equal, full_work=full_work,
                native_trajectory_equal=first_difference is None, first_divergent_round=first_difference,
                native_output_equal=native["output_ids"] == full["output_ids"],
                native_and_split_forwards=[native["generation_forwards"], full["generation_forwards"]],
                eos=[row["eos"] for row in (native, full, double)],
                physical_tail=[row["valid_output_rows_in_last_canvas"] for row in (native, full, double)],
                versions=[[r["encoder_version"] for r in row["trace"] if r["invalidated"]]
                          for row in (native, full, double)])


def pilot_case(row):
    trace, budget = row["trace"], row["nominal_history_q"]
    first = [r for r in trace if r["invalidated"]]
    if not (first and all(r["first_read"] == r["history_q"] == budget and r["cold_before"] == WIDTH
                         and r["reused"] == 0 for r in first)
            and any(r["history_q"] < r["q"] for r in trace)
            and any(r["reused"] > 0 for r in trace)
            and all(not r["close_excess"] or r["close"] for r in trace)):
        raise RuntimeError("Sparse pilot did not demonstrate cold start, actual row savings and reuse")
    return dict(index=row["index"], budget=budget, versions=len(first), rounds=len(trace),
                history_q_quantiles=row["history_q_quantiles"], close_history_excess=row["close_history_excess"],
                edits_without_read=sum(r["edited_without_read"] for r in trace))


def dispatch_endpoint_case(native, full, double):
    report = endpoint_case(native, full, double)
    exact_replay = all(p["logits"]["max_abs"] == 0
                       for row in (full, double) for p in row["probes"])
    right_dispatch = (native["attention_backend"] == "native_sdpa_reference"
                      and all(row["attention_backend"] == "native_sdpa_saturating_dispatch"
                              for row in (full, double)))
    report.update(c256_c512_backend=full["attention_backend"], split_parity_claimed=False,
                  exact_same_input_decoder_replay=exact_replay,
                  pass_=all((right_dispatch, exact_replay, report["numerical_pass"], report["full_work"],
                             report["saturated_endpoints_equal"], report["native_output_equal"],
                             report["native_trajectory_equal"], native["events"] == full["events"])))
    return report


def audit_pilot(row):
    report = pilot_case(row)
    if row["attention_backend"] != "triton_block_history":
        raise RuntimeError("Sparse pilot did not use the Triton block-history backend")
    valid, accepted, dirty = set(), set(), {}
    for tick, r in enumerate(row["trace"]):
        if r["invalidated"]:
            valid, accepted, dirty = set(), set(), {}
        selected = set(r["selected"])
        eligible = valid | selected
        now = set(r["accepted_rows"])
        edit = set(r["edited_rows"])
        admit = set(r["admitted_rows"])
        revoke = set(r["revoked_rows"])
        waits = [tick - dirty[i] for i in sorted(selected & dirty.keys())]
        if (len(selected) != r["history_q"] or set(r["sampler_eligible"]) != eligible
                or not now <= eligible or admit != now - accepted or revoke != accepted - now
                or not edit <= now & accepted or r["dirty_wait"] != waits
                or r["first_read"] != len(selected - valid) or r["reread"] != len(selected & valid)
                or r["reused"] != len(valid - selected) or r["dirty_reuse"] != len((valid - selected) & dirty.keys())
                or r["edited_without_read"] != len(edit - selected)
                or r["admitted_without_read"] != len(admit - selected)
                or (r["closed"] and len(eligible) != WIDTH)):
            raise RuntimeError("Sparse pilot disagrees with the independent sampler and debt ledger")
        dirty = {i: age for i, age in dirty.items() if i not in selected}
        for i in admit | edit | revoke:
            dirty.setdefault(i, tick)
        if set(r["pending_dirty_rows"]) != dirty.keys():
            raise RuntimeError("Sparse pilot dropped or double-settled a dirty row")
        valid, accepted = eligible, now
    report.update(sampler_debt_audit=True, edits=sum(r["edited"] for r in row["trace"]),
                  admitted_without_read=sum(r["admitted_without_read"] for r in row["trace"]),
                  first_reads=sum(r["first_read"] for r in row["trace"]),
                  rereads=sum(r["reread"] for r in row["trace"]),
                  reused_rows=sum(r["reused"] for r in row["trace"]))
    return report


def preserved_split_failure(paths, samples_sha, hashes):
    failed = []
    covered = set()
    kernel = "bdlm_serving/methods/t2t/history_attention.py"
    for path in paths:
        row = read(path)
        if (row["samples_sha256"] != samples_sha or row["source_sha256"][kernel] != hashes[kernel]
                or row["tolerance"] != TOLERANCE or row["formal_pass"] is not False
                or row["pilot_pass"] is not False):
            raise ValueError("Expected an archived, failed split parity gate for the pinned input and kernel")
        for case in row["cases"]:
            if case["numerical_pass"] is False and case["native_trajectory_equal"] is False:
                covered.add(case["index"])
        failed.append(dict(path=str(path.resolve()), sha256=sha(path), formal_pass=False,
                           tolerance=row["tolerance"], cases=row["cases"]))
    if not set(GATE_INDICES) <= covered:
        raise ValueError("Legacy split failures must cover all three registered gate cases")
    return failed


def dispatch_gate(model, processor, tasks, one, meta, args):
    archived = preserved_split_failure(args.split_gate, meta["samples_sha256"], meta["source_sha256"])
    report = dict(meta, protocol=DISPATCH, split_parity_pass=False, legacy_split_failure=archived,
                  ready_for_cohort=False, pass_=False, endpoints=[], operator=[], pilots=[])
    path = args.output / "dispatch-gate.json"
    write(path, report)
    for index in GATE_INDICES:
        report["endpoints"].append(dispatch_endpoint_case(
            *(one(index, c, audit=True) for c in (None, WIDTH, 2 * WIDTH))))
        write(path, report)
    if not all(row["pass_"] for row in report["endpoints"]):
        raise RuntimeError("Native SDPA saturating dispatch did not reproduce the native reference")
    if not any(any(0 < n < WIDTH for n in row["physical_tail"]) for row in report["endpoints"]):
        raise RuntimeError("The gate did not exercise an EOS tail")
    oracle = runpy.run_path(str(ORACLE))["packed_control"]
    for index in GATE_INDICES:
        payload = oracle(model, processor, tasks[index])
        write(args.output / f"operator-{index:02d}.json", payload)
        report["operator"].append(dict(index=index, cases=len(payload["cases"]),
            worst_rel_rmse=max(e["rel_rmse"] for case in payload["cases"].values()
                                for r in case["rounds"] for e in r["errors"].values()),
            worst_lse_error=max(r["lse_max_abs"] for case in payload["cases"].values()
                                for r in case["rounds"])))
        write(path, report)
    for c in BUDGETS[:3]:
        trials = [audit_pilot(one(index, c, audit=True)) for index in GATE_INDICES]
        if not any(r["versions"] > 1 for r in trials) or not any(r["edits_without_read"] for r in trials):
            raise RuntimeError(f"C{c} pilot missed encoder invalidation or warm off-Q T2T")
        report["pilots"].extend(trials)
        write(path, report)
    report.update(ready_for_cohort=True, pass_=True)
    write(path, report)
    print("dispatch gate PASS; legacy split parity remains FAIL", flush=True)


def run(args, before_cuda=None):
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.evaluation.scoring import score
    from bdlm_serving.runtime.diffusiongemma import STUDY, check_snapshot
    import triton

    plan = read(STUDY / "config.json")
    if transformers.__version__ != plan["transformers"]:
        raise ValueError(f"Required transformers=={plan['transformers']}")
    tasks = load_tasks(args.samples)
    dispatch = args.mode == "dispatch-gate" or args.protocol == DISPATCH
    hashes = provenance(dispatch)
    samples_sha = sha(args.samples)
    software = dict(torch=str(torch.__version__), transformers=transformers.__version__, triton=triton.__version__)
    if args.mode == "cohort":
        qualification = read(args.gate)
        qualifies = (qualification.get("protocol") == DISPATCH and qualification.get("ready_for_cohort") is True
                     and qualification.get("split_parity_pass") is False
                     and qualification.get("software") == software
                     and qualification.get("revision") == plan["revision"]
                     and qualification.get("gpu") == torch.cuda.get_device_name(0)
                     if dispatch else qualification.get("formal_pass") and qualification.get("pilot_pass"))
        if not (qualifies and qualification["source_sha256"] == hashes
                and qualification["samples_sha256"] == samples_sha):
            raise ValueError("Fixed-budget work requires a passing gate and pilot for this exact implementation")
        if args.cohort == "added60":
            earlier = read(args.original_complete)
            if (earlier["cohort"] != "original30" or earlier["indices"] != list(range(30))
                    or earlier["budgets"] != args.budgets or earlier["source_sha256"] != hashes
                    or earlier.get("protocol") != args.protocol or earlier["samples_sha256"] != samples_sha):
                raise ValueError("Run the matching original30 before added60")
    args.output.mkdir(parents=True, exist_ok=False)
    print("Checking pinned checkpoint bytes", flush=True)
    check_snapshot(args.checkpoint)
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(args.checkpoint, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation=plan["attention"], experts_implementation=plan["experts"],
        device_map={"": "cpu" if before_cuda else "cuda:0"}).eval()
    if model.config.canvas_length != WIDTH or model.generation_config.max_denoising_steps != 48:
        raise ValueError("Unexpected pinned canvas or clock")
    if before_cuda:
        before_cuda(model)
    model.to("cuda:0")
    gpu = torch.cuda.get_device_properties(0)
    meta = dict(mode=args.mode, protocol=DISPATCH if dispatch else args.protocol,
                checkpoint=str(args.checkpoint), revision=plan["revision"],
                source_sha256=hashes, samples_sha256=samples_sha, kernel_commit=KERNEL_COMMIT,
                host=platform.node(), gpu=gpu.name, visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                torch=str(torch.__version__), transformers=transformers.__version__,
                software=software, batch_size=1, canvas=WIDTH, cap=args.cap)
    if args.mode == "cohort":
        meta["gate_sha256"] = sha(args.gate)
    write(args.output / "run.json", meta)

    def one(index, budget, *, audit=False):
        row = request(model, processor, tasks[index], budget, args.cap, audit=audit, dispatch=dispatch)
        row["protocol"] = meta["protocol"]
        if args.mode == "cohort":
            row["score"] = score(row["output_text"], tasks[index])
        write(args.output / f"{index:02d}-{row['policy']}.json", row)
        print(f"case={index:02d} C={budget} N={row['output_tokens']} F={row['generation_forwards']} "
              f"Q={row['generation_query_rows']} histQ={row['history_query_rows']}", flush=True)
        return row

    with torch.inference_mode():
        if args.mode == "dispatch-gate":
            return dispatch_gate(model, processor, tasks, one, meta, args)
        if args.mode == "gate":
            cases = [endpoint_case(*(one(index, c, audit=True) for c in (None, WIDTH, 2 * WIDTH)))
                     for index in args.gate_indices]
            safe = all(r["numerical_pass"] and r["saturated_endpoints_equal"] and r["full_work"] for r in cases)
            formal = safe and all(r["native_trajectory_equal"] and r["native_output_equal"] for r in cases)
            report = dict(meta, tolerance=TOLERANCE, cases=cases, safe_for_pilot=safe, formal_pass=formal,
                          pilot_pass=False, pilot=[])
            write(args.output / "gate.json", report)
            if safe:
                for c in (32, 64, 128):
                    report["pilot"].append(pilot_case(one(args.gate_indices[0], c)))
                report["pilot_pass"] = True
                write(args.output / "gate.json", report)
            print(f"gate: safe_for_pilot={safe} formal_pass={formal} pilot_pass={report['pilot_pass']}", flush=True)
            return
        indices = list(range(30)) if args.cohort == "original30" else list(range(30, 90))
        # Validate each request before publishing it. Do not treat C512 as an
        # independent mechanism; C256/C512 are paired saturated endpoints.
        for index in indices:
            for budget in args.budgets:
                one(index, budget)
        write(args.output / "complete.json", dict(meta, indices=indices, cohort=args.cohort, budgets=args.budgets))


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gate", "dispatch-gate", "cohort"), default="gate")
    parser.add_argument("--protocol", choices=("split-v1", DISPATCH), default="split-v1")
    parser.add_argument("--split-gate", action="append", type=Path, default=[])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cap", type=int, choices=(256, 512, 1024, 2048))
    parser.add_argument("--gate-indices", nargs="+", type=int, default=[0, 10, 20])
    parser.add_argument("--budgets", nargs="+", type=int, choices=BUDGETS, default=list(BUDGETS))
    parser.add_argument("--cohort", choices=("original30", "added60"), default="original30")
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--original-complete", type=Path)
    args = parser.parse_args(argv)
    if args.cap is None:
        args.cap = 2048 if args.mode == "cohort" else 512
    if (args.mode == "cohort" and (args.cap != 2048 or args.gate is None
            or (args.cohort == "added60" and args.original_complete is None))
            or len(set(args.budgets)) != len(args.budgets)
            or any(not 0 <= index < 30 for index in args.gate_indices)
            or len(set(args.gate_indices)) != len(args.gate_indices)
            or (args.mode == "gate" and args.protocol != "split-v1")
            or (args.mode == "dispatch-gate" and (args.cap != 512 or not args.split_gate
                                                 or tuple(args.gate_indices) != GATE_INDICES))):
        parser.error("Invalid frozen cohort, gate, or fixed budget selection")
    return args


def protected():
    """Reuse the pinned launcher verbatim except for repo and runner entry."""
    runtime = ROOT.parent.parent
    launcher = runtime / "t2t-native-1125/launch_prefix90.py"
    if (ROOT != runtime / "block-history-1127/gemma-repo" or sha(launcher) != LAUNCHER_SHA
            or sys.argv[1:] not in (["gemma", "0"], ["gemma", "0", "worker"])):
        raise RuntimeError("Protected entry requires the reviewed n164 GPU0 launcher and dedicated repo")
    args = parse(json.loads(os.environ["GEMMA_HISTORY_ARGS"]))
    os.environ.update(PREFIX_MODE="optimus", FROZEN90_SAMPLES=str(args.samples), TRACE_OUT=str(args.output))
    source = launcher.read_text()
    edits = {
        'REPO = RUNTIME / "t2t-experiments"': 'REPO = RUNTIME / "block-history-1127/gemma-repo"',
        'from bdlm_serving.runtime.diffusiongemma_optimus import run':
            'from tools.run_history_gemma import protected_worker as run',
        '["bdlm_serving.runtime.diffusiongemma_optimus"]': '["tools.run_history_gemma"]',
    }
    for old, new in edits.items():
        if source.count(old) != 1:
            raise RuntimeError("Reviewed launcher integration points changed")
        source = source.replace(old, new)
    # __file__ still points to this module so the original launcher respawns
    # the same adapter as its own worker. Holder logic and flags are unchanged.
    exec(compile(source, str(launcher), "exec"), {"__file__": __file__, "__name__": "__main__"})


def protected_worker(launch_args, before_cuda):
    args = parse(json.loads(os.environ["GEMMA_HISTORY_ARGS"]))
    if args.checkpoint not in (None, launch_args.checkpoint):
        raise ValueError("Protected runner only uses the launcher's pinned checkpoint")
    args.checkpoint = launch_args.checkpoint
    run(args, before_cuda)


if __name__ == "__main__":
    if sys.argv[1:3] == ["gemma", "0"]:
        protected()
    else:
        args = parse(sys.argv[1:])
        if args.checkpoint is None or not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise SystemExit("Direct mode requires a checkpoint and a verified, explicitly pinned idle GPU")
        run(args)
