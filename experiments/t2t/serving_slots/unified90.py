"""Frozen Gemma five-slot request evaluation. GPU0 is acquired by the protected launcher."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from bdlm_serving.evaluation.frozen import ROOT, sha
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.evaluation.original30 import save_json
from experiments.t2t.serving_slots import gemma_fixed_caps as caps

HERE = Path(__file__).resolve()
SLOTS = (32, 64, 128, 256, 512)
# An external workload replay, not a budget controller inside the request.
OFFERS = (512, 512, 128, 32, 64, 256, 512, 32)
ARMS = {f"q{b}": ((b,), True, {}) for b in SLOTS}
ARMS.update(q128_slack_random=((128,), True, {128: 32}),
            replay=(OFFERS, True, {}), replay_open=(OFFERS, False, {}),
            q512_open=((512,), False, {}))
GROUPS = (("replay", "replay_open"), ("q512", "q512_open"), ("q32",),
          ("q64",), ("q128", "q128_slack_random"), ("q256",))
KEYS = ("index", "task_id", "dataset", "seed")


def read(path):
    return json.loads(path.read_text())


def work_row(task, policy, ids, trace, obs, tok, cfg, cap, prompt_digest):
    from hashlib import sha256
    eos = cfg.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    special = set(tok.all_special_ids) | eos | {cfg.pad_token_id}
    if (not ids or len(ids) > cap or any(t in eos for t in ids[:-1])
            or (ids[-1] not in eos and len(ids) != cap)):
        raise ValueError("Returned sequence violates EOS/cap convention")
    prefill = obs["prefill"]
    series, _, _ = ARMS[policy]
    boundary = lambda e: prefill + 256 * (1 + e["head"])
    def opportunity(e):
        return any(p < boundary(e) for p in e["selected"]) and any(p >= boundary(e) for p in e["visible"])
    def sig(obj):
        return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    first = next((i for i, e in enumerate(trace) if opportunity(e)), None)
    ns = len(trace) if first is None else first
    compact = [{k: e[k] for k in ("tick", "offer", "q", "head", "phase", "mandatory_q",
                                   "overflow", "encoded", "promoted")} |
               dict(prefix=len(e["prefix_selected"]), parked=len(e["refresh_only"]),
                    strict_edits=sum(map(len, e["strict_edits"])), opportunity=opportunity(e))
               for e in trace]
    if (not obs["all_projected_widths_match"] or not trace or
            any(e["offer"] != series[i % len(series)] for i, e in enumerate(trace)) or
            obs["actual_decoder_q"] != [e["q"] for e in trace] or
            obs["F"] != len(trace) + obs["reencodes"] or
            obs["Q"] != sum(obs["actual_decoder_q"]) + 256 * obs["reencodes"]):
        raise ValueError("Physical work disagrees with request")
    n = sum(t not in special for t in ids)
    geom = None if first is None else {k: trace[first][k] for k in
        ("tick", "head", "offer", "visible", "selected", "sampled", "dirty_before", "ages_before")}
    return dict({k: task[k] for k in KEYS}, model="diffusiongemma", policy=policy,
                input_sha256=prompt_digest, status="success", output_ids=ids,
                output_text=tok.decode(ids, skip_special_tokens=True), output_tokens=n,
                generation_forwards=obs["F"], generation_physical_padded_rows=obs["Q"],
                decoder_q=sum(obs["actual_decoder_q"]), encoder_q=256 * obs["reencodes"],
                tpf=n/obs["F"], token_efficiency=n/obs["Q"], truncated=ids[-1] not in eos,
                first_opportunity=first, pretrace_sha256=sig(trace[:ns]), first_geometry_sha256=sig(geom),
                prefill_q=prefill, projection_modules=obs["projection_modules"],
                decoder_layers=obs["decoder_layers"], trace=compact)


def paired(on, off):
    a, b = ARMS[on["policy"]], ARMS[off["policy"]]
    if not (a[0] == b[0] and a[1] and not b[1] and a[2] == b[2]):
        raise ValueError("Not a mask-only intervention")
    for k in (*KEYS, "input_sha256", "first_opportunity", "pretrace_sha256", "first_geometry_sha256"):
        if on[k] != off[k]:
            raise ValueError(f"Different pre-intervention input: {k}")
    if on["first_opportunity"] is None and any(on[k] != off[k] for k in
            ("output_ids", "trace", "output_tokens", "generation_forwards", "generation_physical_padded_rows")):
        raise ValueError("Mask affected an unexposed request")


def run(args, before_cuda=None):
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from bdlm_serving.runtime import diffusiongemma_wave as wave
    from bdlm_serving.runtime.diffusiongemma import STUDY, check_snapshot
    from experiments.t2t.serving_slots.gemma_unified_audit import measure
    from experiments.t2t.serving_slots.mask_pair_gemma import input_digest

    cpu = before_cuda is None
    if cpu and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU pilot requires hidden GPUs")
    tasks = load_tasks(args.samples)
    native = read(STUDY / "config.json")
    plan = wave.resolve_plan(ROOT / "archive/diffusiongemma-wave/config.json")
    if (sha(args.samples) != caps.SAMPLES_SHA or plan["flowblock_spawn_threshold"] != 0.8
            or plan["cap"] != 2048 or transformers.__version__ != native["transformers"]):
        raise ValueError("Nonregistered suite or runtime")
    group = GROUPS[0] if cpu else GROUPS[int(os.environ["UNIFIED_GROUP"])]
    if cpu:
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        tasks = [tasks[i] for i in (0, 10, 20)]
    check_snapshot(args.checkpoint)
    proc = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.checkpoint, local_files_only=True, dtype=torch.bfloat16, device_map={"": "cpu"},
        attn_implementation=native["attention"], experts_implementation=native["experts"]).eval()
    if model.config.canvas_length != 256:
        raise ValueError("Wrong canvas")
    source_paths = [HERE, HERE.with_name("gemma_unified.py"), HERE.with_name("gemma_unified_audit.py"),
                    HERE.with_name("gemma_wave_kv.py"), ROOT / "benchmarks/checkpoints/diffusiongemma.json"]
    args.output.mkdir(mode=0o700, exist_ok=False)
    save_json(args.output / "run.json", dict(
        model=native["model"], revision=native["revision"], torch=str(torch.__version__),
        transformers=transformers.__version__, suite="cpu-pilot" if cpu else "frozen90",
        device="cpu" if cpu else "cuda:0", samples_sha256=sha(args.samples),
        sources={p.name: sha(p) for p in source_paths}, launcher_sha256=caps.LAUNCHER_SHA,
        wave_config=plan, native_config=native, arms={k: ARMS[k] for k in group},
        rng="independent streams seeded by task.seed + canvas ordinal",
        indices=[t["index"] for t in tasks]))
    if not cpu:
        before_cuda(model)
        model.to("cuda")
    with torch.inference_mode():
        for task in tasks:
            data = proc.apply_chat_template([{"role": "user", "content": task["prompt"]}],
                tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
            digest = input_digest(data)
            rows = []
            for arm in group:
                series, causal, prefix = ARMS[arm]
                config = dict(plan, prefix_slots=prefix, prefix_mode="random" if prefix else "zero")
                ids, trace, obs = measure(model, data, config, series, task["seed"], causal)
                row = work_row(task, arm, ids, trace, obs, proc.tokenizer, model.generation_config,
                               plan["cap"], digest)
                if digest != input_digest(data):
                    raise ValueError("Shared input changed")
                rows.append(row)
            if group[-1].endswith("_open"):
                paired(*rows)
            for row in rows:
                save_json(args.output / f"{task['index']:02d}-{row['policy']}.json", row)
            print(task["task_id"], [(r["policy"], r["first_opportunity"],
                  r["generation_forwards"], r["generation_physical_padded_rows"]) for r in rows], flush=True)
    for arm in group:
        save_json(args.output / f"complete-{arm}.json", dict(policy=arm, requests=len(tasks)))


def score(folder):
    from bdlm_serving.evaluation.scoring import score as judge
    tasks = load_tasks(caps.SAMPLES)
    meta = read(folder / "run.json")
    pilot = meta["suite"] == "cpu-pilot"
    indices = [0, 10, 20] if pilot else list(range(90))
    if meta["indices"] != indices or meta["samples_sha256"] != caps.SAMPLES_SHA:
        raise ValueError("Incomplete cohort")
    result = []
    for arm in meta["arms"]:
        if read(folder / f"complete-{arm}.json") != dict(policy=arm, requests=len(indices)):
            raise ValueError("Incomplete arm")
        rows = []
        for i in indices:
            task = tasks[i]
            row = read(folder / f"{i:02d}-{arm}.json")
            if row["policy"] != arm or not all(row[k] == task[k] for k in KEYS):
                raise ValueError("Wrong case")
            n, f, q = row["output_tokens"], row["generation_forwards"], row["generation_physical_padded_rows"]
            if q != row["encoder_q"] + row["decoder_q"] or row["tpf"] != n/f or row["token_efficiency"] != n/q:
                raise ValueError("Inconsistent stored metrics")
            rows.append(dict(row, correct=bool(judge(row["output_text"], task)["correct"])))
        cohorts = {"pilot3": rows} if pilot else {"original30": rows[:30], "new60": rows[30:], "full90": rows}
        for name, subset in cohorts.items():
            for scope in ("all", "exposed", "unexposed"):
                values = [r for r in subset if scope == "all" or
                          (r["first_opportunity"] is not None) == (scope == "exposed")]
                n = sum(r["output_tokens"] for r in values)
                f = sum(r["generation_forwards"] for r in values)
                q = sum(r["generation_physical_padded_rows"] for r in values)
                result.append(dict(arm=arm, cohort=name, scope=scope, cases=len(values),
                    correct=sum(r["correct"] for r in values), N=n, F=f, physical_Q=q,
                    macro_TPF=sum(r["tpf"] for r in values)/len(values) if values else None,
                    token_efficiency=n/q if q else None))
    with (folder / "summary.csv").open("w") as out:
        writer = csv.DictWriter(out, fieldnames=result[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)
    return result


def output(i):
    return caps.DATA / ("gemma-unified90-" + "-".join(GROUPS[i]))


def protected():
    i = int(os.environ["UNIFIED_GROUP"])
    if not 0 <= i < len(GROUPS) or sys.argv[1:] not in (["gemma", "0"], ["gemma", "0", "worker"]):
        raise ValueError("Not registered GPU0 worker")
    if i == 0:
        receipt = read(caps.DATA / caps.OUTPUT / "protected-complete.json")
        if receipt != dict(status="complete", model="gemma", policies=list(caps.POLICIES), requests=90,
                           gpu=0, holder_restored=True, samples_sha256=caps.SAMPLES_SHA, run=caps.OUTPUT):
            raise ValueError("Missing predecessor receipt")
    elif read(output(i-1) / "returned.json") != {"gpu": 0, "restored": True}:
        raise ValueError("Missing preceding group")
    call = b"run_wave(args, before_cuda=prepare)"
    src = caps.LAUNCHER.read_bytes()
    if sha(caps.LAUNCHER) != caps.LAUNCHER_SHA or src.count(call) != 1:
        raise ValueError("Protected launcher changed")
    os.environ.update(PREFIX_MODE="wave", TRACE_OUT=str(output(i)), FROZEN90_SAMPLES=str(caps.SAMPLES))
    exec(compile(src.replace(call, b"from experiments.t2t.serving_slots.unified90 import run\n"
                                 b"        run(args, before_cuda=prepare)"), str(caps.LAUNCHER), "exec"),
         {"__file__": str(HERE), "__name__": "__main__"})
    # Only the parent returns here after verified holder restoration.
    for arm in GROUPS[i]:
        if read(output(i) / f"complete-{arm}.json") != dict(policy=arm, requests=90):
            raise ValueError("Incomplete GPU sweep")
    save_json(output(i) / "returned.json", {"gpu": 0, "restored": True})


def queue():
    if any(output(i).exists() for i in range(len(GROUPS))):
        raise ValueError("Refusing to overwrite an existing registered run")
    while True:
        check = subprocess.run(["tmux", "has-session", "-t", "=prefix90-gemma-fixed-caps"], capture_output=True)
        if check.returncode == 1:
            break
        check.check_returncode()
        time.sleep(10)
    for i in range(len(GROUPS)):
        with output(i).with_suffix(".log").open("x") as log:
            subprocess.run([sys.executable, "-u", str(HERE), "gemma", "0"],
                env=dict(os.environ, UNIFIED_GROUP=str(i)), stdout=log, stderr=subprocess.STDOUT, check=True)
        score(output(i))


if __name__ == "__main__":
    if sys.argv[1:] == ["queue"]:
        queue()
    elif sys.argv[1:2] == ["score"]:
        score(Path(sys.argv[2]))
    elif sys.argv[1:2] == ["cpu-pilot"]:
        from argparse import Namespace
        folder = Path(sys.argv[2])
        run(Namespace(samples=caps.SAMPLES, output=folder,
                      checkpoint=ROOT.parent.parent / "models/diffusiongemma-cd40195"))
        score(folder)
    else:
        protected()
