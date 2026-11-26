"""Protected frozen90 DiffusionGemma B256/nominal Q512 active_w2 mask pair.

Run on n164 with the runtime venv: python -u mask_pair_gemma.py gemma 0
The operator starts it only after prefix90-wave-switch has finished.
"""

from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


MASK_ON = "active_w2_mask_on"
MASK_OFF = "active_w2_mask_off"
POLICIES = (MASK_ON, MASK_OFF)
REFERENCE = "gemma-fixed512-active-w2"
OUTPUT = "gemma-fixed512-mask-pair"
IDENTITY = ("index", "task_id", "seed", "dataset")
LAUNCHER_SHA = "ba80a293773326950738fd0d4ca0a8a04e6df4b86c6e9a7265bd2ba3a92e6e59"
WAVE_SHA = "d4179d8566fba62cf8f2a0b9891a497f05e8327f1145afc5a75da30f29e293e6"
SAMPLES_SHA = "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed"


def read_json(path):
    return json.loads(path.read_text())


def mask_for_arm(masks, cache, layer_types, block, q, *, opened):
    """Validate the native bool SDPA map; change only head-query/tail-key."""
    import torch

    kinds = set(layer_types)
    if not isinstance(masks, dict) or masks.keys() != kinds:
        raise ValueError("Unexpected native mask mapping")
    if q == block:
        if any(m is not None for m in masks.values()):
            raise ValueError("Single-canvas decode must retain native None masks")
        return masks
    if q != 2 * block:
        raise ValueError("Expected one or two physically active canvases")
    checks = []
    for kind, m in masks.items():
        cached = cache.layers[layer_types.index(kind)].keys.shape[-2]
        if (not isinstance(m, torch.Tensor) or m.dtype != torch.bool
                or m.shape != (1, 1, q, cached + q)):
            raise ValueError(f"{kind}: unexpected SDPA mask shape or dtype")
        checks.extend((m[..., :block, :-block].all(), ~m[..., :block, -block:].any(),
                       m[..., block:, :].all()))
    if not torch.stack(checks).all().item():
        raise ValueError("Native mask does not isolate head queries from tail keys")
    if not opened:
        return masks
    changed = {kind: m.clone() for kind, m in masks.items()}
    for m in changed.values():
        m[..., :block, -block:] = True
    return changed


def add_tensor(h, name, value):
    import torch

    if value is None:
        h.update(json.dumps([name, None]).encode())
        return
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name}: expected processor/model tensor")
    h.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
    h.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy())


def input_digest(data):
    h = hashlib.sha256()
    for key in sorted(data):
        add_tensor(h, key, data[key])
    return h.hexdigest()


def generation_settings(cfg):
    settings = cfg.to_dict()
    sampler = settings["sampler_config"]
    if hasattr(sampler, "to_dict"):
        settings["sampler_config"] = sampler.to_dict()
    return settings


@contextmanager
def decoder_probe(model, block, *, opened):
    """Request-local hooks; no global patch or mutation of the original masks."""
    import torch

    audit = dict(decoder_steps=0, dual_block_steps=0, first_dual_block_step=None,
                 pre_intervention_sha256=None)
    prefix = hashlib.sha256()
    layer_types = model.config.text_config.layer_types

    def encode(_, args, kw):
        if audit["first_dual_block_step"] is None:
            add_tensor(prefix, "encoder_ids", kw["input_ids"])
            add_tensor(prefix, "encoder_positions", kw["position_ids"])

    def decode(_, args, kw):
        ids = kw["decoder_input_ids"]
        q = ids.shape[-1]
        masks = kw["decoder_attention_mask"]
        changed = mask_for_arm(masks, kw["past_key_values"], layer_types, block, q,
                               opened=opened)
        step = audit["decoder_steps"]
        audit["decoder_steps"] += 1
        if audit["first_dual_block_step"] is None:
            add_tensor(prefix, "decoder_ids", ids)
            add_tensor(prefix, "decoder_positions", kw["decoder_position_ids"])
            if q == 2 * block:
                audit["first_dual_block_step"] = step
                add_tensor(prefix, "first_dual_self_conditioning", kw["self_conditioning_logits"])
                add_tensor(prefix, "cpu_rng", torch.get_rng_state())
                if ids.is_cuda:
                    add_tensor(prefix, "cuda_rng", torch.cuda.get_rng_state(ids.device))
        if q == 2 * block:
            audit["dual_block_steps"] += 1
        if changed is not masks:
            return args, dict(kw, decoder_attention_mask=changed)
        return None

    with ExitStack() as stack:
        for phase, hook in (("encoder", encode), ("decoder", decode)):
            stack.callback(getattr(model.model, phase).register_forward_pre_hook(
                hook, with_kwargs=True).remove)
        yield audit
    audit["pre_intervention_sha256"] = prefix.hexdigest()


def make_row(task, policy, ids, trace, events, prompt_len, tokenizer, eos, special, block):
    gen = events[1:]
    if (events[:1] != [("encoder", prompt_len)] or not trace
            or any(tag not in ("encoder", "decoder") or (tag == "encoder" and q != block)
                   for tag, q in gen)
            or [q for tag, q in gen if tag == "decoder"] != [t["q"] for t in trace]
            or any(t["width"] != t["active"] or t["width"] not in (1, 2)
                   or t["q"] != t["width"] * block for t in trace)):
        raise ValueError("Observed model work disagrees with active_w2 scheduler")
    n = sum(i not in special for i in ids)
    f, q = len(gen), sum(q for _, q in gen)
    return dict({k: task[k] for k in IDENTITY}, model="diffusiongemma", policy=policy,
                status="success", output_ids=ids,
                output_text=tokenizer.decode(ids, skip_special_tokens=True), output_tokens=n,
                generation_forwards=f, generation_query_rows=q,
                generation_physical_padded_rows=q, max_q=max(q for _, q in gen),
                tpf=n / f, token_efficiency=n / q, events=list(map(list, events)),
                trace=trace, eos=ids[-1] in eos, truncated=ids[-1] not in eos)


def equal_row(row, peer, label):
    normalized = dict(row, policy=peer["policy"])
    if normalized != peer:
        fields = sorted(key for key in normalized.keys() | peer.keys()
                        if key not in normalized or key not in peer or normalized[key] != peer[key])
        raise ValueError(f"{row['task_id']}: {label} mismatch in {fields}")


def check_pair(on, off, on_probe, off_probe, block):
    for row, probe in ((on, on_probe), (off, off_probe)):
        dual = [i for i, step in enumerate(row["trace"]) if step["q"] == 2 * block]
        if (probe["first_dual_block_step"] != (dual[0] if dual else None)
                or probe["dual_block_steps"] != len(dual)
                or probe["decoder_steps"] != len(row["trace"])):
            raise ValueError(f"{row['task_id']}: decoder probe disagrees with trace")
    first = on_probe["first_dual_block_step"]
    if (first != off_probe["first_dual_block_step"]
            or on_probe["pre_intervention_sha256"] != off_probe["pre_intervention_sha256"]):
        raise ValueError(f"{on['task_id']}: input/state differs before first intervention")
    if first is None:
        equal_row(off, on, "no-second-block output/work")
    else:
        def event_prefix(row):
            # Include prefill, committed re-encodes, and the first Q512 call.
            end = [i for i, (tag, _) in enumerate(row["events"]) if tag == "decoder"][first]
            return row["events"][:end + 1]

        if on["trace"][:first + 1] != off["trace"][:first + 1] or event_prefix(on) != event_prefix(off):
            raise ValueError(f"{on['task_id']}: work differs before first intervention")
    return dict(first_intervention_step=first, dual_block_steps_on=on_probe["dual_block_steps"],
                dual_block_steps_off=off_probe["dual_block_steps"],
                decoder_steps_on=on_probe["decoder_steps"], decoder_steps_off=off_probe["decoder_steps"],
                pre_intervention_sha256=on_probe["pre_intervention_sha256"],
                no_second_block_exact=first is None)


def require_complete(folder, policies):
    for policy in policies:
        if read_json(folder / f"complete-{policy}.json") != dict(policy=policy, requests=90):
            raise ValueError(f"Missing complete frozen90 {policy}: {folder}")


def active_settings(tier):
    return dict(tier, policies=["flow_w2", "active_w2"], nominal_q=512,
                qualification="replay packed W2 on all 90 before active_w2")


def require_switch_artifacts(data, active):
    from bdlm_serving.methods.t2t.wave_offer import TRACES

    ref = data / REFERENCE
    if read_json(ref / "run.json") != active:
        raise ValueError("Historical active_w2 settings differ from registered W2")
    require_complete(ref, active["policies"])
    switch = data / "gemma-external-256-512"
    expected = dict(active, policies=list(TRACES), offer_sequences={k: list(v) for k, v in TRACES.items()},
                    nominal_q=None, qualification="flat W1 and flat active W2 output, events and original wave trace")
    if read_json(switch / "run.json") != expected:
        raise ValueError("Switch predecessor settings differ from registered W2")
    require_complete(switch, expected["policies"])


def load_reference(folder, tasks, expected, sha):
    if read_json(folder / "run.json") != expected:
        raise ValueError("Historical active_w2 settings differ")
    require_complete(folder, expected["policies"])
    pinned = []
    for task in tasks:
        path = folder / f"{task['index']:02d}-active_w2.json"
        digest = sha(path)
        row = read_json(path)
        if (row["policy"] != "active_w2" or row["model"] != expected["model"]
                or row["status"] != "success" or any(row[k] != task[k] for k in IDENTITY)
                or sha(path) != digest):
            raise ValueError(f"Wrong reference row: {path}")
        pinned.append((row, digest))
    return pinned


def run_directional_probes(model, data, task, output, measure, save):
    import torch

    device = data["input_ids"].device
    for layout in ("packed", "per_canvas"):
        try:
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                report = measure(model, data, task, expert_layout=layout)
            signals = {(r["mask"], r["changed"], r["read"]): r["max_abs"] for r in report["results"]}
            leaks = signals["block_causal", "tail_first", "head"]
            opened = signals["bidirectional", "tail_first", "head"]
            if (leaks != 0 or not all(v > 0 for k, v in signals.items()
                                     if k != ("block_causal", "tail_first", "head"))
                    or any(d["same_input"]["max_abs"] != 0 for d in report["diagnostics"].values())):
                print(f"WARNING directional {layout}: masked tail->head max_abs={leaks}; "
                      f"opened={opened}; inspect diagnostic controls; continuing quality A/B", flush=True)
        except Exception as exc:
            # A diagnostic execution failure is recorded separately from generation.
            report = dict(status="error", expert_layout=layout, error_type=type(exc).__name__,
                          error=str(exc))
            print(f"WARNING directional {layout}: {exc!r}; continuing quality A/B", flush=True)
        save(output / f"directional-{layout}.json", report)


def run_pair(args, before_cuda):
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

    from bdlm_serving.evaluation.frozen import ROOT, sha
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.evaluation.original30 import save_json
    from bdlm_serving.runtime import diffusiongemma_wave as wave
    from bdlm_serving.runtime.diffusiongemma import check_snapshot, observe
    from experiments.t2t.serving_slots.mask_probe_gemma import measure

    if (before_cuda is None or args.suite != "frozen90" or args.block_size != 256
            or args.flowblock_windows != [1, 2] or not args.flow_only
            or sha(Path(wave.__file__)) != WAVE_SHA or sha(args.samples) != SAMPLES_SHA):
        raise ValueError("Not the reviewed frozen90 active_w2 runtime")
    plan = wave.resolve_plan(args.config, args.block_size, args.flowblock_windows)
    plan.update(include_b3=False, policies=["flow_w2", "active_w2"])
    native = read_json(ROOT / "archive/diffusiongemma-native/config.json")
    if (native["attention"] != "sdpa" or native["dtype"] != "bfloat16"
            or transformers.__version__ != native["transformers"]):
        raise ValueError("Unexpected native attention backend, dtype or Transformers version")
    tier = dict(plan, model=native["model"], revision=native["revision"],
                transformers=transformers.__version__, torch=str(torch.__version__),
                suite="frozen90", seed_offset=0, samples_sha256=SAMPLES_SHA,
                policies=["flow_w1", "flow_w2"])
    if read_json(args.samples.parent / "gemma-wave-reference90" / "run.json") != tier:
        raise ValueError("Runtime versions or settings differ from the archived W2 tier")
    expected = active_settings(tier)
    require_switch_artifacts(args.samples.parent, expected)
    ref = args.samples.parent / REFERENCE
    tasks = load_tasks(args.samples)
    historical = load_reference(ref, tasks, expected, sha)
    check_snapshot(args.checkpoint)
    proc = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.checkpoint, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation=native["attention"], experts_implementation=native["experts"],
        device_map={"": "cpu"}).eval()
    block = plan["block_size"]
    if model.config.canvas_length != block:
        raise ValueError("Checkpoint canvas differs")
    cfg, _ = model._prepare_generation_config(None, max_new_tokens=plan["cap"])
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    save_json(args.output / "run.json", dict(
        expected, policies=list(POLICIES), qualification="exact historical active_w2 replay and paired prefix",
        reference=str(ref), reference_run_sha256=sha(ref / "run.json"),
        reference_row_sha256=[digest for _, digest in historical], runtime_sha256=WAVE_SHA,
        checkpoint_manifest_sha256=sha(ROOT / "benchmarks/checkpoints/diffusiongemma.json"),
        wave_config_sha256=sha(args.config), native_config_sha256=sha(ROOT / "archive/diffusiongemma-native/config.json"),
        launcher_sha256=LAUNCHER_SHA, experiment_sha256=sha(Path(__file__)),
        generation_config=generation_settings(cfg), attention=native["attention"], experts=native["experts"],
        dtype=native["dtype"], decoder_step_indexing="zero-based, includes first Q256 step as step 0",
        treatment="only head query rows may see tail key columns in every two-active-canvas decoder call"))
    before_cuda(model)
    model.to("cuda")
    eos = set(model.generation_config.eos_token_id)
    special = set(proc.tokenizer.all_special_ids) | eos | {model.generation_config.pad_token_id}
    audit_rows = []

    def process(task):
        return proc.apply_chat_template(
            [{"role": "user", "content": task["prompt"]}], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        first_data = process(tasks[0])
        first_digest = input_digest(first_data)
        run_directional_probes(model, first_data, tasks[0], args.output, measure, save_json)
        del first_data
        for task, (peer, reference_sha) in zip(tasks, historical, strict=True):
            data = process(task)
            digest = input_digest(data)
            if task["index"] == 0 and digest != first_digest:
                raise ValueError("First processed input differs from directional diagnostic")
            rows, probes = [], []
            for policy in POLICIES:
                torch.manual_seed(task["seed"])
                with observe(model) as events, decoder_probe(model, block, opened=policy == MASK_OFF) as probe:
                    seq, trace = wave.generate(model, data, "active_w2", plan)
                if input_digest(data) != digest:
                    raise ValueError(f"{task['task_id']}: executor mutated the shared processor input")
                row = make_row(task, policy, seq, trace, events, data["input_ids"].shape[1],
                               proc.tokenizer, eos, special, block)
                if policy == MASK_ON:
                    equal_row(row, peer, "historical active_w2 replay")
                rows.append(row)
                probes.append(probe)
            audit = dict({k: task[k] for k in IDENTITY}, input_sha256=digest,
                         historical_row_sha256=reference_sha, masked_replay_exact=True,
                         **check_pair(*rows, *probes, block))
            for row in rows:
                save_json(args.output / f"{row['index']:02d}-{row['policy']}.json", row)
            audit_rows.append(audit)
            # Keep audit records out of the shared scorer's request-file discovery.
            save_json(args.output / "pair-audit.json", dict(cases=audit_rows))
            print(task["task_id"], f"first={audit['first_intervention_step']}",
                  f"dual_on={audit['dual_block_steps_on']} dual_off={audit['dual_block_steps_off']}",
                  flush=True)
    if not any(a["first_intervention_step"] is not None for a in audit_rows):
        raise ValueError("No request exercised the intervention")
    for policy in POLICIES:
        save_json(args.output / f"complete-{policy}.json", dict(policy=policy, requests=len(audit_rows)))
    save_json(args.output / "qualified.json",
              dict(requests=len(audit_rows), historical_replayed=len(audit_rows),
                   exposed=sum(a["first_intervention_step"] is not None for a in audit_rows),
                   no_second_block_exact=sum(a["no_second_block_exact"] for a in audit_rows)))
    # Standalone real-state diagnostic, after all scored requests and observers.
    from experiments.t2t.serving_slots.gemma_trigger_probe import probe
    case = tasks[75]
    try:
        diag = probe(model, process(case), case, plan, max_decoder_steps=7)
    except (ValueError, RuntimeError) as exc:
        diag = dict(index=75, status="diagnostic_failed", error=str(exc))
    save_json(args.output / "first-dual-diagnostic.json", diag)


def protected_source(launcher):
    src = launcher.read_bytes()
    call = b"run_wave(args, before_cuda=prepare)"
    if hashlib.sha256(src).hexdigest() != LAUNCHER_SHA or src.count(call) != 1:
        raise ValueError("Not the reviewed GPU0 protected launcher")
    replacement = (b"from experiments.t2t.serving_slots.mask_pair_gemma import run_pair\n"
                   b"        run_pair(args, before_cuda=prepare)")
    return compile(src.replace(call, replacement), str(launcher), "exec")


def require_switch_finished(data):
    live = subprocess.run(["tmux", "has-session", "-t", "=prefix90-wave-switch"],
                          capture_output=True, check=False)
    if live.returncode == 0:
        raise RuntimeError("prefix90-wave-switch is still active; not queuing or starting")
    if live.returncode != 1:
        raise RuntimeError(f"Could not verify predecessor session: {live.stderr!r}")
    tier = read_json(data / "gemma-wave-reference90" / "run.json")
    if (tier.get("suite") != "frozen90" or tier.get("samples_sha256") != SAMPLES_SHA
            or tier.get("policies") != ["flow_w1", "flow_w2"]):
        raise ValueError("Not the historical frozen90 tier")
    require_switch_artifacts(data, active_settings(tier))


def mark_protected_complete(output):
    from bdlm_serving.evaluation.original30 import save_json

    require_complete(output, POLICIES)
    meta = read_json(output / "qualified.json")
    if (meta["requests"] != 90 or meta["historical_replayed"] != 90
            or not 1 <= meta["exposed"] <= 90
            or meta["exposed"] + meta["no_second_block_exact"] != 90):
        raise ValueError("Missing qualified paired frozen90 output")
    save_json(output / "protected-complete.json",
              dict(status="complete", requests=90, gpu=0, holder_restored=True))


def main():
    if not __debug__ or sys.argv[1:] not in (["gemma", "0"], ["gemma", "0", "worker"]):
        raise ValueError("Use the protected Gemma GPU0 entry: gemma 0")
    here = Path(__file__).resolve()
    repo = here.parents[3]
    root = repo.parent / "t2t-native-1125"
    data = root / "prefix90"
    launcher = root / "launch_prefix90.py"
    code = protected_source(launcher)
    parent = sys.argv[1:] == ["gemma", "0"]
    if parent:
        # The protected launcher sets PYTHONPATH afresh for the worker.
        sys.path.insert(0, str(repo))
        require_switch_finished(data)
    else:
        signal_dir = Path(os.environ["DG_SIGNAL"]).resolve(strict=True)
        if signal_dir.parent != root or not signal_dir.name.startswith("control-0-"):
            raise ValueError("Worker requires a GPU0 staging directory from the protected launcher")
    os.environ.update(PREFIX_MODE="wave", FROZEN90_SAMPLES=str(data / "samples.json"),
                      TRACE_OUT=str(data / OUTPUT))
    exec(code, {"__file__": str(here), "__name__": "__main__"})
    if parent:
        mark_protected_complete(data / OUTPUT)


if __name__ == "__main__":
    main()
