"""Native DiffusionGemma generation with observational, batch-one work counters."""
import hashlib
import json
import math
from contextlib import ExitStack, contextmanager

from ..evaluation.frozen import ROOT, load_tasks

STUDY = ROOT / "experiments/diffusiongemma-native"


def check_snapshot(folder):
    entries = json.loads((ROOT / "benchmarks/checkpoints/diffusiongemma.json").read_text())
    for entry in entries:
        path = folder / entry["name"]
        size = path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"Wrong size: {path.name}")
        h = hashlib.sha256() if "sha256" in entry else hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                h.update(block)
        if h.hexdigest() != entry.get("sha256", entry.get("git_blob_sha1")):
            raise ValueError(f"Wrong checkpoint bytes: {path.name}")


@contextmanager
def observe(model):
    events = []
    def hook(phase, key):
        def record(module, args, kwargs):
            b, q = kwargs[key].shape
            if b != 1:
                raise ValueError("Only batch one has unambiguous physical per-request accounting")
            events.append((phase, q))
        return record
    with ExitStack() as stack:
        for phase, key in (("encoder", "input_ids"), ("decoder", "decoder_input_ids")):
            mod = getattr(model.model, phase)
            handle = mod.register_forward_pre_hook(hook(phase, key), with_kwargs=True)
            stack.callback(handle.remove)
        yield events


def account(events, suffix, prompt_len, canvas, cap, eos, special, pad, native_tpf):
    if events[:1] != [("encoder", prompt_len)] or not 0 < len(suffix) <= cap or len(suffix) % canvas:
        raise ValueError("Unexpected prefill or output shape")
    steps = [0]
    for phase, rows in events[1:]:
        if rows != canvas or phase not in ("encoder", "decoder"):
            raise ValueError("Unexpected phase or physical Q")
        if phase == "encoder":
            if not steps[-1]:
                raise ValueError("Canvas advanced without denoising")
            steps.append(0)
        else:
            steps[-1] += 1
    if not steps[-1] or len(steps) * canvas != len(suffix):
        raise ValueError("Incomplete canvas trace")
    nonpad = sum(t != pad for t in suffix)
    if not math.isclose(native_tpf, nonpad / sum(steps), rel_tol=1e-6):
        raise ValueError("Native TPF and observed decoder calls disagree")
    stop = next((i for i, t in enumerate(suffix) if t in eos), None)
    if stop is None and len(suffix) != cap:
        raise ValueError("Neither EOS nor length cap")
    delivered = suffix if stop is None else suffix[:stop + 1]
    n = sum(t not in special and t not in eos and t != pad for t in delivered)
    r = len(steps) - 1
    f = sum(steps) + r
    q = sum(q for _, q in events[1:])
    return delivered, dict(output_tokens=n, generation_forwards=f, generation_query_rows=q,
        generation_physical_padded_rows=q, tpf=n / f, token_efficiency=n / q,
        max_q=canvas, prefill_rows=prompt_len, decoder_forwards=sum(steps),
        decoder_query_rows=sum(steps) * canvas, reencode_forwards=r, reencode_query_rows=r * canvas,
        decoder_steps_per_canvas=steps, native_nonpad=nonpad, native_tpf=native_tpf,
        eos=stop is not None, truncated=stop is None)


def run_native(args, before_cuda=None):
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

    plan = json.loads((STUDY / "config.json").read_text())
    if transformers.__version__ != plan["transformers"]:
        raise ValueError(f"This protocol requires transformers=={plan['transformers']}")
    tasks = load_tasks(args.samples)
    check_snapshot(args.checkpoint)
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    tok = processor.tokenizer
    inputs = [processor.apply_chat_template([{"role": "user", "content": t["prompt"]}],
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt") for t in tasks]
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(args.checkpoint, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation=plan["attention"],
        experts_implementation=plan["experts"]).eval()
    canvas = model.config.canvas_length
    cfg = model.generation_config
    if canvas != plan["canvas"] or cfg.max_denoising_steps != 48:
        raise ValueError("Unexpected native configuration")
    args.output.mkdir(exist_ok=False, parents=True)
    if before_cuda is not None:
        before_cuda(model)
    model.to("cuda")
    meta = dict(plan, torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                effective_generation_config=cfg.to_dict())
    (args.output / "run.json").write_text(json.dumps(meta, indent=2))
    policy, = plan["policies"]
    with torch.inference_mode():
        for task, data in zip(tasks, inputs, strict=True):
            data = data.to(model.device)
            torch.manual_seed(task["seed"])
            with observe(model) as events:
                out = model.generate(**data, max_new_tokens=plan["cap"])
            prompt = data["input_ids"]
            plen = prompt.shape[1]
            if not torch.equal(out.sequences[:, :plen], prompt):
                raise ValueError("Changed prompt")
            sequence, counts = account(events, out.sequences[0, plen:].tolist(), plen, canvas,
                plan["cap"], set(cfg.eos_token_id), set(tok.all_special_ids),
                cfg.pad_token_id, out.tokens_per_forward.item())
            row = {k: task[k] for k in ("index", "dataset", "task_id", "seed")}
            row.update(counts, model=plan["model"], policy=policy, status="success",
                       output_text=tok.decode(sequence, skip_special_tokens=True),
                       output_ids=sequence, events=events)
            path = args.output / f"{task['index']:02d}-{policy}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(row, ensure_ascii=False))
            tmp.replace(path)
            print(f"{task['index']:02d} {task['dataset']} N={counts['output_tokens']} "
                  f"F={counts['generation_forwards']} Q={counts['generation_query_rows']}", flush=True)
            del out
