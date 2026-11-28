"""Reusable GPU-only LLaDA shard runtime: ``load`` once, ``evaluate`` one task.

The worker owns model loading and one frozen-task evaluation.  It deliberately
does **not** manage holders, boards or GPU reservation: an external launcher does
that.  Inference is CUDA-only — a CPU host is refused outright.

``evaluate`` maps the frozen ``evaluate_fixed`` contract through the unified row
planner (``unified_llada.factory``) and returns a small, self-describing record:

    status        "complete" | "valid_algorithm_failure"
    correct, N, F, Q, decoder_F, encoder_F, prefill_Q, offer, roles

``roles`` is the disjoint split of the billed decoder Q into
``dirty_write / uninitialized_mask / resident_mask / clean_resolved``; the four
parts are asserted to sum to Q.  ``valid_algorithm_failure`` is exactly the
documented nonconvergent request (no head retirement), which delivers no output;
any other failure is raised.
"""
import hashlib
import json
import os
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from bdlm_serving.evaluation import scoring
from bdlm_serving.methods.t2t.budget_family import Law
from . import unified_llada
from .llada_fixed_family_eval import CAP, SCORER_SHA, TIERS, evaluate_fixed, require

NONCONVERGENT = "No head retirement within 258 paid calls; not a valid completion"
STATUS_COMPLETE = "complete"
STATUS_ALGORITHM_FAILURE = "valid_algorithm_failure"
ROLES = ("dirty_write", "uninitialized_mask", "resident_mask", "clean_resolved")
# Keys this worker reads from the frozen _fixed row (see llada_fixed_family_eval).
REQUIRED = ("correct", "failure", "N", "F", "Q", "decoder_Q", "encoder_Q", "prefill_Q",
            "max_decoder_Q", "input_sha256", "trace")

# Registered per-shard digests for this checkpoint family (measured on A800 and n164).
WEIGHTS = Path(__file__).resolve().parents[3] / "benchmarks" / "checkpoints" / "llada21-weights-sha256.json"
CHUNK = 1 << 22

# Small metadata is content-hashed; weight shards are named, sized and verified
# against the registered digest (streamed, never read whole).
_HASHED = ("config.json", "generation_config.json", "tokenizer_config.json",
           "special_tokens_map.json", "tokenizer.json", "model.safetensors.index.json")
_SMALL = 1 << 20


def reference_root():
    root = os.environ.get("BDLM_LLADA_REFERENCE")
    require(bool(root) and Path(root).is_dir(),
            "BDLM_LLADA_REFERENCE must point at the native reference directory")
    return Path(root)


def weight_manifest(path=None):
    """Registered shard digests; the manifest file itself is tiny and hashed whole."""
    src = Path(path) if path is not None else WEIGHTS
    require(src.is_file(), f"missing weight manifest: {src}")
    reg = json.loads(src.read_bytes())
    require(isinstance(reg, dict) and reg, "weight manifest must be a non-empty object")
    for name, digest in reg.items():
        require(isinstance(name, str) and isinstance(digest, str) and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest),
                f"weight manifest entry is not a sha256 hex digest: {name!r}")
    return reg


def stream_sha256(path, chunk=CHUNK):
    """Digest a shard by streaming; never materialise the whole file in memory."""
    with open(path, "rb") as f:
        try:
            return hashlib.file_digest(f, "sha256").hexdigest()
        except AttributeError:                       # pragma: no cover - python < 3.11
            h = hashlib.sha256()
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
            return h.hexdigest()


def verify_weights(checkpoint, manifest=None, chunk=CHUNK):
    """Fail closed before inference when a registered shard is missing or altered."""
    reg = weight_manifest() if manifest is None else manifest
    ckpt = Path(checkpoint)
    measured, missing, mismatch = {}, [], []
    for name, want in sorted(reg.items()):
        shard = ckpt / name
        if not shard.is_file():
            missing.append(name)
            continue
        have = stream_sha256(shard, chunk)
        measured[name] = have
        if have != want:
            mismatch.append(dict(name=name, registered=want, measured=have,
                                 size=shard.stat().st_size))
    require(not missing, f"missing registered weight shards: {missing}")
    require(not mismatch, f"weight shard digest mismatch: {mismatch}")
    return measured


def fingerprint(checkpoint, manifest=None):
    """Deterministic identity: metadata hashed, weights named, sized and registered."""
    ckpt = Path(checkpoint)
    require(ckpt.is_dir(), f"checkpoint directory missing: {ckpt}")
    files = sorted(p for p in ckpt.rglob("*") if p.is_file())
    require(bool(files), "empty checkpoint directory")
    lines = []
    for p in files:
        rel, size = p.relative_to(ckpt).as_posix(), p.stat().st_size
        digest = sha256(p.read_bytes()).hexdigest() if (rel in _HASHED or size <= _SMALL) else ""
        lines.append(f"{rel}:{size}:{digest}")
    reg = weight_manifest() if manifest is None else manifest
    lines.append("weights-manifest:" + sha256(
        json.dumps(reg, sort_keys=True).encode()).hexdigest())
    lines.extend(f"registered:{name}:{digest}" for name, digest in sorted(reg.items()))
    return sha256("\n".join(lines).encode()).hexdigest()


def canonical_prompt(tok, task, reference_ids):
    """Exactly the project's validate_prompts rule: chat template, no specials.

    The raw request prompt is never tokenised directly; that would silently pass a
    different chat framing than the frozen native reference.
    """
    text = tok.apply_chat_template([dict(role="user", content=task["prompt"])],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
    if ids != list(reference_ids):
        first = next((i for i, (a, b) in enumerate(zip(ids, reference_ids)) if a != b), None)
        raise RuntimeError("prompt token drift vs native reference: %r" % dict(
            index=task["index"], framed_sha256=sha256(text.encode()).hexdigest(),
            local=len(ids), reference=len(reference_ids), first_divergence=first))
    return list(ids), "chat_template"


def roles_of(trace):
    """Disjoint Q roles over billed decoder forwards; the parts must sum to Q."""
    total = {k: 0 for k in ROLES}
    for e in trace:
        rows, dirty = set(e["rows"]), set(e["dirty"])
        cold, mask, prefix = set(e["cold"]), set(e["mask"]), set(e["prefix"])
        groups = dict(dirty_write=dirty, uninitialized_mask=cold,
                      resident_mask=mask-cold, clean_resolved=prefix)
        part = {name: len(group) for name, group in groups.items()}
        require(rows == set.union(*groups.values()), "Q roles do not partition the paid rows")
        require(sum(part.values()) == len(rows) == e["q"],
                "Q roles do not partition the paid rows")
        for k in ROLES:
            total[k] += part[k]
    require(sum(total.values()) == sum(e["q"] for e in trace), "Q roles do not sum to Q")
    return total


def _load_gpu(args, before_cuda):
    """Indirection so preflight ordering is testable without touching the loader."""
    from .cross_race_llada import load_gpu
    return load_gpu(args, before_cuda)


def load(checkpoint, manifest=None):
    """Load one shard on the single pinned CUDA device. No holder handling."""
    import torch
    require(torch.cuda.is_available(), "CUDA is required; CPU inference is refused")
    require(torch.cuda.device_count() == 1, "exactly one visible GPU is required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-"),
            "CUDA_VISIBLE_DEVICES must pin one GPU uuid")
    require(sha256(Path(scoring.__file__).read_bytes()).hexdigest() == SCORER_SHA,
            "unexpected scorer build")
    reg = weight_manifest() if manifest is None else manifest
    weights = verify_weights(checkpoint, reg)          # before any inference
    fp = fingerprint(checkpoint, reg)
    tok, model = _load_gpu(SimpleNamespace(checkpoint=str(Path(checkpoint))), lambda: None)
    require(model.device.type == "cuda", "model is not on CUDA")
    mask = tok.convert_tokens_to_ids("<|mask|>")
    require(isinstance(mask, int) and mask >= 0, "tokenizer has no <|mask|> id")
    return dict(model=model, tok=tok, checkpoint=str(Path(checkpoint)), fingerprint=fp,
                weights=weights, scorer=SCORER_SHA, mask_id=mask, eos_id=tok.eos_token_id,
                prefill_F=1)


def evaluate(handle, task, offer, params):
    """One frozen task through the unified planner on an already loaded shard."""
    from bdlm_serving.methods.t2t.unified_rows import Params
    require(type(offer) is int and offer in TIERS, f"offer must be one of {TIERS}")
    require(isinstance(params, Params), "params must be unified_rows.Params")
    require(handle["model"].device.type == "cuda", "CPU inference is refused")
    ref_path = reference_root() / f"{task['index']:02d}-native32.json"
    require(ref_path.is_file(), f"missing native reference: {ref_path}")
    ref = json.loads(ref_path.read_bytes())
    require(ref["index"] == task["index"], "reference/task index mismatch")
    require(ref["generation_cap"] == CAP, f"native reference must be capped at {CAP}")
    prompt_ids, how = canonical_prompt(handle["tok"], task, ref["prompt_ids"])
    r = evaluate_fixed(handle["model"], handle["tok"], task, prompt_ids, offer, Law(),
                       request_cls=unified_llada.factory(params=params), audit_admission=True)
    missing = [k for k in REQUIRED if k not in r]
    require(not missing, f"_fixed row is missing expected keys: {missing}")
    require(r["encoder_Q"] == 0, "LLaDA has no encoder forward; encoder_Q must be 0")
    trace = r["trace"]
    require(all(e["layer_physical_q"] == [e["q"]] * 20 for e in trace),
            "per-layer charge does not match the paid width")
    require(sum(e["q"] for e in trace) == r["Q"] == r["decoder_Q"], "billed Q mismatch")
    require(r["max_decoder_Q"] <= offer, "decoder width above the offer")
    failure = r["failure"]
    require(failure is None or failure == NONCONVERGENT, f"unexpected failure: {failure}")
    return dict(status=STATUS_COMPLETE if failure is None else STATUS_ALGORITHM_FAILURE,
                correct=bool(r["correct"]) and failure is None,
                N=r["N"], F=r["F"], Q=r["Q"], decoder_F=r["F"],
                encoder_F=0, prefill_Q=r["prefill_Q"], prefill_F=handle["prefill_F"],
                offer=offer, forwards=len(trace), max_decoder_Q=r["max_decoder_Q"],
                decoder_Q=r["decoder_Q"], encoder_Q=0, roles=roles_of(trace),
                prompt_encoding=how, input_sha256=r["input_sha256"],
                reference=str(ref_path), failure=failure)
