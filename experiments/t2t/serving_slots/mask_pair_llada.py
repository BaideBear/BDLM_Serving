"""LLaDA2.1 Q64 same-executor block-mask on/off ablation (frozen90).

This entry reuses the existing rolling two-B32 ``DualBlockKernel`` and the
archived ``llada21-rolling-dual64`` baseline.  It changes only the *model-local
attention visibility* through a forward pre-hook, so prefix construction,
commit order, sampling, RNG and the Q64 physical ledger stay on the shared
executor path.

Mask arms
---------
``on``   Existing block-causal mask.  A head (left B32) query cannot see tail
         (right B32) keys.  The in-row policy is ``dual64_noedit`` and must
         reproduce the archived baseline.
``off``  Open exactly the forbidden head->tail quadrant between the two blocks
         that are active in the same denoising step.  Tail->head and intra-block
         visibility are unchanged.  The in-row policy is
         ``dual64_noedit_bidir``.

The hook never runs during ``prefill``.  A prompt tail that lives inside the
left B32 block is excluded row-by-row (``position >= prompt_length``), so a
prompt/history token can never see a future generated token.  Prefix KV, commit
forwards and sampler calls are untouched.

Verification contract
---------------------
* Every opened forward records the first trigger stage, phase, block, forward
  index, head-row count and tail-key range.
* If a request opens no edge at all, its output ids, work counters and physical
  event fields must equal the archived baseline exactly.  This is fail-closed.
* The ``on`` arm must equal the archived baseline exactly for all 90 requests.
* ``mask_pair_summary.csv`` marks ``novel`` rows (opened_edges > 0); the paired
  scorer can compare off minus on while ignoring the extra metadata.

Scorer compatibility
--------------------
``tools.run_dual_t2t.audit_dual64`` is policy-agnostic: both rows keep the exact
``dual`` and ``dual_saturation`` fields, so the off label passes.  The current
``tools/prefix_cohorts.normalize_request`` branch still names only
``dual64_noedit``.  The minimal branch is::

    elif model == "llada21" and policy in ("dual64_noedit", "dual64_noedit_bidir"):
        from tools.run_dual_t2t import audit_dual64
        audit_dual64(raw)
        fixed_chunk = 64

Without that branch, run the off arm with ``MASK_PAIR_POLICY_OFF=dual64_noedit``;
the ``mask_pair.arm`` field and the ``off/`` folder still distinguish the arms.

Run directory plan
------------------
The protected launcher gives one ``TRACE_OUT`` root.  This entry writes::

    ROOT/run.json
    ROOT/on/{run.json,00-dual64_noedit.json,...}
    ROOT/off/{run.json,00-dual64_noedit_bidir.json,...}
    ROOT/mask_pair_summary.json
    ROOT/mask_pair_summary.csv

With ``MASK_PAIR_ARMS=off`` only ``ROOT/off`` is written; the read-only
archived baseline at ``prefix90/llada21-rolling-dual64`` is the on arm and is
recorded in ``ROOT/run.json``.  ``MASK_PAIR_ARMS=both`` writes both folders.

Queue contract
--------------
Do not start this from the CPU preflight.  On n164 physical GPU4 the current
registered tail is the ``prefix90-llada-cap4`` session.  The main agent should
register this entry after that tail exits, with the protected launcher::

    DATA=/data1/project_data/gx-erqw-20261123/runtime/t2t-native-1125
    ROOT=$DATA/prefix90
    export FROZEN90_SAMPLES=$ROOT/samples.json
    export PREFIX_MODE=llrefs
    export LL_BASELINE_POLICIES=dual64_noedit
    export MASK_PAIR_BASELINE=$ROOT/llada21-rolling-dual64
    export MASK_PAIR_ARMS=both
    export MASK_PAIR_REFERENCE=$DATA/frozen90-native/llada21
    export TRACE_OUT=$ROOT/mask-pair-llada21-v1
    $RUNTIME/venv/bin/python -u \
      $REPO/experiments/t2t/serving_slots/mask_pair_llada.py llada21 4

``python ... --preflight`` is CPU-only and never launches the holder.
"""
from __future__ import annotations

import argparse
import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

BLOCK = 32
PAIR = 2 * BLOCK
ARM_ON = "on"
ARM_OFF = "off"
ARM_BOTH = "both"
POLICY_ON = "dual64_noedit"
POLICY_OFF = "dual64_noedit_bidir"
BASELINE_POLICY = "dual64_noedit"
BASELINE_NAME = "llada21-rolling-dual64"
RUN_NAME = "mask-pair-llada21-v1"
EDGE = "head_to_tail"
LAUNCHER_SHA256 = "ba80a293773326950738fd0d4ca0a8a04e6df4b86c6e9a7265bd2ba3a92e6e59"
SAMPLES_SHA256 = "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed"


def _policy(arm: str) -> str:
    override = os.environ.get(f"MASK_PAIR_POLICY_{arm.upper()}")
    if override:
        return override
    if arm == ARM_ON:
        return POLICY_ON
    if arm == ARM_OFF:
        return POLICY_OFF
    raise ValueError(f"Unknown mask arm: {arm}")


def _arm_description(arm: str) -> str:
    return ("block_causal" if arm == ARM_ON
            else "bidirectional_active_pair_head_to_tail")


@dataclass(frozen=True)
class MaskPairContext:
    arm: str
    prompt_length: int
    block: int | None
    phase: str
    event_index: int
    enabled: bool
    stage_hint: str = ""
    record: dict | None = None


_CURRENT: contextvars.ContextVar[MaskPairContext | None] = contextvars.ContextVar(
    "mask_pair_context", default=None)


@contextmanager
def mask_pair_context(**kwargs):
    """Bind an experiment-local context for one kernel step."""
    token = _CURRENT.set(MaskPairContext(**kwargs))
    try:
        yield
    finally:
        _CURRENT.reset(token)


def head_to_tail_mask(attention_mask, position_ids, prompt_length, block):
    """Open head->tail only inside one active B32 pair.

    ``attention_mask`` is the binary visibility matrix passed by
    ``LLaDASparseKernel._execute``: shape ``(1, 1, q, visible_end)``.  The hook
    is intentionally fail-closed: if the incoming matrix is not binary, or if
    the head->tail quadrant was not fully masked, it raises instead of guessing.
    """
    if attention_mask.ndim != 4 or tuple(attention_mask.shape[:2]) != (1, 1):
        return attention_mask, None
    if position_ids.ndim == 2:
        if position_ids.shape[0] != 1:
            return attention_mask, None
        positions = position_ids[0]
    elif position_ids.ndim == 1:
        positions = position_ids
    else:
        return attention_mask, None
    if positions.numel() != attention_mask.shape[-2]:
        return attention_mask, None

    key_end = int(attention_mask.shape[-1])
    if block is None:
        # Pair passes end at (left_block + 2) * BLOCK.  A single-block commit,
        # right-only reencode, or one-block tail therefore gets a negative or
        # right-empty window and is a no-op.
        block = key_end // BLOCK - 2
    block = int(block)
    if block < 0:
        return attention_mask, None
    left_start = block * BLOCK
    left_end = left_start + BLOCK
    right_start = left_end
    right_end = min(right_start + BLOCK, key_end)
    if right_end <= right_start:
        return attention_mask, None

    head_rows = torch.nonzero(
        (positions >= left_start) & (positions < left_end) & (positions >= prompt_length)
    ).flatten()
    if head_rows.numel() == 0:
        return attention_mask, None
    tail_cols = torch.arange(right_start, right_end, device=attention_mask.device)
    window = attention_mask.index_select(-2, head_rows).index_select(-1, tail_cols)
    if window.dtype == torch.bool:
        masked = ~window
    else:
        values = set(torch.unique(window).tolist())
        if not values <= {0, 1}:
            raise RuntimeError("Mask-pair hook expected a binary visibility matrix")
        masked = window == 0
    opened = int(masked.sum().item())
    if opened == 0:
        return attention_mask, None
    if opened != head_rows.numel() * tail_cols.numel():
        raise RuntimeError("Head->tail quadrant was not fully masked before intervention")

    updated = attention_mask.clone()
    updated[..., head_rows[:, None], tail_cols[None, :]] = True
    info = dict(
        block=block,
        left_query_count=int(head_rows.numel()),
        left_query_first=int(positions[head_rows[0]].item()),
        left_query_last=int(positions[head_rows[-1]].item()),
        right_key_start=right_start,
        right_key_end=right_end,
        opened_edges=opened,
    )
    return updated, info


def _phase_for_stage(stage: str) -> str:
    return {
        "bootstrap": "denoise",
        "selected": "denoise",
        "reencode": "denoise",
        "commit": "commit",
        "prefill": "prefill",
    }.get(stage, stage)


def _stage_for_call(context: MaskPairContext, kwargs) -> str:
    cache = kwargs.get("past_key_values")
    cache_name = type(cache).__name__ if cache is not None else ""
    if cache_name == "_SelectedCache":
        return "selected"
    if cache_name == "_AppendCache":
        return "bootstrap"
    if cache_name == "_AuditedDenseCache":
        # In the dual path, a dense call with an open head->tail quadrant is
        # the pair bootstrap.  Commit/reencode calls have no tail or no head.
        return "bootstrap"
    if context.stage_hint:
        return context.stage_hint
    if context.phase:
        return context.phase
    return "forward"


def _pre_hook(_module, _args, kwargs):
    context = _CURRENT.get()
    if context is None or context.arm != ARM_OFF or not context.enabled:
        return None
    mask = kwargs.get("attention_mask")
    positions = kwargs.get("position_ids")
    if mask is None or positions is None:
        return None
    record = context.record
    if record is not None:
        record["forward_count"] += 1
    updated, info = head_to_tail_mask(mask, positions, context.prompt_length, context.block)
    if info is None:
        return None
    kwargs["attention_mask"] = updated
    if record is not None:
        record["opened_edges"] += info["opened_edges"]
        record["opened_forward_count"] += 1
        stage = _stage_for_call(context, kwargs)
        record["edge_forward_stages"][stage] = record["edge_forward_stages"].get(stage, 0) + 1
        if record["first_trigger"] is None:
            record["first_trigger"] = dict(
                stage=stage,
                phase=_phase_for_stage(stage),
                block=info["block"],
                event_index=(None if context.event_index < 0 else context.event_index),
                forward_index=record["forward_count"] - 1,
                left_query_count=info["left_query_count"],
                left_query_first=info["left_query_first"],
                left_query_last=info["left_query_last"],
                right_key_start=info["right_key_start"],
                right_key_end=info["right_key_end"],
                opened_edges=info["opened_edges"],
            )
    return None


def install_mask_pair_hook(model):
    """Install the experiment-local pre-hook; no shared kernel file changes."""
    target = getattr(model, "model", model)
    # The real LLaDA wrapper calls ``model.model``; toy test modules may expose
    # ``model.model`` only as a layer container.  Fall back to the called module.
    if getattr(type(target), "forward", None) is torch.nn.Module.forward:
        target = model
    if not hasattr(target, "register_forward_pre_hook"):
        raise TypeError("Model does not expose a forward pre-hook target")
    return target.register_forward_pre_hook(_pre_hook, with_kwargs=True)


def new_record(arm: str) -> dict:
    return dict(
        arm=arm,
        policy=_policy(arm),
        mask=_arm_description(arm),
        edge=EDGE,
        scope="same_active_pair_only",
        prefill_history_causal=True,
        forward_count=0,
        opened_edges=0,
        opened_forward_count=0,
        edge_forward_stages={},
        first_trigger=None,
    )


def _stage_hint(phase: str) -> str:
    return {
        "bootstrap": "bootstrap",
        "bootstrap_right": "bootstrap",
        "commit_left": "commit",
        "commit_right": "commit",
        "reencode_right": "reencode",
    }.get(phase, "")


def run_kernel_steps(kernel, state, arm: str, record: dict, max_steps: int = 8192):
    """Step the shared dual kernel with an experiment-local mask context."""
    for _ in range(max_steps):
        if state.phase == "done":
            return state
        phase = state.phase
        context = MaskPairContext(
            arm=arm,
            prompt_length=state.prompt_length,
            block=state.block,
            phase=phase,
            event_index=len(state.events),
            enabled=(arm == ARM_OFF),
            stage_hint=_stage_hint(phase),
            record=record,
        )
        token = _CURRENT.set(context)
        try:
            kernel.step(state)
        finally:
            _CURRENT.reset(token)
    raise RuntimeError("Mask-pair action limit exceeded")


PARITY_KEYS = (
    "output_ids", "output_tokens", "generation_forwards", "generation_query_rows",
    "generation_physical_padded_rows", "max_q", "q_histogram", "forwards_by_phase",
    "q_by_phase", "prefill_rows", "t2t_edits", "eos", "truncated",
)
EVENT_KEYS = (
    "phase", "stage", "physical_q", "actual_q", "effective_q", "absolute_positions",
    "layer_physical_q", "visible_kv", "blocks", "active_width", "saturated",
    "full_two_block", "q_budget", "overflow",
)


def assert_same_output_work(row: dict, baseline: dict, *, context: str = "zero-edge"):
    """Fail closed when a request that opened no edge diverges from baseline."""
    for key in PARITY_KEYS:
        if row.get(key) != baseline.get(key):
            raise ValueError(f"{context}: {key} differs from archived baseline")
    if len(row.get("events", [])) != len(baseline.get("events", [])):
        raise ValueError(f"{context}: physical event count differs from archived baseline")
    for index, (new, old) in enumerate(zip(row["events"], baseline["events"])):
        for key in EVENT_KEYS:
            if new.get(key) != old.get(key):
                raise ValueError(
                    f"{context}: event {index} {key} differs from archived baseline")


def _read(path):
    return json.loads(Path(path).read_text())


def durable(path: Path, data) -> None:
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("x") as stream:
        json.dump(data, stream, ensure_ascii=False)
    temp.replace(path)


def load_baseline(root: Path, samples_sha: str):
    meta = _read(root / "run.json")
    expected = dict(
        model="llada21", suite="frozen90", samples_sha256=samples_sha,
        block=BLOCK, pair_blocks=2, nominal_q=PAIR,
        mode="llada_dual_rolling_pair", cap=2048, batch_size=1,
    )
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Archived baseline {root}: wrong {key}")
    if BASELINE_POLICY not in meta.get("policies", []):
        raise ValueError(f"Archived baseline {root}: missing {BASELINE_POLICY}")
    if _read(root / f"complete-{BASELINE_POLICY}.json") != dict(
            policy=BASELINE_POLICY, requests=90):
        raise ValueError(f"Archived baseline {root}: incomplete marker")
    rows = {}
    for index in range(90):
        row = _read(root / f"{index:02d}-{BASELINE_POLICY}.json")
        if row.get("index") != index or row.get("policy") != BASELINE_POLICY:
            raise ValueError(f"Archived baseline {root}: wrong request {index}")
        rows[index] = row
    return meta, rows


def _reference_complete(path: Path) -> bool:
    return all((path / f"{index:02d}-native32.json").is_file() for index in range(90))


def _prompt_reference(args, samples: Path) -> Path:
    """Return the complete 90-case native32 prompt reference.

    The old ``fixed-baselines-llada-gpu4-v4/requests`` directory only covers
    the original 30 cases, so it is intentionally rejected here.  The archived
    dual64 baseline was produced against ``frozen90-native/llada21``.
    """
    explicit = os.environ.get("MASK_PAIR_REFERENCE")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    reference = getattr(args, "reference", None)
    if reference:
        candidates.extend([Path(reference), Path(reference) / "requests"])
    candidates.extend([
        samples.parent.parent / "frozen90-native" / "llada21",
        samples.parent / "frozen90-native" / "llada21",
    ])
    for candidate in candidates:
        if _reference_complete(candidate):
            return candidate
    if explicit:
        raise FileNotFoundError(
            f"MASK_PAIR_REFERENCE={explicit} is not a complete 90-task native32 reference")
    raise FileNotFoundError(
        "No complete 90-task native32 reference found; expected "
        "frozen90-native/llada21 beside prefix90")


def _tokenize_prompts(tok, tasks, prompt_reference: Path):
    lengths = {}
    for task in tasks:
        prompt = tok.apply_chat_template(
            [dict(role="user", content=task["prompt"])],
            tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
        reference = _read(prompt_reference / f"{task['index']:02d}-native32.json")
        if ids != reference["prompt_ids"]:
            raise ValueError(f"Changed prompt on request {task['index']:02d}")
        lengths[task["index"]] = len(ids)
    return lengths


def _runtime_provenance(args):
    """Fail closed if the launcher resolves shared modules from another tree."""
    import bdlm_serving
    import experiments
    import tools
    import torch
    import transformers

    repo = Path(__file__).resolve().parents[3]
    venv = (repo.parent / "venv").resolve()
    if Path(sys.prefix).resolve() != venv:
        raise RuntimeError(f"Expected launcher venv {venv}, got {sys.prefix}")
    shared = {}
    for module in (bdlm_serving, experiments, tools):
        paths = sorted({Path(path).resolve() for path in getattr(module, "__path__", [])})
        if not paths:
            raise RuntimeError(f"{module.__name__} has no package path")
        for path in paths:
            if repo not in path.parents:
                raise RuntimeError(f"{module.__name__} resolved outside repo: {path}")
        shared[module.__name__] = [str(path) for path in paths]
    # Fail if any sys.path entry exposes a second copy of a shared package.
    for entry in sys.path:
        if not entry:
            continue
        base = Path(entry)
        if not base.is_dir():
            continue
        for name in ("bdlm_serving", "experiments", "tools"):
            candidate = base / name
            if candidate.is_dir() and repo not in candidate.parents:
                raise RuntimeError(f"extra {name} on sys.path: {candidate}")
    package = Path(torch.__file__).resolve()
    if "site-packages" not in package.parts:
        raise RuntimeError(f"torch resolved outside site-packages: {package}")
    model_pkg = Path(transformers.__file__).resolve()
    if "site-packages" not in model_pkg.parts:
        raise RuntimeError(f"transformers resolved outside site-packages: {model_pkg}")
    return dict(
        repo=str(repo),
        module_paths=shared,
        torch=dict(version=str(torch.__version__), path=str(package)),
        transformers=dict(version=str(transformers.__version__), path=str(model_pkg)),
        python=sys.version.split()[0],
        venv=str(venv),
        executable=str(sys.executable),
        sys_path=sorted({str(Path(path).resolve()) for path in sys.path
                         if path and ("site-packages" in Path(path).parts
                                      or "fixed-baseline-deps" in str(path)
                                      or repo in Path(path).resolve().parents)}),
        checkpoint=str(Path(args.checkpoint).resolve()),
    )


def _install_hook_via_serial():
    """Wrap the shared mask installer so the experiment hook survives loading."""
    import bdlm_serving.runtime.serial as serial
    original = serial.install_llada_rectangular_mask

    def wrapped(model, *args, **kwargs):
        info = original(model, *args, **kwargs)
        install_mask_pair_hook(model)
        return info

    serial.install_llada_rectangular_mask = wrapped
    return original


def _run_one(run_task, dual_arm, task, arm, policy, baseline_row, prompt_length):
    from bdlm_serving.methods.t2t.protocol import verify_work
    from tools.run_dual_t2t import audit_dual64

    record = new_record(arm)
    context = MaskPairContext(
        arm=arm, prompt_length=prompt_length, block=None, phase="",
        event_index=-1, enabled=(arm == ARM_OFF), stage_hint="", record=record)
    token = _CURRENT.set(context)
    try:
        row = run_task(task, dual_arm)
    finally:
        _CURRENT.reset(token)
    trigger = record["first_trigger"]
    if trigger is not None and trigger.get("event_index") is None:
        # The dual kernel appends exactly one event per model forward.
        trigger["event_index"] = trigger["forward_index"]
    row.update({key: task[key] for key in ("index", "dataset", "seed", "task_id")},
               model="llada21", policy=policy)
    record["baseline_policy"] = BASELINE_POLICY
    record["zero_edge_identity"] = bool(record["opened_edges"] == 0)
    row["mask_pair"] = record
    verify_work(row)
    audit_dual64(row)
    if arm == ARM_ON or record["opened_edges"] == 0:
        assert_same_output_work(row, baseline_row, context=f"{arm}/{task['index']:02d}")
    return row


def _summary_rows(arm, rows):
    stages = {}
    for row in rows:
        for stage, count in row["mask_pair"]["edge_forward_stages"].items():
            stages[stage] = stages.get(stage, 0) + count
    return dict(
        arm=arm,
        policy=_policy(arm),
        mask=_arm_description(arm),
        requests=len(rows),
        correct=sum(int(row["correct"]) for row in rows),
        opened_edges=sum(int(row["mask_pair"]["opened_edges"]) for row in rows),
        opened_forward_count=sum(int(row["mask_pair"]["opened_forward_count"]) for row in rows),
        zero_edge_requests=sum(int(row["mask_pair"]["opened_edges"] == 0) for row in rows),
        novel_requests=sum(int(row["mask_pair"]["opened_edges"] > 0) for row in rows),
        first_trigger_stages={stage: sum(
            1 for row in rows
            if row["mask_pair"]["first_trigger"] is not None
            and row["mask_pair"]["first_trigger"]["stage"] == stage)
            for stage in sorted({row["mask_pair"]["first_trigger"]["stage"]
                                 for row in rows
                                 if row["mask_pair"]["first_trigger"] is not None})},
        edge_forward_stages=dict(sorted(stages.items())),
        N=sum(int(row["output_tokens"]) for row in rows),
        F=sum(int(row["generation_forwards"]) for row in rows),
        Q=sum(int(row["generation_query_rows"]) for row in rows),
    )


def _write_summary_csv(path: Path, rows):
    import csv
    columns = ("arm", "index", "dataset", "correct", "novel", "opened_edges",
               "opened_forward_count", "first_trigger_stage", "first_trigger_phase",
               "first_trigger_block", "first_trigger_event_index",
               "first_trigger_forward_index", "N", "F", "Q")
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            trigger = row["mask_pair"]["first_trigger"]
            writer.writerow(dict(
                arm=row["mask_pair"]["arm"],
                index=row["index"],
                dataset=row["dataset"],
                correct=int(row["correct"]),
                novel=int(row["mask_pair"]["opened_edges"] > 0),
                opened_edges=row["mask_pair"]["opened_edges"],
                opened_forward_count=row["mask_pair"]["opened_forward_count"],
                first_trigger_stage="" if trigger is None else trigger["stage"],
                first_trigger_phase="" if trigger is None else trigger["phase"],
                first_trigger_block="" if trigger is None else trigger["block"],
                first_trigger_event_index="" if trigger is None else trigger.get("event_index", ""),
                first_trigger_forward_index="" if trigger is None else trigger.get("forward_index", ""),
                N=row["output_tokens"],
                F=row["generation_forwards"],
                Q=row["generation_query_rows"],
            ))
    temp.replace(path)


def _resolve_arms(text: str):
    requested = str(text).strip()
    arms = ([ARM_ON, ARM_OFF] if requested == ARM_BOTH
            else [arm.strip() for arm in requested.split(",") if arm.strip()])
    if not arms or any(arm not in (ARM_ON, ARM_OFF) for arm in arms):
        raise ValueError("MASK_PAIR_ARMS must contain on, off, or both")
    if len(set(arms)) != len(arms):
        raise ValueError("Duplicate mask arms")
    return arms


def run(args, prepare=None):
    """Launcher-compatible entry: run the requested mask arm(s) on frozen90."""
    import copy
    from bdlm_serving.evaluation.frozen import sha
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from tools.run_dual_t2t import default_arms, dual_backend, durable

    samples = Path(args.samples)
    tasks = load_tasks(samples)
    if len(tasks) != 90 or [task["index"] for task in tasks] != list(range(90)):
        raise ValueError("Mask-pair ablation requires frozen90 in index order")
    samples_sha = sha(samples)
    if samples_sha != SAMPLES_SHA256:
        raise ValueError("Unexpected frozen90 sample digest")
    baseline_root = Path(os.environ.get(
        "MASK_PAIR_BASELINE", samples.parent / BASELINE_NAME)).resolve()
    baseline_meta, baseline_rows = load_baseline(baseline_root, samples_sha)
    block, pair = int(baseline_meta["block"]), int(baseline_meta["nominal_q"])
    if (block, pair) != (BLOCK, PAIR):
        raise RuntimeError("Archived baseline geometry changed")
    gen_length = int(baseline_meta["cap"])
    requested_cap = os.environ.get("MASK_PAIR_GEN_LENGTH")
    if requested_cap is not None and int(requested_cap) != gen_length:
        raise ValueError("MASK_PAIR_GEN_LENGTH must match the archived baseline cap")
    arms = _resolve_arms(os.environ.get("MASK_PAIR_ARMS", ARM_BOTH))
    prompt_reference = _prompt_reference(args, samples)
    provenance = _runtime_provenance(args)

    # Validate all 90 prompts before creating output or loading the model.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        Path(args.checkpoint), trust_remote_code=True, local_files_only=True)
    prompt_lengths = _tokenize_prompts(tok, tasks, prompt_reference)

    root = Path(args.output)
    root.mkdir(exist_ok=False, parents=True)

    manifest = dict(
        model="llada21", suite="frozen90", samples_sha256=samples_sha,
        block=block, pair_blocks=2, nominal_q=pair,
        mode="mask_pair_head_to_tail", batch_size=1, cap=gen_length,
        baseline=str(baseline_root), baseline_policy=BASELINE_POLICY,
        arms=[dict(arm=arm, policy=_policy(arm), mask=_arm_description(arm),
                   scope="same_active_pair_only", prefill_history_causal=True)
              for arm in arms],
        prompt_reference=str(prompt_reference),
        executor=dict(entry="tools.run_dual_t2t.dual_backend",
                      arm="t2t_q64_f0_m64_hard_noedit"),
        provenance=provenance,
        note=("Same DualBlockKernel and Q64 no-edit executor; only the active-pair "
              "head->tail visibility is ablated. Prompt/prefill history stays causal."),
    )
    durable(root / "run.json", manifest)

    # Reuse the original entry for model loading, native config, fixed Q64 arm,
    # sampler, packaging and scoring.  Only the visibility hook is local.
    call_args = copy.copy(args)
    call_args.model = "llada21"
    call_args.suite = "frozen90"
    call_args.reference = prompt_reference
    call_args.gen_length = gen_length
    dual_arm = default_arms()[0]
    if (dual_arm.budget, dual_arm.fresh, dual_arm.mask, dual_arm.selection) != (
            pair, 0, pair, "hard_noedit"):
        raise RuntimeError("Original dual entry no longer exposes the Q64 no-edit arm")
    import bdlm_serving.runtime.serial as serial
    original_install = _install_hook_via_serial()
    try:
        run_task = dual_backend(call_args, prepare)
    finally:
        serial.install_llada_rectangular_mask = original_install

    all_rows = []
    summaries = []
    for arm in arms:
        policy = _policy(arm)
        folder = root / arm
        folder.mkdir(exist_ok=False)
        durable(folder / "run.json", dict(
            model="llada21", suite="frozen90", samples_sha256=samples_sha,
            block=block, pair_blocks=2, nominal_q=pair,
            mode="llada_dual_rolling_pair", cap=gen_length, batch_size=1,
            policies=[policy], reference_policy30="t2t_q64_f0_m64_hard_noedit",
            mask_pair=dict(arm=arm, mask=_arm_description(arm), edge=EDGE,
                           scope="same_active_pair_only",
                           prefill_history_causal=True,
                           baseline=str(baseline_root),
                           baseline_policy=BASELINE_POLICY),
            provenance=provenance,
        ))
        rows = []
        for task in tasks:
            index = task["index"]
            row = _run_one(run_task, dual_arm, task, arm, policy,
                           baseline_rows[index], prompt_lengths[index])
            durable(folder / f"{index:02d}-{policy}.json", row)
            rows.append(row)
            trigger = row["mask_pair"]["first_trigger"]
            print(
                f"{arm}/{policy} {index:02d} N={row['output_tokens']} "
                f"F={row['generation_forwards']} Q={row['generation_query_rows']} "
                f"opened={row['mask_pair']['opened_edges']} "
                f"first={None if trigger is None else trigger['stage']} "
                f"correct={int(row['correct'])}",
                flush=True)
        durable(folder / f"complete-{policy}.json", dict(policy=policy, requests=90))
        summaries.append(_summary_rows(arm, rows))
        all_rows.extend(rows)

    durable(root / "mask_pair_summary.json", dict(
        baseline=str(baseline_root), baseline_policy=BASELINE_POLICY,
        samples_sha256=samples_sha, arms=summaries,
        provenance=provenance))
    _write_summary_csv(root / "mask_pair_summary.csv", all_rows)
    print("mask-pair summary:", json.dumps(summaries, ensure_ascii=False), flush=True)
    return summaries


def _queue_prerequisites(data: Path):
    # ``private-llada-native-slot90-v1`` lives beside ``prefix90``; the other
    # registered outputs are inside ``prefix90``.
    root = data.parent
    required = [
        root / "private-llada-native-slot90-v1" / "verified.json",
        data / "llada21-native-high-prefix16" / "complete-slottrace_16r1.json",
        data / "llada21-native-high-prefix16" / "complete-slottrace_32n.json",
        data / "llada21-native-high-prefix16" / "complete-slottrace_32n_32n_16r1_16r1.json",
        data / "llada21-native-high-prefix16" / "complete-slottrace_16r1_16r1_32n_32n.json",
        data / "llada21-optimus-hardcap-anchor" / "complete-optcap_ibs_32.json",
        data / "llada21-optimus-hardcap-anchor" / "complete-optcap_obs_32.json",
        data / "llada21-optimus-hardcap-anchor" / "replay-qualified.json",
        data / "llada21-optimus-hardcap-endpoints" / "complete-optcap_ibs_8.json",
        data / "llada21-optimus-hardcap-endpoints" / "complete-optcap_obs_8.json",
        data / "llada21-optimus-hardcap-endpoints" / "complete-optcap_ibs_16.json",
        data / "llada21-optimus-hardcap-endpoints" / "complete-optcap_obs_16.json",
        data / "llada21-optimus-hardcap-low" / "complete-optcap_ibs_4.json",
        data / "llada21-optimus-hardcap-low" / "complete-optcap_obs_4.json",
        data / "llada21-optimus-hardcap-switch" / "complete-optcap_ibs_8_8_16_16.json",
        data / "llada21-optimus-hardcap-switch" / "complete-optcap_obs_8_8_16_16.json",
        data / "llada21-optimus-hardcap-switch" / "complete-optcap_ibs_16_16_8_8.json",
        data / "llada21-optimus-hardcap-switch" / "complete-optcap_obs_16_16_8_8.json",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    return dict(gpu="4", tail_session="prefix90-llada-cap4",
                predecessors=["prefix90-llada-references", "prefix90-llada-fifo",
                              "prefix90-llada-native-slot", "prefix90-llada-cap4"],
                required=missing, ready=not missing)


def preflight(ns) -> int:
    samples = Path(ns.samples)
    if not samples.is_file():
        raise SystemExit(f"Missing samples: {samples}")
    digest = hashlib.sha256(samples.read_bytes()).hexdigest()
    if digest != SAMPLES_SHA256:
        raise SystemExit(f"Wrong frozen90 sample digest: {digest}")
    baseline = Path(ns.baseline or samples.parent / BASELINE_NAME)
    if not baseline.is_dir():
        raise SystemExit(f"Missing baseline: {baseline}")
    meta, rows = load_baseline(baseline, digest)
    output = Path(ns.output or samples.parent / RUN_NAME)
    if output.exists():
        raise SystemExit(f"Output already exists: {output}")
    provenance = _runtime_provenance(ns)
    prompt_reference = _prompt_reference(ns, samples)
    queue = _queue_prerequisites(samples.parent)
    print("mask-pair labels")
    print("  on : policy=%(on)s mask=%(on_mask)s" % dict(
        on=_policy(ARM_ON), on_mask=_arm_description(ARM_ON)))
    print("  off: policy=%(off)s mask=%(off_mask)s" % dict(
        off=_policy(ARM_OFF), off_mask=_arm_description(ARM_OFF)))
    print(f"baseline      : {baseline} ({len(rows)} requests)")
    print(f"prompt ref    : {prompt_reference} (90 native32 cases)")
    print(f"run root      : {output}")
    print(f"  on folder   : {output / ARM_ON}")
    print(f"  off folder  : {output / ARM_OFF}")
    print(f"summary       : {output / 'mask_pair_summary.json'}")
    print("provenance:", json.dumps(provenance, ensure_ascii=False))
    print("queue prerequisites:", json.dumps(queue, ensure_ascii=False))
    if not queue["ready"]:
        print("queue not ready; do not launch. Wait for the registered GPU4 tail.")
    print("CPU-only preflight; no model, CUDA command, holder or tmux was started.")
    return 0 if queue["ready"] or not ns.require_queue else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path,
                        default=Path(os.environ.get("FROZEN90_SAMPLES", "")))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/data0/ysy/models/LLaDA2.1-mini"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--require-queue", action="store_true")
    return parser.parse_args(argv)


def launcher_main(argv):
    """Re-exec the pinned protected launcher with this module as its worker."""
    if argv not in (["llada21", "4"], ["llada21", "4", "worker"]):
        raise ValueError("Only the registered n164 GPU4 LLaDA entry is supported")
    samples = Path(os.environ["FROZEN90_SAMPLES"]).resolve()
    if argv == ["llada21", "4"]:
        queue = _queue_prerequisites(samples.parent)
        if not queue["ready"]:
            raise RuntimeError(
                "GPU4 tail prerequisites are incomplete: "
                + ", ".join(queue["required"])
                + "; register after prefix90-llada-cap4 and do not seize GPU4")
    os.environ.setdefault("LL_BASELINE_POLICIES", POLICY_ON)
    os.environ.setdefault(
        "MASK_PAIR_REFERENCE", str(samples.parents[1] / "frozen90-native" / "llada21"))
    if not os.environ.get("TRACE_OUT"):
        raise RuntimeError("Set TRACE_OUT to the new mask-pair run root")
    launcher = samples.parents[1] / "launch_prefix90.py"
    source = launcher.read_bytes()
    if hashlib.sha256(source).hexdigest() != LAUNCHER_SHA256:
        raise RuntimeError("Protected launcher SHA256 changed; refusing to exec")
    target = b"from experiments.t2t.serving_slots.llada_reference import run"
    if source.count(target) != 1:
        raise RuntimeError("Protected launcher import target changed")
    if os.environ.get("PREFIX_MODE") != "llrefs":
        raise RuntimeError("Set PREFIX_MODE=llrefs for the protected llrefs path")
    replacement = b"from experiments.t2t.serving_slots.mask_pair_llada import run"
    exec(compile(source.replace(target, replacement), str(launcher), "exec"), globals())


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["llada21"]:
        return launcher_main(argv)
    ns = parse_args(argv)
    if ns.preflight:
        return preflight(ns)
    if ns.samples is None or not ns.samples.is_file():
        raise SystemExit("--samples is required")
    if ns.output is None:
        raise SystemExit("--output is required for a real run")
    os.environ.setdefault("MASK_PAIR_ARMS", ARM_BOTH)
    return run(ns)


if __name__ == "__main__":
    raise SystemExit(main())
