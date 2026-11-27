"""Inventory visible cross-block mask opportunities from frozen90 traces.

This is a CPU-only, read-only audit of already-written per-request traces.  It
does not load model weights, acquire a GPU, submit jobs, or edit any existing
experiment file.  The only writes are the new ``controls/offer_mask_inventory.*``
artifacts selected by ``--out-prefix``.

Two trace dialects are understood:

* ``gemma`` (``experiments/t2t/serving_slots/unified90.py``): a compact
  ``trace`` list.  Each decoder step stores ``tick, offer, q, head, phase,
  mandatory_q, overflow, encoded, promoted, prefix, parked, strict_edits,
  opportunity``.  ``opportunity`` is the persisted boolean produced from the
  full ``selected``/``visible`` geometry.  The compact trace does not retain
  those sets, so this audit checks first-opportunity consistency and physical
  work invariants rather than re-deriving the boolean from raw positions.

* ``llada21``: an ``events`` list.  ``opportunity`` is recomputed here from
  ``absolute_positions`` and ``visible_kv`` using the same definition as
  ``mask_pair_report.eligible`` (head-block position selected while the next
  32-token block is visible).

Important reporting rule: zero visible opportunities means the cross-block
contrast was **not measured**, never that it had a zero effect.  Single-block
budgets cannot test a cross-block mask at all.

Typical n164 invocation (Python 3.12 venv supplies math_verify deps)::

    PYTHONPATH=.:/data1/project_data/gx-erqw-20261123/runtime/fixed-baseline-deps \
    /data1/project_data/gx-erqw-20261123/runtime/venv/bin/python -m \
      experiments.t2t.serving_slots.offer_mask_inventory \
      --data-root /data1/project_data/gx-erqw-20261123/runtime/t2t-native-1125/prefix90
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[3] if len(_HERE.parents) > 3 else Path.cwd()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BLOCK = {"gemma": 256, "llada21": 32}
COHORTS = {"original30": (0, 30), "new60": (30, 90), "full90": (0, 90)}
PREINPUT = "prefill"


@dataclass(frozen=True)
class ArmSpec:
    key: str
    model: str
    kind: str                 # fixed | switch | control | variant | reference
    tier: str                 # nominal budget or path label
    policy: str
    directory: str
    mask: str                 # on | off
    pair: Optional[str] = None
    pair_side: Optional[str] = None
    root: str = "prefix90"
    note: str = ""
    expected_requests: int = 90


# Curated primary arms: every completed frozen90 arm needed to answer "which
# tier has a real, visible cross-block opportunity, and is a matched mask-on/off
# contrast available?"  Additional exploratory arms can be appended without
# touching the analyzer.
REGISTRY: Tuple[ArmSpec, ...] = (
    # DiffusionGemma fixed budgets.  Q32..Q256 occupy one 256-row physical block.
    ArmSpec("gemma-fixed-q32", "gemma", "fixed", "32", "q32",
            "gemma-unified90-q32", "on"),
    ArmSpec("gemma-fixed-q64", "gemma", "fixed", "64", "q64",
            "gemma-unified90-q64", "on"),
    ArmSpec("gemma-fixed-q128", "gemma", "fixed", "128", "q128",
            "gemma-unified90-q128-q128_slack_random", "on"),
    ArmSpec("gemma-fixed-q128-k32", "gemma", "variant", "128+k32",
            "q128_slack_random", "gemma-unified90-q128-q128_slack_random", "on",
            note="fixed Q128 with up to 32 in-block decoded prefixes"),
    ArmSpec("gemma-fixed-q256", "gemma", "fixed", "256", "q256",
            "gemma-unified90-q256", "on"),
    ArmSpec("gemma-fixed-q512", "gemma", "fixed", "512", "q512",
            "gemma-unified90-q512-q512_open", "on", "gemma-q512", "on"),
    ArmSpec("gemma-fixed-q512-open", "gemma", "control", "512_open", "q512_open",
            "gemma-unified90-q512-q512_open", "off", "gemma-q512", "off"),
    ArmSpec("gemma-fixed-q512-spawn50", "gemma", "variant", "512+spawn50",
            "q512_spawn50", "gemma-unified90-q512-q512_spawn50", "on",
            note="earlier second-canvas spawn; no mask-off twin in this set"),
    # DiffusionGemma five-slot external replay (switching) and variants.
    ArmSpec("gemma-switch-replay", "gemma", "switch", "replay", "replay",
            "gemma-unified90-replay-replay_open", "on", "gemma-replay", "on"),
    ArmSpec("gemma-switch-replay-open", "gemma", "control", "replay_open",
            "replay_open", "gemma-unified90-replay-replay_open", "off",
            "gemma-replay", "off"),
    ArmSpec("gemma-switch-replay-k32", "gemma", "variant", "replay+k32",
            "replay_q128_prefix32", "gemma-unified90-replay-replay_q128_prefix32",
            "on", note="replay path with Q128 in-block prefix injection"),
    ArmSpec("gemma-switch-dwell", "gemma", "switch", "dwell", "dwell",
            "gemma-unified90-dwell-dwell_q128_prefix32", "on",
            note="permuted replay path; no mask-off twin in this set"),
    ArmSpec("gemma-switch-dwell-k32", "gemma", "variant", "dwell+k32",
            "dwell_q128_prefix32", "gemma-unified90-dwell-dwell_q128_prefix32",
            "on", note="permuted replay path with Q128 prefix injection"),
    # LLaDA fixed budgets.  Only Q64 nominally spans two 32-row blocks.
    ArmSpec("llada-fixed-q4", "llada21", "fixed", "4",
            "t2t_q4_f0_m4_hard_prefix_zero", "llada21-4-matched", "on"),
    ArmSpec("llada-fixed-q8", "llada21", "fixed", "8",
            "t2t_q8_f0_m8_hard_prefix_zero", "llada-q8", "on"),
    ArmSpec("llada-fixed-q16", "llada21", "fixed", "16",
            "t2t_q16_f0_m16_hard_prefix_zero", "llada21-16-matched", "on"),
    ArmSpec("llada-fixed-q32", "llada21", "fixed", "32",
            "t2t_q32_f0_m32_hard_prefix_zero", "llada21-32-matched", "on"),
    ArmSpec("llada-fixed-q64", "llada21", "fixed", "64",
            "t2t_q64_f0_m64_hard_prefix_zero", "llada21-64-matched", "on"),
    ArmSpec("llada-fixed-q64-progress80", "llada21", "fixed",
            "64+progress80", "progress80-causal",
            "llada21-q64-progress80-mask-gpu90-v1", "on",
            "llada-q64-progress80", "on"),
    ArmSpec("llada-fixed-q64-progress80-open", "llada21", "control",
            "64+progress80_open", "progress80-open",
            "llada21-q64-progress80-mask-gpu90-v1", "off",
            "llada-q64-progress80", "off"),
    # LLaDA in-block switching arms (8<->16).  These never create a second
    # physical block, so they cannot test a cross-block mask.
    ArmSpec("llada-switch-8", "llada21", "switch", "8", "slottrace_8",
            "llada21-slots-8-16", "on"),
    ArmSpec("llada-switch-16", "llada21", "switch", "16", "slottrace_16",
            "llada21-slots-8-16", "on"),
    ArmSpec("llada-switch-8-8-16-16", "llada21", "switch", "8,8,16,16",
            "slottrace_8_8_16_16", "llada21-slots-8-16", "on"),
    ArmSpec("llada-switch-16-16-8-8", "llada21", "switch", "16,16,8,8",
            "slottrace_16_16_8_8", "llada21-slots-8-16", "on"),
    ArmSpec("llada-reference-dual64", "llada21", "reference", "dual64",
            "dual64_noedit", "llada21-rolling-dual64", "on",
            note="true two-block reference; no mask-off 90-row twin here"),
)

CAVEATS = (
    "Zero visible opportunities means the cross-block contrast was not measured; "
    "it must not be reported as a zero difference.",
    "A single 256-row (Gemma) or 32-row (LLaDA) physical block cannot host a "
    "cross-block read, so Q32/Q64/Q128/Q256 fixed Gemma arms and Q4/Q8/Q16/Q32 "
    "fixed LLaDA arms are structurally outside the mask question.",
    "Gemma compact traces retain a persisted opportunity boolean but not the "
    "selected/visible sets; this audit verifies the boolean against physical work "
    "and first_opportunity, and cross-checks per-tier counts against existing "
    "offer-work.csv files when present.",
    "LLaDA opportunities are recomputed from absolute_positions and visible_kv, "
    "matching mask_pair_report.eligible; stored per-request opportunity counts "
    "are checked when present.",
    "Low-tier opportunities inside the replay path occur only after a 512-row "
    "exposure and are derived (cached/tail-refresh) opportunities, not "
    "independent per-tier mask interventions.",
)

SCOPE = (
    "Included: prefix90 unified90 DiffusionGemma fixed/replay arms and LLaDA "
    "matched fixed/slots/dual64 arms.",
    "Excluded from this run: LLaDA 16<->32n raw traces under "
    "private-llada-native-slot90-v1 and the older Gemma slottrace roots.  They "
    "are separate data roots; append ArmSpec entries to audit them, and do not "
    "infer their cross-block effect from this inventory.",
)


@dataclass
class Step:
    offer: int
    q: int
    F: int
    Q: int
    phase: str
    opportunity: bool
    dual_active: bool
    tail_refresh: bool
    cached_only: bool
    parked: int = 0
    visible_kv: Optional[int] = None


@dataclass
class Request:
    index: int
    dataset: str
    N: int
    F: int
    Q: int
    steps: List[Step]
    correct: Optional[bool]
    score_error: Optional[str]
    output_sha: str
    source: Path

    @property
    def tpf(self) -> float:
        return self.N / self.F if self.F else 0.0

    @property
    def efficiency(self) -> float:
        return self.N / self.Q if self.Q else 0.0

    @property
    def opportunity_ticks(self) -> int:
        return sum(1 for s in self.steps if s.opportunity)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def request_files(data_root: Path, spec: ArmSpec) -> List[Path]:
    folder = data_root / spec.directory
    if not folder.is_dir():
        raise FileNotFoundError(f"missing trace directory: {folder}")
    out = []
    for path in folder.glob("*.json"):
        stem = path.stem
        if len(stem) < 4 or not stem[:2].isdigit() or stem[2] != "-":
            continue
        if stem[3:] == spec.policy:
            out.append(path)
    return sorted(out, key=lambda p: int(p.stem[:2]))


def lla_eligible(event: Dict[str, Any], prompt_rows: int, block: int = 32) -> bool:
    """Same definition as mask_pair_report.eligible."""
    if event.get("phase") == PREINPUT:
        return False
    end = int(event["visible_kv"])
    if end % block:
        raise ValueError(f"Unexpected LLaDA visible width {end}")
    low = max(prompt_rows, end - 2 * block)
    high = end - block
    return any(low <= int(p) < high for p in event["absolute_positions"])


def gemma_steps(row: Dict[str, Any]) -> List[Step]:
    block = BLOCK["gemma"]
    steps: List[Step] = []
    for event in row.get("trace") or []:
        encoded = len(event.get("encoded") or ())
        offer = int(event["offer"])
        q = int(event["q"])
        parked = int(event.get("parked", 0))
        opportunity = bool(event.get("opportunity"))
        # At Q>256 the tail rows are actually sampled, not merely parked.
        dual = opportunity and q > block
        refresh = opportunity and not dual and parked > 0
        cached = opportunity and not dual and parked == 0
        steps.append(Step(
            offer=offer, q=q, F=1 + encoded, Q=q + block * encoded,
            phase=str(event.get("phase")), opportunity=opportunity,
            dual_active=dual, tail_refresh=refresh, cached_only=cached,
            parked=parked,
        ))
    return steps


def llada_steps(row: Dict[str, Any]) -> List[Step]:
    block = BLOCK["llada21"]
    prompt_rows = int(row.get("prefill_rows", block))
    steps: List[Step] = []
    for event in row.get("events") or []:
        if event.get("phase") == PREINPUT:
            continue
        end = int(event["visible_kv"])
        positions = [int(p) for p in event["absolute_positions"]]
        opportunity = lla_eligible(event, prompt_rows, block)
        tail_sampled = any(p >= end - block for p in positions)
        dual = opportunity and tail_sampled
        cached = opportunity and not tail_sampled
        nominal = event.get("q_budget")
        actual = event.get("actual_q", event.get("physical_q"))
        trial = nominal if nominal is not None else actual
        steps.append(Step(
            offer=int(trial if trial is not None else event["physical_q"]),
            q=int(event["physical_q"]), F=1, Q=int(event["physical_q"]),
            phase=str(event.get("phase", "")), opportunity=opportunity,
            dual_active=dual, tail_refresh=False, cached_only=cached,
            visible_kv=end,
        ))
    return steps


def validate_physical(
    spec: ArmSpec, row: Dict[str, Any], steps: Sequence[Step], path: Path
) -> None:
    if spec.model == "gemma":
        total_f = sum(s.F for s in steps)
        total_q = sum(s.Q for s in steps)
        decoder_q = sum(s.q for s in steps)
        encoder_q = total_q - decoder_q
        if row.get("decoder_q") is not None and int(row["decoder_q"]) != decoder_q:
            raise ValueError(f"{path}: decoder_q {row['decoder_q']} != {decoder_q}")
        if row.get("encoder_q") is not None and int(row["encoder_q"]) != encoder_q:
            raise ValueError(f"{path}: encoder_q {row['encoder_q']} != {encoder_q}")
        first = next((i for i, s in enumerate(steps) if s.opportunity), None)
        if row.get("first_opportunity") is not None and int(row["first_opportunity"]) != first:
            raise ValueError(
                f"{path}: first_opportunity {row['first_opportunity']} != {first}"
            )
    else:
        total_f = len(steps)
        total_q = sum(s.Q for s in steps)
        if row.get("opportunities") is not None:
            observed = sum(1 for s in steps if s.opportunity)
            if int(row["opportunities"]) != observed:
                raise ValueError(
                    f"{path}: stored opportunities {row['opportunities']} != {observed}"
                )
    for key, observed in (
        ("generation_forwards", total_f),
        ("generation_physical_padded_rows", total_q),
    ):
        stored = row.get(key)
        if stored is not None and int(stored) != observed:
            raise ValueError(f"{path}: {key} {stored} != {observed}")
    if row.get("tpf") is not None and row.get("generation_forwards"):
        expected = float(row["output_tokens"]) / float(row["generation_forwards"])
        if abs(float(row["tpf"]) - expected) > 1e-9:
            raise ValueError(f"{path}: stored tpf disagrees with N/F")
    if row.get("token_efficiency") is not None and row.get("generation_physical_padded_rows"):
        expected = float(row["output_tokens"]) / float(row["generation_physical_padded_rows"])
        if abs(float(row["token_efficiency"]) - expected) > 1e-9:
            raise ValueError(f"{path}: stored token_efficiency disagrees with N/Q")


class QualityScorer:
    """Score delivered text with the registered frozen90 scorer, with caching."""

    def __init__(self, tasks: Sequence[Dict[str, Any]], mode: str = "auto") -> None:
        self.tasks = tasks
        self.mode = mode
        self.cache: Dict[Tuple[int, str], Tuple[Optional[bool], Optional[str]]] = {}
        self.dataset_errors: Counter = Counter()
        self._score = None
        self.import_error: Optional[str] = None
        if mode != "off":
            try:
                from bdlm_serving.evaluation.scoring import score as score_fn
                self._score = score_fn
            except Exception as exc:  # pragma: no cover - environment dependent
                self.import_error = f"{type(exc).__name__}: {exc}"
                if mode == "on":
                    raise

    @property
    def enabled(self) -> bool:
        return self._score is not None

    def __call__(self, index: int, text: str) -> Tuple[Optional[bool], Optional[str]]:
        key = (index, sha256_text(text))
        if key in self.cache:
            return self.cache[key]
        if self._score is None:
            result = (None, self.import_error or "scorer disabled")
            self.cache[key] = result
            return result
        task = self.tasks[index]
        try:
            outcome = self._score(text, task)
            result = (bool(outcome["correct"]), None)
        except Exception as exc:  # e.g. missing math_verify or sandbox
            dataset = str(task.get("dataset", "?"))
            self.dataset_errors[dataset] += 1
            result = (None, f"{type(exc).__name__}: {exc}")
        self.cache[key] = result
        return result


def load_requests(
    data_root: Path, spec: ArmSpec, scorer: QualityScorer
) -> List[Request]:
    files = request_files(data_root, spec)
    requests: List[Request] = []
    for path in files:
        row = read_json(path)
        index = int(row["index"])
        steps = gemma_steps(row) if spec.model == "gemma" else llada_steps(row)
        validate_physical(spec, row, steps, path)
        text = row.get("output_text")
        if not isinstance(text, str):
            correct, error = None, "missing output_text"
        else:
            correct, error = scorer(index, text)
        requests.append(Request(
            index=index,
            dataset=str(row.get("dataset", "")),
            N=int(row["output_tokens"]),
            F=int(row["generation_forwards"]),
            Q=int(row["generation_physical_padded_rows"]),
            steps=list(steps),
            correct=correct,
            score_error=error,
            output_sha=hashlib.sha256(
                json.dumps(row.get("output_ids"), separators=(",", ":")).encode()
            ).hexdigest(),
            source=path,
        ))
    if len({r.index for r in requests}) != len(requests):
        raise ValueError(f"{spec.key}: duplicate request indices")
    return requests


def aggregate(requests: Sequence[Request]) -> Dict[str, Any]:
    if not requests:
        return dict(
            requests=0, requests_with_opportunity=0, opportunity_ticks=0,
            opportunity_dual_active=0, opportunity_tail_refresh=0,
            opportunity_cached_only=0, steps=0, N=0, F=0, Q=0,
            macro_tpf=None, micro_eff=None, correct=None, quality=None,
            quality_n=0, quality_computable=False, quality_status="empty",
            quality_note="no requests",
        )
    opportunities = [r for r in requests if r.opportunity_ticks]
    steps = [s for r in requests for s in r.steps]
    scored = [r for r in requests if r.correct is not None]
    correct = sum(1 for r in scored if r.correct)
    n = sum(r.N for r in requests)
    f = sum(r.F for r in requests)
    q = sum(r.Q for r in requests)
    missing = Counter(r.dataset for r in requests if r.correct is None)
    quality = (correct / len(scored)) if scored else None
    if len(scored) == len(requests):
        status = "computed_all"
    elif scored:
        status = "partial"
    else:
        status = "uncomputable"
    note = ""
    if missing:
        note = "; ".join(f"{k}:{v} unscored" for k, v in sorted(missing.items()))
    return dict(
        requests=len(requests),
        requests_with_opportunity=len(opportunities),
        opportunity_ticks=sum(1 for s in steps if s.opportunity),
        opportunity_dual_active=sum(1 for s in steps if s.dual_active),
        opportunity_tail_refresh=sum(1 for s in steps if s.tail_refresh),
        opportunity_cached_only=sum(1 for s in steps if s.cached_only),
        steps=len(steps),
        N=n, F=f, Q=q,
        macro_tpf=sum(r.tpf for r in requests) / len(requests),
        micro_eff=(n / q) if q else None,
        correct=(correct if scored else None),
        quality=quality,
        quality_n=len(scored),
        quality_computable=(len(scored) == len(requests)),
        quality_status=status,
        quality_note=note,
    )


def tier_rows(spec: ArmSpec, cohort: str, requests: Sequence[Request]) -> List[Dict[str, Any]]:
    tiers: Dict[int, Dict[str, Any]] = {}
    for request in requests:
        for step in request.steps:
            cell = tiers.setdefault(step.offer, dict(
                requests=set(), requests_with_opportunity=set(),
                opportunity_ticks=0, dual_active=0, tail_refresh=0,
                cached_only=0, steps=0, decoder_F=0, decoder_Q=0, F=0, Q=0,
            ))
            cell["requests"].add(request.index)
            cell["steps"] += 1
            cell["decoder_F"] += 1
            cell["decoder_Q"] += step.q
            cell["F"] += step.F
            cell["Q"] += step.Q
            if step.opportunity:
                cell["requests_with_opportunity"].add(request.index)
                cell["opportunity_ticks"] += 1
                cell["dual_active"] += int(step.dual_active)
                cell["tail_refresh"] += int(step.tail_refresh)
                cell["cached_only"] += int(step.cached_only)
    out = []
    for offer in sorted(tiers):
        cell = tiers[offer]
        out.append(dict(
            model=spec.model, kind=spec.kind, tier=spec.tier,
            policy=spec.policy, mask=spec.mask, cohort=cohort, offer=offer,
            requests_with_tier=len(cell["requests"]),
            requests_with_opportunity=len(cell["requests_with_opportunity"]),
            opportunity_ticks=cell["opportunity_ticks"],
            dual_active=cell["dual_active"],
            tail_refresh=cell["tail_refresh"],
            cached_only=cell["cached_only"],
            steps=cell["steps"], decoder_F=cell["decoder_F"],
            decoder_Q=cell["decoder_Q"], F=cell["F"], Q=cell["Q"],
        ))
    return out


def leading_int(text: str) -> Optional[int]:
    digits = ""
    for ch in text:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else None


def classify(
    spec: ArmSpec, agg: Dict[str, Any], has_pair: bool, expected: int
) -> Tuple[str, str]:
    if agg["requests"] != expected:
        return (
            "not_computable_incomplete",
            f"{agg['requests']}/{expected} requests present",
        )
    if agg["opportunity_ticks"] == 0:
        block = BLOCK[spec.model]
        nominal = leading_int(spec.tier)
        if nominal is not None and nominal < 2 * block:
            return (
                "not_computable_no_visible_opportunity",
                f"single physical block: nominal {nominal} < 2x{block}",
            )
        if spec.model == "llada21" and nominal == 64:
            return (
                "not_computable_no_visible_opportunity",
                "nominal two-block tier but observed execution samples one "
                "32-row block; no cross-block read",
            )
        return (
            "not_computable_no_visible_opportunity",
            "zero observed visible cross-block opportunities",
        )
    if has_pair:
        return "computable_paired_intervention", "matched mask-on/off arms present"
    return "descriptive_only", "visible opportunities but no matched mask-off arm"


def identity_groups(
    per_arm: Dict[str, Sequence[Request]], specs: Dict[str, ArmSpec]
) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[str]] = defaultdict(list)
    for key, requests in per_arm.items():
        spec = specs[key]
        if len(requests) != spec.expected_requests:
            continue
        signature = (
            spec.model,
            tuple((r.index, r.output_sha, r.N, r.F, r.Q) for r in requests),
        )
        buckets[signature].append(key)
    groups = []
    for keys in buckets.values():
        if len(keys) > 1:
            groups.append(dict(
                model=specs[keys[0]].model,
                policies=sorted(keys),
                requests=specs[keys[0]].expected_requests,
                note="bit-identical delivered ids, N, F and Q on every request",
            ))
    return sorted(groups, key=lambda g: (g["model"], g["policies"]))


def pair_rows(
    specs: Dict[str, ArmSpec],
    arm_rows: Dict[Tuple[str, str], Dict[str, Any]],
    per_arm: Dict[str, Sequence[Request]],
) -> List[Dict[str, Any]]:
    pairs: Dict[str, Dict[str, ArmSpec]] = defaultdict(dict)
    for spec in specs.values():
        if spec.pair and spec.pair_side:
            pairs[spec.pair][spec.pair_side] = spec
    out = []
    for _, sides in sorted(pairs.items()):
        if "on" not in sides or "off" not in sides:
            continue
        on, off = sides["on"], sides["off"]
        for cohort, (lo, hi) in COHORTS.items():
            on_req = [r for r in per_arm.get(on.key, []) if lo <= r.index < hi]
            off_req = [r for r in per_arm.get(off.key, []) if lo <= r.index < hi]
            if not on_req or not off_req:
                continue
            on_by = {r.index: r for r in on_req}
            off_by = {r.index: r for r in off_req}
            shared = sorted(set(on_by) & set(off_by))
            if len(shared) != len(on_req) or len(shared) != len(off_req):
                continue
            exposed = [i for i in shared if on_by[i].opportunity_ticks]
            unexposed = [i for i in shared if not on_by[i].opportunity_ticks]
            both_scored = [i for i in shared
                           if on_by[i].correct is not None and off_by[i].correct is not None]
            on_correct = sum(1 for i in both_scored if on_by[i].correct)
            off_correct = sum(1 for i in both_scored if off_by[i].correct)
            rescued = sum(1 for i in both_scored
                          if on_by[i].correct and not off_by[i].correct)
            lost = sum(1 for i in both_scored
                       if off_by[i].correct and not on_by[i].correct)
            on_tpf = sum(on_by[i].tpf for i in shared) / len(shared)
            off_tpf = sum(off_by[i].tpf for i in shared) / len(shared)
            on_n = sum(on_by[i].N for i in shared)
            off_n = sum(off_by[i].N for i in shared)
            on_q = sum(on_by[i].Q for i in shared)
            off_q = sum(off_by[i].Q for i in shared)
            on_f = sum(on_by[i].F for i in shared)
            off_f = sum(off_by[i].F for i in shared)
            opp = sum(on_by[i].opportunity_ticks for i in shared)
            status = (
                "not_computable_no_visible_opportunity" if opp == 0
                else "computable_paired_intervention"
            )
            note = (
                "no visible cross-block opportunity in either arm; the contrast "
                "is not measured and must not be reported as zero difference"
                if opp == 0 else
                "paired mask-on/off; deltas are arm-level, not per-tier causal "
                "attribution"
            )
            out.append(dict(
                pair=on.pair, cohort=cohort, status=status,
                on_policy=on.policy, off_policy=off.policy,
                requests=len(shared), exposed=len(exposed),
                unexposed=len(unexposed),
                opportunity_ticks_on=opp,
                opportunity_ticks_off=sum(off_by[i].opportunity_ticks for i in shared),
                quality_n=len(both_scored),
                on_correct=(on_correct if both_scored else None),
                off_correct=(off_correct if both_scored else None),
                rescued=(rescued if both_scored else None),
                lost=(lost if both_scored else None),
                on_quality=(on_correct / len(both_scored) if both_scored else None),
                off_quality=(off_correct / len(both_scored) if both_scored else None),
                quality_delta=(
                    (on_correct - off_correct) / len(both_scored)
                    if both_scored else None
                ),
                on_macro_tpf=on_tpf, off_macro_tpf=off_tpf,
                macro_tpf_delta=on_tpf - off_tpf,
                on_micro_eff=(on_n / on_q if on_q else None),
                off_micro_eff=(off_n / off_q if off_q else None),
                micro_eff_delta=(
                    (on_n / on_q - off_n / off_q) if on_q and off_q else None
                ),
                on_N=on_n, off_N=off_n, delta_N=on_n - off_n,
                on_F=on_f, off_F=off_f, delta_F=on_f - off_f,
                on_Q=on_q, off_Q=off_q, delta_Q=on_q - off_q,
                note=note,
            ))
    return out


def gemma_offer_work_crosscheck(
    data_root: Path, specs: Iterable[ArmSpec], tiers: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Compare per-tier counts with the already-published offer-work.csv."""
    by_key = {(t["policy"], t["cohort"], t["offer"]): t for t in tiers}
    checks = []
    for spec in specs:
        if spec.model != "gemma":
            continue
        path = data_root / (spec.directory + "-paired") / "offer-work.csv"
        if not path.is_file():
            continue
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("policy") != spec.policy:
                    continue
                cohort = row.get("cohort")
                offer = int(row["offer"])
                observed = by_key.get((spec.policy, cohort, offer))
                if observed is None:
                    continue
                checks.append(dict(
                    policy=spec.policy, cohort=cohort, offer=offer,
                    expected_opportunity_ticks=int(row["exposed_ticks"]),
                    observed_opportunity_ticks=observed["opportunity_ticks"],
                    expected_decoder_f=int(row["decoder_f"]),
                    observed_decoder_f=observed["decoder_F"],
                    expected_decoder_q=int(row["decoder_q"]),
                    observed_decoder_q=observed["decoder_Q"],
                    ok=(
                        int(row["exposed_ticks"]) == observed["opportunity_ticks"]
                        and int(row["decoder_f"]) == observed["decoder_F"]
                        and int(row["decoder_q"]) == observed["decoder_Q"]
                    ),
                ))
    return checks


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_outputs(
    out_prefix: Path,
    meta: Dict[str, Any],
    arms: Sequence[Dict[str, Any]],
    tiers: Sequence[Dict[str, Any]],
    pairs: Sequence[Dict[str, Any]],
    identities: Sequence[Dict[str, Any]],
    checks: Sequence[Dict[str, Any]],
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    arm_fields = [
        "model", "kind", "tier", "policy", "mask", "pair", "cohort",
        "requests", "expected_requests", "incomplete",
        "requests_with_opportunity", "opportunity_ticks",
        "opportunity_dual_active", "opportunity_tail_refresh",
        "opportunity_cached_only", "steps", "N", "F", "Q",
        "macro_tpf", "micro_eff", "correct", "quality", "quality_n",
        "quality_computable", "quality_status", "quality_note",
        "cross_block_status", "cross_block_reason",
        "identity_group", "identical_to", "note", "source_dir",
    ]
    with (out_prefix.with_suffix(".csv")).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=arm_fields, lineterminator="\n")
        writer.writeheader()
        for row in arms:
            writer.writerow({k: row.get(k) for k in arm_fields})

    tier_fields = [
        "model", "kind", "tier", "policy", "mask", "cohort", "offer",
        "requests_with_tier", "requests_with_opportunity",
        "opportunity_ticks", "dual_active", "tail_refresh", "cached_only",
        "steps", "decoder_F", "decoder_Q", "F", "Q",
    ]
    with (out_prefix.parent / (out_prefix.name + "_tiers.csv")).open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tier_fields, lineterminator="\n")
        writer.writeheader()
        for row in tiers:
            writer.writerow({k: row.get(k) for k in tier_fields})

    pair_fields = [
        "pair", "cohort", "status", "on_policy", "off_policy",
        "requests", "exposed", "unexposed",
        "opportunity_ticks_on", "opportunity_ticks_off",
        "quality_n", "on_correct", "off_correct", "rescued", "lost",
        "on_quality", "off_quality", "quality_delta",
        "on_macro_tpf", "off_macro_tpf", "macro_tpf_delta",
        "on_micro_eff", "off_micro_eff", "micro_eff_delta",
        "on_N", "off_N", "delta_N", "on_F", "off_F", "delta_F",
        "on_Q", "off_Q", "delta_Q", "note",
    ]
    with (out_prefix.parent / (out_prefix.name + "_pairs.csv")).open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=pair_fields, lineterminator="\n")
        writer.writeheader()
        for row in pairs:
            writer.writerow({k: row.get(k) for k in pair_fields})

    payload = dict(meta)
    payload.update(
        arms=list(arms), tiers=list(tiers), pairs=list(pairs),
        identity_groups=list(identities), cross_checks=list(checks),
        caveats=list(CAVEATS), scope=list(SCOPE),
    )
    (out_prefix.with_suffix(".json")).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    full = [r for r in arms if r["cohort"] == "full90"]
    lines = [
        "# Offer/mask inventory (frozen90, CPU trace audit)",
        "",
        f"- Data root: `{meta['data_root']}`",
        f"- Samples: `{meta['samples']}` (sha256 `{meta['samples_sha256']}`)",
        f"- Python: `{meta['python']}`",
        f"- Scorer: {meta['scorer']}",
        f"- Generated: {meta['generated_utc']}",
        "",
        "All N/F/Q are recomputed from per-request traces and checked against the",
        "stored physical ledgers.  Zero visible opportunities means **not measured**,",
        "not zero effect.",
        "",
        "## Full90 arms",
        "",
        "| model | kind | tier | mask | req | trig | opp ticks | dual | refresh | cached | F(all) | Q(all) | quality | macroTPF | micro eff | status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in full:
        quality = (
            f"{r['correct']}/{r['quality_n']}" if r.get("quality_n") else "—"
        )
        eff = "—" if r["micro_eff"] is None else f"{100.0 * r['micro_eff']:.2f}"
        values = dict(r)
        values.update(
            quality=quality, tpf=fmt(r["macro_tpf"]), eff=eff,
            status=r["cross_block_status"],
        )
        lines.append(
            "| {model} | {kind} | {tier} | {mask} | {requests} | "
            "{requests_with_opportunity} | {opportunity_ticks} | "
            "{opportunity_dual_active} | {opportunity_tail_refresh} | "
            "{opportunity_cached_only} | {F} | {Q} | {quality} | "
            "{tpf} | {eff}% | {status} |".format(**values)
        )
    lines += [
        "",
        "## Paired mask-on/off contrasts",
        "",
        "Only rows with visible opportunities and both arms are contrasts.  Rows",
        "marked `not_computable_no_visible_opportunity` are **not** zero-effect",
        "estimates.",
        "",
        "| pair | cohort | status | exposed | quality on/off | resc/lost | macroTPF on/off (Δ) | eff on/off (Δ pp) | F on/off | Q on/off |",
        "|---|---|---|---:|---|---:|---|---:|---:|---:|",
    ]
    for r in pairs:
        eff_on = "—" if r["on_micro_eff"] is None else f"{100.0*r['on_micro_eff']:.2f}%"
        eff_off = "—" if r["off_micro_eff"] is None else f"{100.0*r['off_micro_eff']:.2f}%"
        eff_d = (
            "—" if r["micro_eff_delta"] is None
            else f"{100.0*r['micro_eff_delta']:+.2f}"
        )
        values = dict(r)
        values.update(
            eff_on=eff_on, eff_off=eff_off, eff_d=eff_d,
            on_correct=fmt(r["on_correct"]), off_correct=fmt(r["off_correct"]),
            on_macro_tpf=fmt(r["on_macro_tpf"]),
            off_macro_tpf=fmt(r["off_macro_tpf"]),
            macro_tpf_delta=(r["macro_tpf_delta"] or 0.0),
            rescued=fmt(r["rescued"]), lost=fmt(r["lost"]),
        )
        lines.append(
            "| {pair} | {cohort} | {status} | {exposed} | {on_correct}/{off_correct} | "
            "{rescued}/{lost} | {on_macro_tpf}/{off_macro_tpf} ({macro_tpf_delta:+.3f}) | "
            "{eff_on}/{eff_off} ({eff_d}) | {on_F}/{off_F} | {on_Q}/{off_Q} |".format(
                **values
            )
        )
    lines += [
        "",
        "## Per-offer opportunities in switching paths",
        "",
        "| model | arm | cohort | offer | steps | dec F | dec Q | opp ticks | dual | refresh | cached | trig req |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in tiers:
        if r["kind"] not in ("switch", "variant", "reference") or r["cohort"] != "full90":
            continue
        lines.append(
            "| {model} | {policy} | {cohort} | {offer} | {steps} | "
            "{decoder_F} | {decoder_Q} | {opportunity_ticks} | {dual_active} | {tail_refresh} | "
            "{cached_only} | {requests_with_opportunity} |".format(**r)
        )
    lines += ["", "## Identity findings", ""]
    if identities:
        for g in identities:
            lines.append(
                f"- `{g['model']}`: {', '.join('`'+p+'`' for p in g['policies'])} "
                f"— {g['note']}."
            )
    else:
        lines.append("- No bit-identical full90 arm pairs found.")
    lines += ["", "## Cross-checks against existing offer-work.csv", ""]
    if checks:
        bad = [c for c in checks if not c["ok"]]
        lines.append(
            f"- {len(checks)-len(bad)}/{len(checks)} per-tier cells matched the "
            "published offer-work.csv."
        )
        for c in bad[:20]:
            lines.append(
                f"- MISMATCH `{c['policy']}` {c['cohort']} offer={c['offer']}: "
                f"opportunity {c['observed_opportunity_ticks']} vs "
                f"{c['expected_opportunity_ticks']}, F {c['observed_decoder_f']} vs "
                f"{c['expected_decoder_f']}, Q {c['observed_decoder_q']} vs "
                f"{c['expected_decoder_q']}."
            )
    else:
        lines.append("- No offer-work.csv files were available for this root.")
    lines += ["", "## Caveats", ""]
    lines += [f"- {c}" for c in CAVEATS]
    lines += ["", "## Scope", ""]
    lines += [f"- {s}" for s in SCOPE]
    lines.append("")
    out_prefix.with_suffix(".md").write_text("\n".join(lines))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("/data1/project_data/gx-erqw-20261123/runtime/t2t-native-1125/prefix90"),
        help="directory containing per-request trace folders and samples.json",
    )
    parser.add_argument(
        "--samples", type=Path, default=None,
        help="frozen90 samples.json (default: <data-root>/samples.json)",
    )
    parser.add_argument(
        "--out-prefix", type=Path,
        default=REPO / "experiments/t2t/serving_slots/controls/offer_mask_inventory",
        help="output prefix; .csv/_tiers.csv/_pairs.csv/.json/.md are added",
    )
    parser.add_argument(
        "--score", choices=("auto", "on", "off"), default="auto",
        help="recompute quality with the registered scorer (auto disables if unavailable)",
    )
    parser.add_argument(
        "--cohorts", default="full90,original30,new60",
        help="comma-separated cohorts to emit",
    )
    parser.add_argument(
        "--only", default=None,
        help="comma-separated arm keys or substrings to include",
    )
    parser.add_argument(
        "--no-cross-check", action="store_true",
        help="skip per-tier comparison with existing offer-work.csv",
    )
    args = parser.parse_args(argv)

    data_root = args.data_root.resolve()
    samples = (args.samples or (data_root / "samples.json")).resolve()
    if not samples.is_file():
        raise FileNotFoundError(f"missing samples.json: {samples}")
    selected = list(COHORTS)
    if args.cohorts.strip():
        selected = [c.strip() for c in args.cohorts.split(",") if c.strip()]
        unknown = [c for c in selected if c not in COHORTS]
        if unknown:
            raise ValueError(f"unknown cohorts: {unknown}")

    specs = list(REGISTRY)
    if args.only:
        needles = [s.strip() for s in args.only.split(",") if s.strip()]
        specs = [
            s for s in specs
            if any(n == s.key or n in s.key or n in s.policy for n in needles)
        ]
    if not specs:
        raise ValueError("no arms selected")

    from bdlm_serving.evaluation.frozen90 import load_tasks
    tasks = load_tasks(samples)
    scorer = QualityScorer(tasks, args.score)

    specs_by_key = {s.key: s for s in specs}
    per_arm: Dict[str, List[Request]] = {}
    for spec in specs:
        try:
            per_arm[spec.key] = load_requests(data_root, spec, scorer)
        except Exception as exc:
            raise RuntimeError(f"{spec.key}: {exc}") from exc

    identities = identity_groups(per_arm, specs_by_key)
    identity_for: Dict[str, str] = {}
    identical_to: Dict[str, str] = {}
    for i, group in enumerate(identities, 1):
        label = f"identity-{i}"
        for key in group["policies"]:
            identity_for[key] = label
            identical_to[key] = ",".join(k for k in group["policies"] if k != key)

    arms_out: List[Dict[str, Any]] = []
    tiers_out: List[Dict[str, Any]] = []
    for spec in specs:
        requests = per_arm[spec.key]
        has_pair = bool(spec.pair and any(
            s.pair == spec.pair and s.pair_side != spec.pair_side for s in specs
        ))
        for cohort in selected:
            lo, hi = COHORTS[cohort]
            subset = [r for r in requests if lo <= r.index < hi]
            agg = aggregate(subset)
            status, reason = classify(spec, agg, has_pair, hi - lo)
            row = dict(
                model=spec.model, kind=spec.kind, tier=spec.tier,
                policy=spec.policy, mask=spec.mask, pair=spec.pair or "",
                cohort=cohort,
                requests=agg["requests"],
                expected_requests=(hi - lo),
                incomplete=(agg["requests"] != hi - lo),
                requests_with_opportunity=agg["requests_with_opportunity"],
                opportunity_ticks=agg["opportunity_ticks"],
                opportunity_dual_active=agg["opportunity_dual_active"],
                opportunity_tail_refresh=agg["opportunity_tail_refresh"],
                opportunity_cached_only=agg["opportunity_cached_only"],
                steps=agg["steps"], N=agg["N"], F=agg["F"], Q=agg["Q"],
                macro_tpf=agg["macro_tpf"], micro_eff=agg["micro_eff"],
                correct=agg["correct"], quality=agg["quality"],
                quality_n=agg["quality_n"],
                quality_computable=agg["quality_computable"],
                quality_status=agg["quality_status"],
                quality_note=agg["quality_note"],
                cross_block_status=status,
                cross_block_reason=reason,
                identity_group=identity_for.get(spec.key, ""),
                identical_to=identical_to.get(spec.key, ""),
                note=spec.note,
                source_dir=str(data_root / spec.directory),
            )
            arms_out.append(row)
            tiers_out.extend(tier_rows(spec, cohort, subset))

    pairs_out = pair_rows(specs_by_key, {
        (r["policy"], r["cohort"]): r for r in arms_out
    }, per_arm)
    checks = [] if args.no_cross_check else gemma_offer_work_crosscheck(
        data_root, specs, tiers_out
    )

    meta = dict(
        schema="offer_mask_inventory/v1",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        data_root=str(data_root),
        samples=str(samples),
        samples_sha256=sha256_file(samples),
        python=sys.version.replace("\n", " "),
        scorer=(
            "bdlm_serving.evaluation.scoring.score"
            if scorer.enabled else
            f"disabled ({scorer.import_error or args.score})"
        ),
        scorer_dataset_errors=dict(sorted(scorer.dataset_errors.items())),
        module=str(Path(__file__).resolve()),
        registry=[asdict(s) for s in specs],
        cohorts=selected,
    )
    write_outputs(
        args.out_prefix.resolve(), meta, arms_out, tiers_out, pairs_out,
        identities, checks,
    )
    print(f"wrote {args.out_prefix.resolve()}.{{csv,json,md}}")
    print(
        "full90 arms:",
        ", ".join(
            f"{r['policy']}:{r['opportunity_ticks']}"
            for r in arms_out if r["cohort"] == "full90"
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
