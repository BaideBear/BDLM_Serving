"""Opt-in batch-one B256 head observer. No sampler, projector or commit policy.

Source checked against Transformers v5.12.0, generation_diffusion_gemma.py
(SHA256 cded57b7e365f0265b1c088e993c358b5f70acdf48055fce396540b4d9d691bc),
checkpoint cd40195554d3cfcb56f34b9adfb66efc790ed5b6. Its stopper receives
argmax(processed_logits), processed_logits after normal acceptance and renoise,
before the embedding-dtype SC cast. The inner loop includes cur_step=1;
exhaustion is not a positive stopper decision. Upstream emits the last draft;
the fixed executor must still pay outstanding writes and run its own verify.

Use one observer per request and one call per physical decoder tick, for the
current head only. Produce HeadSample directly from the current _consume's
FP32 scores, never from SC, an earlier draft, tail predictions or a row cache.
Pass None on an absent/non-sampling head. The producer tags the measurement
at projection, the caller independently identifies the live head at publish.
No interface can detect a caller deliberately relabelling cached data as fresh.

A True return is a one-tick signal for StrictRows.finish(..., native_stop=True)
after the ordinary sample. The executor must also consult that latched flag
in its next can_sample decision; dirty -> drain -> verify -> commit remains
unchanged. Do not consult the observer to suppress tail work in the hit tick.
"""
from dataclasses import dataclass

import torch


B = 256
FULL = tuple(range(B))


@dataclass(frozen=True)
class HeadTick:
    tick: int       # Request-wide physical decoder tick, not sampler age.
    ordinal: int    # Committed head, not a speculative tail.
    epoch: int      # Encoder-cache generation (or its committed boundary).

    def __post_init__(self):
        if any(type(x) is not int or x < 0 for x in (self.tick, self.ordinal, self.epoch)):
            raise ValueError("Expected nonnegative integer tick, ordinal and epoch")


@dataclass(frozen=True)
class HeadSample:
    source: HeadTick
    rows: tuple[int, ...]   # Sampled block-local positions, in scores order.
    scores: torch.Tensor   # Actual post-processor result for this one call.


class FullHeadStop:
    def __init__(self, criterion):
        """Own a fresh native criterion (or None); never share it between requests."""
        self.criterion = criterion
        self.last = None
        self._reset()

    @classmethod
    def from_model(cls, model, generation_config):
        """Use the pinned factory, including its disabled (None) configuration."""
        return cls(model._prepare_diffusion_stopping_criteria(generation_config))

    def _reset(self):
        if self.criterion is not None:
            self.criterion.reset()

    @torch.inference_mode()
    def observe(self, current: HeadTick, *, phase: str, sample: HeadSample | None = None) -> bool:
        """Return a stop hit, not a committable draft. Retain no logits."""
        if not isinstance(current, HeadTick):
            raise TypeError("Expected live head tick")
        last = self.last
        if last is not None:
            if current.tick <= last.tick:
                self._reset()
                raise ValueError("Replayed or out-of-order decoder tick")
            if (current.tick != last.tick + 1 or current.ordinal != last.ordinal
                    or current.epoch != last.epoch):
                self._reset()
        self.last = current
        if phase in ("absent", "drain", "rebuild", "verify"):
            self._reset()
            if sample is not None:
                raise ValueError("Non-sampling head cannot publish a sample")
            return False
        if phase != "sample" or sample is None or sample.source != current:
            self._reset()
            raise ValueError("Expected a fresh sample from this head, tick and epoch")
        rows, scores = sample.rows, sample.scores
        if (not rows or any(type(r) is not int or not 0 <= r < B for r in rows)
                or tuple(sorted(set(rows))) != rows or not isinstance(scores, torch.Tensor)
                or scores.dtype != torch.float32 or scores.ndim != 3
                or scores.shape[:2] != (1, len(rows)) or scores.shape[-1] < 2):
            self._reset()
            raise ValueError("Expected ordered local rows and aligned [1,Q,V] FP32 processed scores")
        if rows != FULL:
            self._reset()
            return False
        if self.criterion is None:
            return False
        return bool(self.criterion(scores.argmax(dim=-1), scores).item())
