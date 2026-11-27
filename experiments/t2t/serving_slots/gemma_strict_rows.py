"""Hard physical Q for one B256 Gemma canvas. No logits, KV tensors or sampler here.

Project ``rows`` once; run denoising only on ``sampled`` and scatter only there.
MASK means not accepted, not a token ID. Dirty means an acceptance transition
or a changed decoder input (include renoise; include changed SC for input-exact
replay). KV refreshed against a partially resident block is not dense native KV.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .gemma_unified import _choose

B = 256
OFFERS = (32, 64, 128, 256)


@dataclass(frozen=True)
class RowPlan:
    offer: int
    rows: tuple[int, ...]       # Sorted block-local Q, suitable for CanvasProjector.
    sampled: tuple[int, ...]    # Empty means drain only: no RNG, SC or stop update.
    cold: tuple[int, ...]       # Newly allocated resident rows.
    prefix: tuple[int, ...]     # Optional, sampled (T2T/revocation allowed).

    @property
    def drain_only(self):
        return not self.sampled

    @property
    def sample_indices(self):
        """Offsets into the measured [1, len(rows), vocab] logits."""
        take = set(self.sampled)
        return tuple(i for i, r in enumerate(self.rows) if r in take)


class StrictRows:
    """One outstanding plan. Commit after every layer successfully wrote its KV.

    ``accepted``, ``initialized`` and ``dirty`` are sets of block-local row IDs.
    ``age`` counts sampler calls; ``tick`` counts physical decoder calls. After
    at most max_steps samples and up to eight drains per sample, plan returns
    None. ``incomplete`` is a failure, not permission to emit the random canvas.
    """
    def __init__(self, max_steps, *, seed=0, ordinal=0):
        if type(max_steps) is not int or max_steps < 1 or type(seed) is not int or (
                type(ordinal) is not int or ordinal < 0):
            raise ValueError("Expected positive max_steps, integer seed and nonnegative ordinal")
        self.max_steps, self.seed, self.ordinal = max_steps, seed, ordinal
        self.age = self.tick = 0
        self.accepted = self.dirty = self.initialized = frozenset()
        self._stamps = [-1] * B
        self._native_stop = False
        self._pending = None

    @property
    def status(self):
        if self.dirty:
            return "running"
        if self._native_stop:
            return "native_stop"
        if len(self.accepted) == B:
            return "resolved"  # Local quiescence, not native adaptive stopping.
        if self.age == self.max_steps:
            return "limit" if len(self.initialized) == B else "incomplete"
        return "running"

    def plan(self, offer, *, prefix_quota=0, prefix_mode="right"):
        if type(offer) is not int or offer not in OFFERS or type(prefix_quota) is not int or (
                not 0 <= prefix_quota <= offer or prefix_mode not in ("zero", "right", "oldest", "random")
                or (prefix_mode == "zero" and prefix_quota)):
            raise ValueError("Invalid strict decoder offer or prefix policy")
        if self._pending is not None:
            raise RuntimeError("Finish the outstanding plan first")
        if self.status != "running":
            return None
        debt = sorted(self.dirty)
        missing = [r for r in range(B) if r not in self.initialized]
        cold = prefix = ()
        closing = self.age == self.max_steps or self._native_stop or len(self.accepted) == B
        if closing or (len(debt) >= offer and offer < B):
            rows, sampled = tuple(debt[:offer]), ()
        elif missing:
            # One physical call, but only the cold cohort is sampled. Mixing
            # first visits and SC-bearing revisits in one sampler call is invalid.
            cold = sampled = tuple(missing[:offer - len(debt)])
            rows = tuple(sorted((*debt, *cold)))
        else:
            # Once warm, _choose already implements debt / MASK / slack-prefix.
            bits = np.fromiter((r in self.accepted for r in range(B)), dtype=bool)
            view = SimpleNamespace(sampler=SimpleNamespace(accepted_token_mask=bits[None]),
                                   dirty=self.dirty, ordinal=self.ordinal,
                                   last_projected=self._stamps)
            picked, prefix, _ = _choose(view, offer, prefix_quota, prefix_mode,
                                        seed=self.seed, tick=self.tick)
            rows = sampled = tuple(picked)
        if not 0 < len(rows) <= offer or list(rows) != sorted(set(rows)):
            raise RuntimeError("Invalid physical Q")
        self._pending = RowPlan(offer, rows, sampled, cold, tuple(prefix))
        return self._pending

    def finish(self, plan, *, accepted_after=(), input_changed=(), native_stop=False):
        """Publish bool vectors aligned exactly to plan.sampled, never B-wide.

        input_changed compares pre-forward vs next inputs for each sampled row.
        Pass no vectors on drains. A native stop requires a full sampled canvas.
        Caller owns tokens, drafts, SC, sampler and RNG; never call them on drains.
        """
        if self._pending is None or plan is not self._pending:
            raise RuntimeError("Wrong or already finished plan")
        accepted, changed = tuple(accepted_after), tuple(input_changed)
        if (len(accepted) != len(plan.sampled) or len(changed) != len(accepted)
                or any(type(b) is not bool for b in accepted + changed)
                or type(native_stop) is not bool or (native_stop and len(accepted) != B)):
            raise ValueError("Expected sampled-row bool vectors; native stop requires B sampled rows")
        after = {r for r, ok in zip(plan.sampled, accepted) if ok}
        flips = (self.accepted.intersection(plan.sampled)) ^ after
        self.dirty = (self.dirty.difference(plan.rows) | flips |
                      {r for r, new in zip(plan.sampled, changed) if new})
        self.accepted = self.accepted.difference(plan.sampled) | after
        self.initialized = self.initialized.union(plan.rows)
        for r in plan.rows:
            self._stamps[r] = self.tick
        self.age += bool(plan.sampled)
        self.tick += 1
        self._native_stop |= native_stop
        self._pending = None
