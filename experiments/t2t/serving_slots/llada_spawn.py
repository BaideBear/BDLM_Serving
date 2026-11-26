"""One-factor candidate: a Q64 offer permits, but need not open, a new tail."""
from math import isfinite

from experiments.t2t.serving_slots.llada_all_slots import AllSlotKernel, BLOCK, PAIR


class SpawnGatedKernel(AllSlotKernel):
    """Require measured head progress before opening a new tail; don't park an existing one.

    None is the eager control. 0.0 still requires an observation. This is a
    no-logits progress signal, NOT FlowBlock's raw-confidence acceptance ratio.
    """

    def __init__(self, model, *, spawn_threshold, **kwargs):
        super().__init__(model, **kwargs)
        if spawn_threshold is not None and (type(spawn_threshold) not in (int, float)
                or not isfinite(spawn_threshold) or not 0 <= spawn_threshold <= 1):
            raise ValueError("Expected None or a finite spawn threshold in [0,1]")
        self.spawn_threshold = spawn_threshold

    def admit_new_tail(self, s):
        if self.spawn_threshold is None:
            return True
        if s.resident is None or not s.events:
            return False
        obs = s.events[-1].get("block_counts", {}).get(str(s.block))
        if not obs or not obs["masked_before"]:
            return False
        return obs["masked_before"] - obs["masked_after"] >= self.spawn_threshold * obs["masked_before"]

    def _denoise(self, state, q, pending):
        new_tail = q == PAIR and self._has_right(state) and not self._tail_resident(state)
        deferred = new_tail and not self.admit_new_tail(state)
        # Keep the original offer in the public event and in the outer clock.
        action = super()._denoise(state, BLOCK if deferred else q, pending)
        if new_tail:
            state.events[-1]["tail_admission"] = "defer" if deferred else "open"
        return action
