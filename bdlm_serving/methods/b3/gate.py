"""Paper-derived B3 gate, not an official implementation.

Equations3–6 of archived arXiv2611.14387v1. Empty candidate set defers spawn.
The carry/reset interpretation at promotion is explicit for sensitivity tests.
"""
from dataclasses import dataclass, field

import torch


@dataclass
class B3Gate:
    block_size: int = 32
    reset_previous_on_promotion: bool = False
    frontier: int = -1
    candidate_sum: float = 0.
    candidate_count: int = 0
    previous_entropy: dict = field(default_factory=dict)
    previous_current: int = -1

    def observe(self, current_block, current_entropy, accepted_absolute,
                future_active, predicts_eos, future_entropy=None):
        if current_entropy.shape != (self.block_size,):
            raise ValueError("One entropy per native current-block position required")
        if self.reset_previous_on_promotion and current_block != self.previous_current:
            self.previous_entropy.pop(current_block, None)
        old_frontier = self.frontier
        if accepted_absolute:
            self.frontier = max(self.frontier, max(accepted_absolute))
        previous = self.previous_entropy.get(current_block)
        start = current_block*self.block_size
        candidate = None
        if (not future_active and self.frontier > old_frontier and previous is not None
                and start < self.frontier < start+self.block_size):
            candidate = float(previous[:self.frontier-start].max())
            self.candidate_sum += candidate
            self.candidate_count += 1
        tau = self.candidate_sum/self.candidate_count if self.candidate_count else None
        entropy_max = float(current_entropy.max())
        open_next = bool(not future_active and not predicts_eos and
                         tau is not None and entropy_max <= tau)
        self.previous_entropy[current_block] = current_entropy.detach().clone()
        if future_entropy is not None:
            if not future_active or future_entropy.shape != current_entropy.shape:
                raise ValueError("Future entropy requires an active future block")
            self.previous_entropy[current_block+1] = future_entropy.detach().clone()
        self.previous_current = current_block
        return {
            "frontier_before": old_frontier, "frontier_after": self.frontier,
            "future_active": future_active, "candidate": candidate,
            "candidate_count": self.candidate_count, "tau": tau,
            "entropy_max": entropy_max, "predicts_eos": predicts_eos,
            "open_on_next_pass": open_next,
            "promotion_previous_entropy": "reset" if self.reset_previous_on_promotion else "carry",
        }


def logits_entropy(logits):
    log_p = logits.float().log_softmax(-1)
    return -(log_p.exp()*log_p).sum(-1)
