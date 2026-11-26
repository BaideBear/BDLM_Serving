"""Promote an unchanged full verification's KV at a serial LLaDA commit.

Opt-in for the single-block hard-prefix executor. Never promote sparse or
speculative KV. Counters in events describe real forwards; logical commit
history is separate. Request state, including its caches, has a single owner.
"""
from dataclasses import dataclass, field

import torch
from transformers.cache_utils import DynamicCache

from .llada_sparse import LLaDASparseKernel, SparseRequestState


@dataclass
class VerifiedState(SparseRequestState):
    verifier_input: tuple | None = None
    commits: list = field(default_factory=list)


class VerifiedCommitKernel(LLaDASparseKernel):
    def create(self, *args, **kwargs):
        state = super().create(*args, **kwargs)
        if self.prefix_quota is None:
            raise ValueError("Verified promotion requires the single-block hard-prefix executor")
        return VerifiedState(**vars(state))

    def _forward(self, state, q_start, q_end, store, phase, stage=None, budget=None):
        candidate = state.verifier_input
        state.verifier_input = None
        if phase == "commit" and store:
            previous = state.events[-1] if state.events else {}
            if (candidate is not None and previous.get("stage") == "verify"
                    and previous.get("edited") == 0 and previous.get("masked_before") == 0
                    and previous.get("masked_after") == 0 and q_end-q_start == 32
                    and (q_start, q_end, state.context_version) == candidate[:3]
                    and state.cache is candidate[3] and state.cache.get_seq_length() == q_start
                    and state.resident is candidate[4] and len(state.resident) == len(self.projections)
                    and all(k.shape[-2] == v.shape[-2] == q_end for k,v in state.resident)
                    and torch.equal(candidate[5], state.x[:,q_start:q_end])):
                # The normal serial commit does not consume logits. Its caller
                # still advances the prefix and checks EOS exactly once.
                state.cache = DynamicCache.from_legacy_cache(state.resident)
                state.commits.append(dict(block=state.block, context_version=state.context_version,
                                          promoted=True, source_forward=len(state.events)-1))
                return None
        verifying = phase == "denoise" and stage == "verify" and not store
        ids = state.x[:,q_start:q_end].clone() if verifying else None
        logits = super()._forward(state,q_start,q_end,store,phase,stage,budget)
        if verifying:
            state.verifier_input = (q_start,q_end,state.context_version,state.cache,state.resident,ids)
        if phase == "commit" and store:
            state.commits.append(dict(block=state.block, context_version=state.context_version,
                                      promoted=False, source_forward=len(state.events)-1))
        return logits
