"""B32-aligned paper-derived B3 control on the common native backend.

No official code provenance is claimed. Carry/reset of the future sampler's
post counter on promotion is an explicit sensitivity choice. Empty gate
candidates defer spawning. The Q64 cap is below the paper's B64/Q128 default.
"""
from dataclasses import dataclass, field

import torch

from bdlm_serving.methods.b3.gate import B3Gate, logits_entropy
from bdlm_serving.runtime.rolling import ExactRollingKernel, RetirementState


@dataclass
class B3State(RetirementState):
    gate: object = None
    opened_blocks: set = field(default_factory=set)
    future_sampler: dict = field(default_factory=dict)


class B3AlignedKernel(ExactRollingKernel):
    def __init__(self, model, promotion_post="carry", entropy_promotion="carry"):
        super().__init__(model, 64, seal=True, fuse_refresh=False)
        if promotion_post not in ("carry", "reset") or entropy_promotion not in ("carry", "reset"):
            raise ValueError("Explicit carry/reset interpretation required")
        self.promotion_post = promotion_post
        self.entropy_promotion = entropy_promotion
        self.capture_logits = True
        self.future_logits = None
        self.joint_requested = False

    def create(self, *args, **kwargs):
        base = super().create(*args, **kwargs)
        state = B3State(**vars(base))
        state.gate = B3Gate(reset_previous_on_promotion=self.entropy_promotion == "reset")
        return state

    def _forward(self, state, q_start, q_end, store, phase):
        if phase == "denoise" and self.joint_requested:
            logits = self._execute(state, q_start, q_end+32, phase)
            self.future_logits = logits[:, 32:]
            return logits[:, :32]
        return super()._forward(state, q_start, q_end, store, phase)

    def _update_future(self, state, block, logits):
        current = state.x[:, block*32:(block+1)*32]
        control = state.future_sampler.setdefault(block, {"steps": 0, "post_steps": 0, "finished": False})
        if control["finished"]:
            return {"accepted_positions": [], "accepted": 0, "edited": 0,
                    "finished": True, "post_steps": control["post_steps"]}
        before = current.clone()
        masked = before == state.mask_id
        count = int(masked.sum())
        if count == 0:
            control["post_steps"] += 1
            if control["post_steps"] > state.config.get("max_post_steps", 16):
                control["finished"] = True
                return {"accepted_positions": [], "accepted": 0, "edited": 0,
                        "finished": True, "post_steps": control["post_steps"]}
        proposed, confidence = self.model._sample_with_temperature_topk_topp(
            logits, temperature=0., top_k=None, top_p=None)
        accept = masked & (confidence > state.config["confidence_threshold"])
        if count and not bool(accept.any()):
            probability = confidence.masked_fill(~masked, -torch.inf)
            accept[0, probability[0].argmax()] = True
        edit = ~masked & (proposed != before) & (confidence > state.config["editing_threshold"])
        changed = accept | edit
        current[changed] = proposed[changed]
        control["steps"] += 1
        if count == 0 and not bool(edit.any()):
            control["finished"] = True
        if bool(changed.any()):
            state.block_revisions[block] = state.block_revisions.get(block, 0)+1
            state.kv_status[block]["status"] = "DIRTY"
        return {
            "accepted_positions": (accept[0].nonzero().flatten()+block*32).tolist(),
            "accepted": int(accept.sum()), "edited": int(edit.sum()),
            "finished": control["finished"], "post_steps": control["post_steps"],
        }

    @torch.inference_mode()
    def step(self, state):
        old_block = state.block
        start, end = old_block*32, (old_block+1)*32
        before = state.x[:, start:end].clone()
        self.future_logits = None
        self.joint_requested = (
            state.phase == "denoise" and old_block in state.opened_blocks
            and end+32 <= state.x.shape[1]
            and not any(bool((before == eos).any()) for eos in state.eos_ids))
        action = super().step(state)
        if action == "commit":
            # Nested calls from the native post-cap path may reach this branch
            # twice; assigning the same saved counter is idempotent.
            control = state.future_sampler.get(state.block)
            if control and self.promotion_post == "carry":
                state.post_steps = control["post_steps"]
            return action
        if action != "denoise":
            return action
        event = state.events[-1]
        accepted = ((before == state.mask_id) & (state.x[:, start:end] != state.mask_id))[0]
        accepted_positions = (accepted.nonzero().flatten()+start).tolist()
        future_active = self.future_logits is not None
        future_entropy = None
        if future_active:
            future_update = self._update_future(state, old_block+1, self.future_logits)
            accepted_positions += future_update["accepted_positions"]
            future_entropy = logits_entropy(self.future_logits)[0]
            event["future_sampler"] = future_update
            if event["accepted"] or event["edited"]:
                state.kv_status[old_block+1]["status"] = "DIRTY"
            # The dependency-closed current block may seal even though the
            # still-provisional future rows remain in absolute storage.
            if state.seal_ready is not None:
                state.cache.crop(end)
        current_entropy = logits_entropy(self.last_logits)[0]
        proposed = self.last_logits.argmax(-1)
        predicts_eos = any(bool((proposed == eos).any()) for eos in state.eos_ids)
        gate = state.gate.observe(old_block, current_entropy, accepted_positions,
                                  future_active, predicts_eos, future_entropy)
        if gate["open_on_next_pass"]:
            state.opened_blocks.add(old_block)
        event.update(
            b3_gate=gate, future_active=future_active,
            promotion_post_policy=self.promotion_post,
            baseline_scope="B32-aligned paper-derived control, not official implementation")
        return action
