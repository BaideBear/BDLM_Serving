"""Fixed IBS/OBS controls and the SDAR adapters for the frozen baseline matrix."""
import types
import numpy as np
import torch
from transformers.cache_utils import DynamicCache
from bdlm_serving.methods.optimus.upstream import load_controls, SelectedResidentCache, COMMIT


class FixedOptimus:
    def __init__(self, model, kind, chunk, mode):
        assert chunk in (4, 8, 16, 32) and mode in ("ibs", "obs")
        self.model, self.kind, self.chunk, self.mode = model, kind, chunk, mode
        self.controls, strategy, self.hashes = load_controls()
        self.unmask = self.controls.UnmaskingProcessor(types.SimpleNamespace(
            confidence_threshold=.9, block_length=32, denoising_steps=32,
            unmasking_strategy=strategy.LOW_CONFIDENCE_DYNAMIC))

    @torch.inference_mode()
    def run(self, prompt, gen_length, seed, mask_id, eos_ids):
        p = prompt.shape[1]
        total = ((p + gen_length + 31)//32)*32
        x = torch.full((1, total), mask_id, dtype=torch.long, device=prompt.device)
        x[:, :p] = prompt
        cache = SelectedResidentCache(total)
        start = p//32*32
        events = []

        def forward(positions, phase):
            cache.configure(positions, positions.tolist())
            if phase == "prefill":
                visible = positions[:, None]//32 >= positions[None, :]//32
            else:
                # Official delayed-cache attention sees all resident K/V rows
                # through the rightmost selected position, including OBS tail.
                visible = torch.ones((len(positions), cache.visible_end),
                                     dtype=torch.bool, device=x.device)
            kw = {"store_kv": True} if self.kind == "sdar" else {"output_attentions": True}
            out = self.model(x[:, positions], position_ids=positions[None],
                             attention_mask=visible[None, None], past_key_values=cache,
                             use_cache=True, **kw)
            events.append({"phase": phase, "q_tokens": len(positions),
                           "physical_forwards": 1, "absolute_positions": positions.tolist()})
            return out.logits

        if start:
            forward(torch.arange(start, device=x.device), "prefill")
        previous_shifted = None
        while start < total:
            width = min(32 + (self.chunk if self.mode == "obs" else 0), total-start)
            control = self.controls.DelayedCacheState.new(
                32, window_length=width, chunked_decode_enabled=True,
                chunk_size=self.chunk, extended_window_enabled=self.mode == "obs")
            if previous_shifted is not None:
                prior = np.zeros(width, dtype=np.uint8)
                prior[:min(width, len(previous_shifted))] = previous_shifted[:width]
                control.prime_from_shifted_window(prior)
            for step in range(2*width + 4):
                before = x[:, start:start+width].clone()
                old_mask = (before[0].cpu().numpy() != mask_id).astype(np.uint8)
                index_buffer = np.empty(width, dtype=np.int64)
                prefix_before = control.cached_prefix_len
                count, _ = control.fill_processing_buffers(index_buffer, 0, old_mask)
                if not count:
                    raise RuntimeError("Optimus returned an empty unfinished step")
                selected = torch.as_tensor(index_buffer[:count].copy(), device=x.device)
                positions = selected + start
                if self.mode == "ibs":
                    assert bool((positions < start+32).all())
                logits = forward(positions, "optimus")
                masked = before[0, selected] == mask_id
                proposed = logits[0].argmax(-1)
                accepted = torch.zeros(width, dtype=torch.bool, device=x.device)
                if bool(masked.any()):
                    selected_masks = selected[masked]
                    states = torch.as_tensor(old_mask.copy(), device=x.device)
                    new_states = self.unmask.low_confidence_dynamic_sparse(
                        logits[0, masked], proposed[masked], states, selected_masks,
                        row_width=width)
                    accepted = (new_states != 0) & (before[0] == mask_id)
                    take = accepted[selected_masks]
                    x[0, start+selected_masks[take]] = proposed[masked][take]
                edited = torch.zeros(count, dtype=torch.bool, device=x.device)
                if self.kind == "llada21":
                    confidence = logits[0].float().softmax(-1).gather(
                        -1, proposed[:, None]).squeeze(-1)
                    edited = (~masked & (positions >= p) &
                              (proposed != before[0, selected]) & (confidence > .5))
                    x[0, positions[edited]] = proposed[edited]
                now_mask = (x[0, start:start+width].cpu().numpy() != mask_id).astype(np.uint8)
                # A T2T output has not been encoded in this forward's stored KV.
                # Give it the same one-more-encoding requirement as a fresh M2T
                # output. Keep the original planner and prefix-update algorithm.
                cache_mask = old_mask.copy()
                deferred = selected[edited].cpu().numpy()
                cache_mask[deferred] = 0
                control.update_from_mask(cache_mask, mark_ready_if_all_left_non_mask=True,
                                         dllm_mask_cur=now_mask)
                ready = control.cached_prefix_len
                events[-1].update(
                    block=start//32, step=step, fixed_chunk=self.chunk, mode=self.mode,
                    cached_prefix_before=prefix_before, cached_prefix_after=ready,
                    selected_local=selected.tolist(), selected_masked=masked.tolist(),
                    accepted=int(accepted.sum()), edited=int(edited.sum()),
                    edited_positions=positions[edited].tolist(),
                    cache_deferred_t2t_positions=positions[edited].tolist(),
                    cross_boundary_rows=int((positions >= start+32).sum()),
                    visible_kv=cache.visible_end, active_width=width)
                assert not bool((positions[edited] < start+ready).any())
                assert not bool((x[:, :p] != prompt).any())
                if ready >= 32:
                    assert bool((x[:, start:start+32] != mask_id).all())
                    previous_shifted = cache_mask[32:].copy() if self.mode == "obs" else None
                    break
            else:
                raise RuntimeError("Optimus progress bound exceeded")
            if any(bool((x[:, p:start+32] == eos).any()) for eos in eos_ids):
                break
            start += 32
        return x, events, {
            "commit": COMMIT, "source_hashes": self.hashes, "mode": self.mode,
            "fixed_chunk": self.chunk, "window_switches": 0,
            "m2t_threshold": .9, "t2t_threshold": .5 if self.kind == "llada21" else None,
            "t2t_cache_rule": "edited input must be re-encoded before prefix freeze",
            "scope": "unchanged official prefix/unmask controls; common backend; selected-only T2T",
        }
