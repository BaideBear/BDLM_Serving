"""Fixed IBS/OBS controls and the SDAR adapters for the frozen baseline matrix."""
import types
import numpy as np
import torch
from transformers.cache_utils import DynamicCache
from bdlm_serving.methods.optimus.upstream import load_controls, SelectedResidentCache, COMMIT
from bdlm_serving.methods.optimus.prefix_refresh import plan_prefix_refresh


class FixedOptimus:
    """Fixed IBS/OBS controls, with optional KV-only prefix refresh.

    ``prefix_refresh=k`` is a controlled budget intervention: it may consume
    only physical-Q slack left by the original hard-capped LLaDA2.1 IBS
    planner (``offer - original picked``).  Injected rows are stable, initialized,
    non-dirty prefix rows projected as real Q rows so their K/V is refreshed;
    their logits are discarded and they are never sampled or T2T-edited.
    ``prefix_refresh=0`` is byte-identical to the baseline.
    """
    def __init__(self, model, kind, chunk, mode, *, physical_caps=(),
                 prefix_refresh=0, prefix_order="right"):
        self.caps = tuple(physical_caps)
        if self.caps:
            if (kind != "llada21" or chunk != max(self.caps) or len(self.caps)>16 or
                    any(type(n) is not int or not 1 <= n <= 32 for n in self.caps)):
                raise ValueError("Expected LLaDA B32 decoder caps with chunk=max(caps)")
        elif chunk not in (4, 8, 16, 32):
            raise ValueError("Unknown uncapped Optimus chunk")
        assert mode in ("ibs", "obs")
        if type(prefix_refresh) is not int or prefix_refresh < 0:
            raise ValueError("prefix_refresh must be a non-negative integer")
        if prefix_order not in ("right", "oldest"):
            raise ValueError("prefix_order must be 'right' or 'oldest'")
        if prefix_refresh and (kind != "llada21" or mode != "ibs" or not self.caps):
            raise ValueError("prefix_refresh requires hard-capped LLaDA2.1 IBS")
        self.model, self.kind, self.chunk, self.mode = model, kind, chunk, mode
        self.prefix_refresh, self.prefix_order = prefix_refresh, prefix_order
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
        tick = 0
        dirty = set()
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
                offer = self.caps[tick % len(self.caps)] if self.caps else None
                count, _ = control.fill_processing_buffers(index_buffer, 0, old_mask,
                                                            chunk_size_override=offer)
                if not count:
                    raise RuntimeError("Optimus returned an empty unfinished step")
                candidate = index_buffer[:count].copy()
                if offer is not None:
                    pending = sorted(a-start for a in dirty)
                    if not set(pending).issubset(candidate):
                        raise RuntimeError("Upstream window omitted an unrefreshed write")
                    # Select pending writes first. Remaining seats retain upstream
                    # order, also initializing a partial first prompt block in order.
                    if count > offer:
                        chosen = pending + [int(i) for i in candidate if i+start not in dirty]
                        picked = np.array(sorted(chosen[:offer]),dtype=candidate.dtype)
                    else:
                        picked = candidate
                    count = len(picked)
                else:
                    picked = candidate
                selected = torch.as_tensor(picked, device=x.device)
                positions = selected + start
                if self.mode == "ibs":
                    assert bool((positions < start+32).all())
                extra_abs = np.empty(0, dtype=np.int64)
                eligible_abs = np.empty(0, dtype=np.int64)
                slack = 0
                if self.prefix_refresh:
                    # KV-only refresh: fill physical slack after the original
                    # picked rows.  The helper only returns stable, initialized,
                    # non-dirty prefix rows strictly inside the cached prefix and
                    # strictly earlier than the largest original picked row, so
                    # the cache visible span is unchanged.
                    extra_abs, eligible_abs, slack = plan_prefix_refresh(
                        prompt_len=p, block_start=start,
                        cached_prefix_before=prefix_before,
                        old_mask=old_mask, dirty=dirty,
                        initialized=cache.initialized,
                        original_positions=positions.tolist(), offer=offer,
                        quota=self.prefix_refresh, order=self.prefix_order)
                projected = positions
                if len(extra_abs):
                    # Project the union in one ascending order, then gather the
                    # original rows' logits by absolute position.  Sorting keeps
                    # absolute_positions batch/audit friendly and leaves the
                    # cache visible end equal to max(original picked)+1.
                    extra = torch.as_tensor(extra_abs, device=x.device,
                                            dtype=positions.dtype)
                    projected = torch.sort(torch.cat((positions, extra))).values
                    original_index = torch.searchsorted(projected, positions)
                    assert torch.equal(projected[original_index], positions)
                if self.prefix_refresh:
                    if offer is None:
                        raise RuntimeError("prefix_refresh requires a physical cap")
                    # The physical cap is a hard invariant: original picked rows
                    # plus injected rows may never exceed the offered Q budget.
                    assert len(projected) <= offer
                    assert len(extra_abs) <= min(self.prefix_refresh, slack)
                    assert bool((projected[1:] > projected[:-1]).all())
                logits = forward(projected, "optimus")
                if len(extra_abs):
                    # Only original rows may sample or T2T-write.  Injected rows
                    # contribute K/V only; their logits are never gathered.
                    logits = logits[:, original_index]
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
                if offer is not None:
                    ran = {start+int(i) for i in picked}
                    dirty.difference_update(ran)
                    written = np.flatnonzero((old_mask == 0) & (now_mask != 0)).tolist()
                    written = set(written) | {int(i) for i in deferred if now_mask[i]}
                    dirty.update(start+i for i in written)
                    # Do not freeze uninitialized prompt rows, a carried OBS tail,
                    # or an admission whose new token has not yet entered KV.
                    for i in range(width):
                        if start+i in dirty or start+i not in cache.initialized:
                            cache_mask[i] = 0
                    events[-1].update(offer_tick=tick,physical_cap=offer,
                                      candidate_local=candidate.tolist(),
                                      dirty_before=[start+i for i in pending],
                                      written_absolute=sorted(start+i for i in written))
                    tick += 1
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
                    visible_kv=cache.visible_end, active_width=width,
                    **({
                        "original_q_tokens": int(count),
                        "prefix_refresh_original_absolute": positions.tolist(),
                        "prefix_refresh_q_tokens": int(len(extra_abs)),
                        "prefix_refresh_quota": int(self.prefix_refresh),
                        "prefix_refresh_order": self.prefix_order,
                        "prefix_refresh_slack": int(slack),
                        "prefix_refresh_eligible": eligible_abs.tolist(),
                        "prefix_refresh_added": int(len(extra_abs)),
                        "prefix_refresh_local": (extra_abs - start).tolist(),
                        "prefix_refresh_absolute": extra_abs.tolist(),
                        "prefix_refresh_projection_order": projected.tolist(),
                        "prefix_refresh_visible_end": int(cache.visible_end),
                        "prefix_refresh_visible_span": [0, int(cache.visible_end)],
                    } if self.prefix_refresh else {}))
                assert not bool((positions[edited] < start+ready).any())
                if offer is not None and any(a < start+ready for a in dirty):
                    raise RuntimeError("Retired an unrefreshed write")
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
            "scope": ("upstream candidate plan with a shared hard cap before physical projection; selected-only T2T"
                      if self.caps else "unchanged official prefix/unmask controls; common backend; selected-only T2T"),
            **({"physical_caps":list(self.caps)} if self.caps else {}),
            **({"prefix_refresh": self.prefix_refresh,
                "prefix_order": self.prefix_order,
                "prefix_refresh_scope": ("slack-only KV refresh; injected rows are never sampled or T2T-edited")}
               if self.prefix_refresh else {}),
        }
