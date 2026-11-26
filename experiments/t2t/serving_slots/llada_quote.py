"""Read-only next-tick cost for the pinned AllSlotKernel. Not an offer policy."""
from experiments.t2t.serving_slots.llada_all_slots import (
    AllSlotKernel, BLOCK, OFFERS, PAIR, TopRequestState,
)
from experiments.t2t.serving_slots.llada_spawn import SpawnGatedKernel


def with_quotes(kernel, state, next_budget):
    """Adapt an external (tick, five_prices) callback for kernel.run on state."""
    if not callable(next_budget):
        raise TypeError("Expected an external budget callback")
    def supply(tick):
        if tick.tick != state.offer_tick or tick.block != state.block:
            raise ValueError("Cannot price a different request tick")
        return next_budget(tick, {offer: quote(kernel, state, offer) for offer in sorted(OFFERS)})
    return supply


def quote(kernel, state, offer):
    """Quote one offer on the current, stationary state, without reserving it.

    physical_q excludes prefill_q. mandatory_positions are outstanding writes
    (absolute IDs); required_q is their union with dense protocol rows. phase
    and action describe the next generation event, not the input request phase.
    All supported routes have exact widths, even if the selected IDs are random.
    No estimate of future offers, quality, wall time, or total request cost.
    """
    if type(offer) is not int or offer not in OFFERS:
        raise ValueError(f"Expected an integer offer in {OFFERS}")
    if (type(kernel) not in (AllSlotKernel, SpawnGatedKernel) or not isinstance(state, TopRequestState)
            or state.model_kind != kernel.kind):
        raise ValueError("Quote supports only the pinned AllSlotKernel and its request state")
    phase = state.phase
    start, end = state.block * BLOCK, (state.block + 1) * BLOCK
    prefill = start if phase == "prefill" else 0
    debt = [] if phase == "done" else [
        start + i for i, (w, r) in enumerate(zip(state.last_write, state.last_query, strict=True))
        if w >= 0 and w >= r]

    def result(action, q, dense=()):
        return dict(offer=offer, offer_tick=state.offer_tick, request_phase=phase,
                    phase={"done": "done", "commit_head": "commit"}.get(action, "denoise"),
                    action=action, exact=True, physical_q=q, prefill_q=prefill,
                    mandatory_q=len(debt), mandatory_positions=debt,
                    dense_required_q=len(dense), required_q=len(set(debt).union(dense)))

    if phase == "done":
        return result("done", 0)
    if phase not in ("prefill", "reencode", "commit", "denoise"):
        raise ValueError(f"Unsupported request phase: {phase}")
    if phase != "prefill" and (state.committed_end != start or state.cache.get_seq_length() != start):
        raise ValueError("Committed prefix does not end at head start")

    head = range(start, end)
    if phase == "reencode":
        return result("promote_verify", BLOCK, head)
    tail_debt = sum(p >= end for p in debt)
    limit = state.config.get("max_post_steps", 16)
    head_ids = None
    if phase != "commit" and state.left_post_steps >= limit:
        head_ids = state.x[0, start:end].tolist()
    if phase == "commit" or (head_ids is not None and state.mask_id not in head_ids):
        return result("commit_head", BLOCK + tail_debt, head)

    wants_tail = offer == PAIR and end < state.x.shape[1]
    if wants_tail and type(kernel) is SpawnGatedKernel and not kernel._tail_resident(state):
        wants_tail = kernel.admit_new_tail(state)
    if state.resident is None:
        rows = PAIR if wants_tail else BLOCK
        return result("pair_bootstrap" if wants_tail else "head_bootstrap",
                      rows, range(start, start + rows))
    has_tail = state.resident[0][0].shape[-2] == end + BLOCK
    if wants_tail and not has_tail:
        if kernel.fuse_growth:
            return result("fused_growth", PAIR, range(start, end + BLOCK))
        return result("split_growth", BLOCK + len(debt), range(end, end + BLOCK))

    editable = max(start, state.prompt_length)
    if offer >= BLOCK:
        # _dual_pick gives select_hard budget=B32 per active block. Even when
        # reservations underfill, ordinary backfill reaches every editable row.
        if wants_tail and not state.right_done and (state.right_post_steps < limit or
                state.mask_id in state.x[0, end:end + BLOCK].tolist()):
            return result("pair_denoise", end - editable + BLOCK)
        return result("head_denoise", end - editable + tail_debt)

    if head_ids is None:
        head_ids = state.x[0, start:end].tolist()
    if state.mask_id not in head_ids:
        return result("head_verify", BLOCK + tail_debt, head)
    first = start + head_ids.index(state.mask_id)
    # All non-dirty suffix rows are MASK or ordinary backfill. Never count a
    # dirty suffix row twice, including a token edited back to MASK.
    suffix = end - first - sum(first <= p < end for p in debt)
    base = len(debt) + min(max(0, offer - len(debt)), suffix)
    if base == tail_debt:
        return result("tail_dirty_drain", base)
    pool = first - editable - sum(p < first for p in debt)
    extra = min(kernel.prefix_slots.get(offer, 0), pool, max(0, offer - base))
    return result("head_denoise", base + extra)
