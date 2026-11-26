"""Batch-one request profiles. Configure selection, never the offer sequence."""


def gemma_plan(base, profile="economy"):
    if base.get("block_size") != 256:
        raise ValueError("Profiles require the native 256-row Gemma canvas")
    if profile not in ("economy", "q128_prefix_candidate"):
        raise ValueError(f"Unknown Gemma request profile: {profile}")
    candidate = profile == "q128_prefix_candidate"
    return dict(base, flowblock_spawn_threshold=.8, exchange_limit=0,
                prefix_slots={128:32} if candidate else {},
                prefix_mode="random" if candidate else "zero")


def open_gemma(model, data, base_plan, *, seed, profile="economy"):
    """Return a serial request handle with menu()/advance(current_price).

    q128_prefix_candidate is a fixed-slot quality candidate, not a guarantee
    on mixed or previously unseen load paths. No profile sets the next offer.
    """
    from experiments.t2t.serving_slots.request_handle import open_gemma as open_handle
    return open_handle(model,data,gemma_plan(base_plan,profile),seed=seed,causal=True)


def open_llada(model, input_ids, config, *, gen_length, seed, mask_id, eos_ids,
               profile="economy"):
    """Use a model with install_llada_rectangular_mask(..., allow_selected_cache=True).

    economy opens an eligible second block on Q64; progress80_candidate
    first requires one observed head step that clears >=80% of its MASKs.
    Neither profile opens future attention edges or selects the next offer.
    """
    from experiments.t2t.serving_slots.llada_spawn import SpawnGatedKernel
    from experiments.t2t.serving_slots.request_handle import open_llada as open_handle
    if profile not in ("economy", "progress80_candidate"):
        raise ValueError(f"Unknown LLaDA request profile: {profile}")
    kernel = SpawnGatedKernel(model, spawn_threshold=.8 if profile=="progress80_candidate" else None,
                              fuse_growth=True, prefix_mode="zero")
    return open_handle(kernel,input_ids,config,gen_length,seed,mask_id,eos_ids)
