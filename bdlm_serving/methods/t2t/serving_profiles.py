"""Batch-one request profiles. Configure selection, never the offer sequence."""

from threading import current_thread

from .budget_family import Law


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


class StrictHandle:
    """Drive the same hard-cap executor across ticks. No forecast or load policy."""

    def __init__(self, request, offers, laws, block_size, *, full_head_min_unresolved=None):
        if (len(offers) != len(laws) or len(set(offers)) != len(offers)
                or not all(isinstance(law, Law) for law in laws)):
            raise ValueError("Incomplete fixed-offer profile")
        self._request, self._offers, self._laws = request, tuple(offers), tuple(laws)
        self._block_size, self._owner = block_size, current_thread()
        self._full_head_min_unresolved = full_head_min_unresolved
        self.opened_prefill_q = request.origin
        self._status, self._result = "ready", None

    @property
    def offers(self):
        return self._offers

    @property
    def status(self):
        return self._status

    def _lane(self):
        if current_thread() is not self._owner:
            raise RuntimeError("Use the owning serial inference lane")

    def step(self, q):
        self._lane()
        if self.status != "ready":
            raise RuntimeError(f"Request is {self.status}")
        if type(q) is not int or q not in self.offers:
            raise ValueError(f"Select a supported offer from {self.offers}")
        req = self._request
        # Both strict planners read the request law when selecting positions.
        # Installing a dose has no effect on KV, ordinal, RNG, or the sampler.
        law = self._laws[self.offers.index(q)]
        req.law = law
        self._status = "advancing"
        try:
            if self._full_head_min_unresolved is not None:
                remaining = self._block_size - len(req.rows.accepted)
                req.full_head_at_b = q == self._block_size and remaining >= self._full_head_min_unresolved
            event = req.step(q)
            width = event["q"]
            enc_f, enc_q = event.get("encoder_F", 0), event.get("encoder_Q", 0)
            if (event["offer"] != q or not 0 < width <= q or
                    width != event.get("decoder_Q", width) or event.get("decoder_F", 1) != 1
                    or enc_f not in (0, 1) or enc_q != self._block_size*enc_f):
                raise RuntimeError("Paid work does not match strict offer")
            # Step returns the actual projection ledger, never a menu estimate.
            out = dict(event, prefix_share=(law.num, law.den), decoder_F=1,
                       decoder_Q=width, history_Q=width, encoder_F=enc_f, encoder_Q=enc_q,
                       generation_F=1+enc_f, physical_generation_Q=width+enc_q)
            if req.status == "done":
                self._result = tuple(req.tokens)
                self._request = None
            elif req.status != "ready":
                raise RuntimeError(f"Unexpected request status: {req.status}")
            self._status = req.status
            return out
        except BaseException:
            self._status, self._request = "failed", None
            raise

    def result(self):
        if self.status != "done":
            raise RuntimeError(f"No completed output: {self.status}")
        return self._result

    def close(self):
        self._lane()
        if self.status == "ready":
            self._status, self._request = "cancelled", None
        elif self.status == "advancing":
            raise RuntimeError("Cancel only at a tick boundary")


def open_strict_gemma(model, data, *, seed, initial_offer, max_new_tokens=2048,
                      profile="fixed90_candidate"):
    """Half gate at 2B, no subtop borrow; optional guarded close and full-head policy."""
    from experiments.t2t.serving_slots.gemma_fixed_dual import AblationRequest, B, TIERS
    if type(initial_offer) is not int or initial_offer not in TIERS:
        raise ValueError("Invalid initial Gemma offer")
    engine = AblationRequest
    if profile=="dense_stop_joint_verified_eos":
        from experiments.t2t.serving_slots.gemma_verified_eos import SwitchableVerifiedEOSRequest
        engine = SwitchableVerifiedEOSRequest
    single_top = profile in ("dense_stop_joint_single512","dense_stop_joint_verified_eos")
    joint = single_top or profile=="dense_stop_joint_verify"
    promotion = joint or profile=="dense_stop_promotion_fused"
    stop = promotion or profile=="dense_b_stop_fused"
    if not stop and profile not in ("fixed90_candidate", "reference", "fixed90_fused", "fixed90_dense_fused","phase_high_fused"):
        raise ValueError("Unknown strict Gemma profile")
    # Only Q128 switches doses: the 90 paired tasks support 1/4 here.
    # Other sparse tiers retain the reference; dense top calls bypass the quota.
    laws = tuple(Law(1, 4) if profile != "reference" and c == B//2 else Law()
                 for c in TIERS)
    raw = engine(model, data, q=initial_offer, seed=seed, ablation="00",
                          max_new_tokens=max_new_tokens, law=laws[TIERS.index(initial_offer)],
                          top_gate="half", prefix_order="right",
                          fuse_first_verify=stop or profile in ("fixed90_fused","fixed90_dense_fused","phase_high_fused"),
                          fullstop=True if stop else None, stop_at_single_b_only=stop,
                          sample_full_rebuild=promotion)
    raw.joint_verify = joint
    raw.stop_at_single_2b = single_top
    raw.full_head_at_b = stop or profile == "fixed90_dense_fused"
    return StrictHandle(raw, TIERS, laws, B, full_head_min_unresolved=B//2 if profile=="phase_high_fused" else None)


def open_strict_llada(model, input_ids, *, seed, mask_id, eos_ids, max_new_tokens=2048,
                      coverage="sparse", top_gate="half", profile=None):
    """Hard-cap LLaDA family; explicit profiles bind the fixed arms across tiers."""
    from .llada_budget_request import B, LLaDABudgetRequest
    offers = (4, 8, 16, 32, 64)
    opts = {}
    engine = LLaDABudgetRequest
    if profile is not None:
        top_repeat = profile=="quarter_window_repeat64"
        window = top_repeat or profile=="quarter_edit_window"
        if not (window or profile in ("quarter_fullwrite","quarter_edit_warm")) or (coverage, top_gate) != ("sparse", "half"):
            raise ValueError("Unknown LLaDA profile or conflicting explicit settings")
        coverage, top_gate = "head", "quarter"
        editable = window or profile=="quarter_edit_warm"
        if window:
            from experiments.t2t.serving_slots.llada_window_switch import SwitchableWindowRequest, TopAgreementWindowRequest
            engine = TopAgreementWindowRequest if top_repeat else SwitchableWindowRequest
        opts = dict(close_policy="edit_once" if editable else "full_write", borrow_at_half=not window,
                    warm_head_sample=editable)
    raw = engine(model, input_ids, seed=seed, mask_id=mask_id, eos_ids=eos_ids,
                             max_new_tokens=max_new_tokens, law=Law(), confidence_threshold=.9,
                             editing_threshold=.5, switchable=True, top_gate=top_gate,
                             coverage=coverage, lower_borrow=False, **opts)
    return StrictHandle(raw, offers, (Law(),)*len(offers), B)
