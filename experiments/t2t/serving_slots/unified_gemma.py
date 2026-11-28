"""Opt-in Gemma B256 executor of the common LLaDA/Gemma row planner.

Use ``create_request(model, data, q=32, seed=..., meter=meter)`` and
``req.step(offer)``. Only ordinary denoising/admission is common. The strict
selected-row Gemma sampler, verified argmax, and encoder are checkpoint-specific.
There is no IBS fallback, dense shortcut, native-stop claim or public profile.
RequestMeter counts requests, not calls, canvases, arms, or offers.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache

from bdlm_serving.methods.t2t import unified_rows as common


B = 256
FULL = frozenset(range(B))
OFFERS = (32, 64, 128, 256, 512)


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


@dataclass
class RequestMeter:
    started: int = 0
    completed: int = 0
    failed: int = 0

    @property
    def in_flight(self):
        return self.started - self.completed - self.failed


def block_view(base, canvas, live, *, sampling):
    """Absolute decoder positions. SC is not proof of KV residency this epoch."""
    span = frozenset(range(base, base + B))
    if canvas is None:
        return common.Block(base, B, span, frozenset(), frozenset())
    rows = canvas.rows
    resident = span.intersection(live)
    check(resident == {base + i for i in rows.initialized},
          "Sampler coverage disagrees with epoch KV; rebuild before ordinary planning")
    masks = (frozenset(base + i for i in FULL - rows.accepted)
             if sampling and canvas.can_sample else frozenset())
    return common.Block(base, B, masks, frozenset(base + i for i in rows.dirty), resident)


class _Unified:
    """Mixin uses only the existing projection, sampler, writeback and commit transaction."""

    def __init__(self, model, data, *, q, seed, max_new_tokens=2048,
                 params=common.Params(), meter=None):
        if not isinstance(params, common.Params) or (meter is not None and
                                                      not isinstance(meter, RequestMeter)):
            raise ValueError("Expected shared planner Params and a RequestMeter")
        self.params = params
        self.meter = RequestMeter() if meter is None else meter
        self.meter.started += 1
        self.encoder_receipts = []
        self.last_attempt = None
        self.decoder_F = self.decoder_Q = 0
        try:
            super().__init__(model, data, q=q, seed=seed, max_new_tokens=max_new_tokens)
        except BaseException:
            self.meter.failed += 1
            raise

    def _encode(self, length, *, first):
        # Includes pure prompt prefill; do not include it in generation F or TE.
        from bdlm_serving.runtime.diffusiongemma import observe
        with observe(self.model) as physical:
            super()._encode(length, first=first)
        check(physical == [("encoder", length)], "Incorrect physical encoder charge")
        self.encoder_receipts.append(dict(prefill=first, F=1, Q=length))

    def completed_bill(self, special_ids):
        """Pass the frozen scorer's special IDs; incomplete requests have no TPF/TE."""
        check(self.status == "done", "Cannot score a partial or failed request")
        n = sum(i not in special_ids for i in self.tokens)
        enc = [e for e in self.encoder_receipts if not e["prefill"]]
        f = self.decoder_F + len(enc)
        return dict(requests=1, N=n, F=f, decoder_F=self.decoder_F, decoder_Q=self.decoder_Q,
                    encoder_F=len(enc), encoder_Q=sum(e["Q"] for e in enc), prefill_Q=self.origin,
                    TPF=n/f, TE=n/self.decoder_Q)

    def step(self, q=None):
        if self.status != "ready":
            raise RuntimeError(f"Request is {self.status}")
        q = self.fixed_q if q is None else q
        if type(q) is not int or q not in OFFERS:
            raise ValueError("Unsupported external Gemma decoder offer")
        try:
            event = super().step(q)  # calls only our _advance; parent poisons on failure
            self.meter.completed += int(self.status == "done")
            return event
        except BaseException:
            self.status = "failed"
            self.meter.failed += 1
            raise

    def _advance(self, q):
        head = self._head_canvas()
        base = self.origin + self.ordinal * B
        states = [head] if self.tail is None else [head, self.tail]
        opened = self.tail is not None
        if opened:
            check(self.tail.ordinal == self.ordinal + 1, "Nonconsecutive Gemma successor")
        live = set(self.bank.live)

        if self.rebuild:
            check(not opened, "Promotion must reset old decoder KV")
            missing = sorted({base + i for i in head.rows.initialized} - live)
            check(missing, "Spurious epoch rebuild")
            return self._execute(states, q, "rebuild", tuple(missing[:q]), opened=opened)

        sampling = head.can_sample
        h = block_view(base, head, live, sampling=sampling)
        n = None
        if opened:
            n = block_view(base+B, self.tail, live, sampling=sampling)
        elif sampling and self.ordinal + 1 < self.max_new_tokens // B:
            # Only a view. Do not allocate a tail, initialize RNG or open a gate here.
            n = block_view(base+B, None, live, sampling=True)

        if not sampling and not h.dirty and (n is None or not n.dirty):
            check(head.rows.initialized == FULL and len(head.sampler.sc) == B,
                  "Gemma clock closed before complete sampler coverage")
            check(0 <= self.cursor < B, "Invalid ordered terminal cursor")
            pos = tuple(range(base+self.cursor, base+min(B, self.cursor+q)))
            return self._execute(states, q, "verify", pos, opened=opened)

        # Even terminal writebacks use shared debt selection (no further samples).
        out = common.plan_rows(q, B, h, n, opened=opened, params=self.params)
        pos, sampled = tuple(out.rows), tuple(out.sampled)
        check(0 < len(pos) <= q and pos == tuple(sorted(set(pos))) and
              sampled == tuple(sorted(set(out.mask) | set(out.prefix))) and
              set(pos) == set(out.dirty) | set(sampled) and not set(out.dirty) & set(sampled) and
              set(out.cold) == set(out.mask)-live and out.drain_only == (not sampled),
              "Shared planner returned unexecutable physical rows or sampler slice")
        check(out.open_next == (opened or any(p >= base+B for p in pos)),
              "Invalid shared successor admission")
        if out.open_next and not opened:
            check(n is not None, "Planner admitted a nonexistent successor")
            self.tail = self._spawn()
            states.append(self.tail)
        check(all(base <= p < base+B*len(states) for p in pos), "Rows outside admitted canvases")
        return self._execute(states, q, "ordinary", pos, decision=out, opened=opened)

    def _head_canvas(self):
        from .gemma_fixed_dual import Canvas
        return Canvas(self.ordinal, self.rows, self.sampler)

    def _execute(self, states, q, route, pos, *, decision=None, opened):
        """The only decoder dispatch, including mandatory calls. Fail on any widening."""
        from .gemma_unified_audit import projectors
        sample, prefix, cold = ((), (), ()) if decision is None else (
            decision.sampled, decision.prefix, decision.cold)
        enc_start = len(self.encoder_receipts)
        with projectors(self.model) as (widths, positions):
            # Keep actual pre-hook observations, even when a downstream layer fails.
            self.last_attempt = dict(route=route, offer=q, rows=pos,
                                     measured_positions=positions, projection_widths=widths)
            event = self._apply(states, pos, set(sample), prefix, cold, q, route)
        check(positions == [list(pos)] and len(widths) == 85 and
              all(w == [len(pos)] for w in widths.values()) and
              event["positions"] == list(pos) and event["decoder_F"] == 1 and
              event["decoder_Q"] == len(pos), "Physical decoder projection differs from paid rows")
        actual = tuple(self.origin + p["ordinal"]*B + i
                       for p in event["parts"] for i in p["sampled"])
        check(actual == tuple(sample), "Runtime added or dropped a planner sample")
        check(all(p["age_after"]-p["age_before"] == int(bool(p["sampled"]))
                  for p in event["parts"]), "Wrong number of sampler ticks")
        enc = self.encoder_receipts[enc_start:]
        check(len(enc) == event["encoder_F"] and sum(c["Q"] for c in enc) == event["encoder_Q"],
              "Physical encoder charge mismatch")
        event.update(shared_plan=None if decision is None else {
            **asdict(decision), "drain_only": decision.drain_only},
            opened_before=opened, sampled_absolute=list(actual),
            physical_decoder_positions=positions[0], projection_streams=len(widths))
        self.decoder_F += 1
        self.decoder_Q += len(pos)
        return event


@lru_cache(maxsize=1)
def request_class():
    from .gemma_fixed_dual import AblationRequest

    class UnifiedGemmaRequest(_Unified, AblationRequest):
        pass

    return UnifiedGemmaRequest


def create_request(model, data, **kwargs):
    """Construct exactly one batch-one request; no checkpoint load or implicit loop."""
    return request_class()(model, data, **kwargs)
