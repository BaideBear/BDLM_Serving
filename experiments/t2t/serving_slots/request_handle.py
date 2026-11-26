"""Serial, batch-one serving adapter. Keep one inference lane per shared model.

The outer scheduler selects a menu entry, never a prescribed sequence. Model
dependencies are imported only by their adapter (the pinned HF stacks differ).
"""
from dataclasses import dataclass
from threading import current_thread
from types import MappingProxyType


@dataclass(frozen=True)
class Price:
    tick: int
    offer: int
    action: str
    main_q: int           # Exact rows in the one budgeted forward.
    mandatory_q: int     # Included in main_q, not an extra charge.
    prefill_q: int       # Pure prompt work inside this advance, if any.
    post_q_max: int      # Conditional paid encoder after the main forward.

    @property
    def paid_q_max(self):
        return self.main_q + self.post_q_max

    @property
    def paid_f_max(self):
        return 1 + int(self.post_q_max > 0)


@dataclass(frozen=True)
class Step:
    tick: int
    offer: int
    action: str
    main_q: int
    prefill_q: int
    post_q: int
    done: bool

    @property
    def paid_q(self):
        return self.main_q + self.post_q

    @property
    def paid_f(self):
        return 1 + int(self.post_q > 0)

    @property
    def total_f(self):
        return self.paid_f + int(self.prefill_q > 0)


class RequestHandle:
    """Use one thread and serial calls per model; no overlapping forwards."""
    def __init__(self, *, opened_prefill_q, prices, advance, output, close):
        self.opened_prefill_q = opened_prefill_q
        self._thread = current_thread()
        self._prices, self._advance, self._output, self._close = prices, advance, output, close
        self._menu, self._result = None, None
        self._tick, self._status = 0, "ready"

    @property
    def status(self):
        return self._status

    def _lane(self):
        if current_thread() is not self._thread:
            raise RuntimeError("Use the owning serial inference lane")

    def menu(self):
        self._lane()
        if self._status != "ready":
            raise RuntimeError(f"Request is {self._status}")
        if self._menu is None:
            options = self._prices(self._tick)
            self._menu = MappingProxyType(options)
        return self._menu

    def advance(self, price):
        self._lane()
        if (self._status != "ready" or self._menu is None or not isinstance(price, Price)
                or self._menu.get(price.offer) is not price):
            raise ValueError("Select a price from this request's current menu")
        self._menu, self._status = None, "advancing"
        try:
            step = self._advance(price.offer)
            if ((step.tick, step.offer, step.action, step.main_q, step.prefill_q) !=
                    (price.tick, price.offer, price.action, price.main_q, price.prefill_q)
                    or step.post_q not in (0, price.post_q_max)):
                raise RuntimeError("Actual forward costs differ from current quote")
            self._tick += 1
            if step.done:
                self._result = tuple(self._output())
                self._release("done")
            else:
                self._status = "ready"
            return step
        except BaseException:
            self._release("failed")
            raise

    def _release(self, status):
        closer = self._close
        self._status = status
        self._menu = self._prices = self._advance = self._output = self._close = None
        if closer is not None:
            closer()

    def close(self):
        self._lane()
        if self._status == "ready":
            self._release("cancelled")
        elif self._status == "advancing":
            raise RuntimeError("Cancel only at a tick boundary")

    def result(self):
        if self._status != "done":
            raise RuntimeError(f"No completed output: {self._status}")
        return self._result


def open_llada(kernel, input_ids, config, gen_length, seed, mask_id, eos_ids):
    """Use the already configured AllSlotKernel/SpawnGatedKernel on this lane."""
    from .llada_all_slots import OFFERS
    from .llada_quote import quote
    state = kernel.create(input_ids, config, gen_length, seed, mask_id, eos_ids)

    def prices(tick):
        options = (quote(kernel, state, q) for q in OFFERS)
        result = {}
        for o in options:
            if o["offer_tick"] != tick:
                raise RuntimeError("LLaDA request clock differs from handle")
            q = o["offer"]
            result[q] = Price(tick, q, o["action"], o["physical_q"], o["mandatory_q"],
                              o["prefill_q"], 0)
        return result

    def advance(q):
        start = len(state.events)
        e = kernel.step(state, q)
        pre = state.events[start:-1]
        if len(pre) > 1 or any(x["phase"] != "prefill" for x in pre):
            raise RuntimeError("Unexpected additional LLaDA forward")
        return Step(e["offer_tick"], e["nominal_q"], e["top_action"], e["physical_q"],
                    sum(x["physical_q"] for x in pre), 0, state.phase == "done")

    return RequestHandle(opened_prefill_q=0, prices=prices, advance=advance,
                         output=lambda: kernel.output(state)[0].tolist(), close=None)


def open_gemma(model, data, plan, *, seed=0, causal=True):
    """Gemma performs pure prompt prefill at open, not on the first advance."""
    from .gemma_unified import start_request
    raw = start_request(model, data, plan, seed=seed, causal=causal, quote_costs=True)

    def prices(tick):
        if raw.tick.tick != tick:
            raise RuntimeError("Gemma request clock differs from handle")
        return {q: Price(tick, q, o.phase, o.decoder_q, o.mandatory_q, 0,
                         o.post_reencode_q_max) for q, o in raw.options.items()}

    def advance(q):
        e = raw.advance(q)
        return Step(e["tick"], e["offer"], e["phase"], e["q"], 0,
                    len(e["encoded"])*256, raw.done)

    return RequestHandle(opened_prefill_q=int(data["input_ids"].shape[1]), prices=prices,
                         advance=advance, output=lambda: raw.result()[0], close=raw.close)
