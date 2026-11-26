"""Before-forward work options for the batch-one, 256-row Gemma unified loop.

Use generate(..., next_budget=callback, quote_costs=True). callback(tick, options)
receives options before spawn/denoise and returns the externally chosen offer.
Tick alone lacks the acceptance/clock metadata. No token IDs, logits values,
model calls, canvas initialization or random draws are needed for a quote.
"""
from dataclasses import dataclass
from typing import Optional


BLOCK = 256
OFFERS = (32, 64, 128, 256, 512)


@dataclass(frozen=True)
class Quote:
    offer: int
    phase: str
    verify_reason: Optional[str]
    opens_tail: bool
    head_q: int
    tail_sample_q: int
    parked_dirty_q: int
    mandatory_q: int  # Already included in projected widths; never add again.
    prefix_q: int     # Optional seats already included in head_q.
    post_reencode_q_max: int

    @property
    def decoder_q(self):
        return self.head_q + self.tail_sample_q + self.parked_dirty_q

    @property
    def decoder_f(self):
        return 1  # One packed invocation, even for a sampled head and tail.

    @property
    def post_reencode_f_max(self):
        return int(self.post_reencode_q_max != 0)


def exchange_count(masks, slack, clean, quota, limit, tick):
    """Swap only paid nonmandatory unresolved seats, at most one in four."""
    return 0 if tick % 2 else min(limit, masks//4, quota-slack, clean-slack)


def quote_options(live, *, may_open, steps, cap, prefix_slots, choose, exchange_limit=0, tick=0):
    """Return {offer: Quote} for one unchanged pre-decision request state.

    decoder_* is exact. post_reencode_*_max is a separate branch bound: charge
    one encoder of 256 only on full head commit with no EOS and room to continue.
    Prompt prefill is already sunk. Nothing here prices the following tick.
    `choose` must be gemma_unified._choose; the executor owns row selection.
    """
    if (len(live) not in (1, 2) or type(steps) is not int or steps < 1
            or type(cap) is not int or cap < BLOCK or cap % BLOCK
            or type(may_open) is not bool):
        raise ValueError("Expected a live unified request and a whole-canvas cap")
    head = live[0]
    tail = live[1] if len(live) == 2 else None
    for i, s in enumerate(live):
        if (not 0 <= s.age < steps or not 0 <= s.ordinal < cap // BLOCK
                or s.ordinal != head.ordinal + i
                or any(type(r) is not int or not 0 <= r < BLOCK for r in s.dirty)):
            raise ValueError("Invalid resident clock, ordinal or dirty rows")
    room = head.ordinal + 1 < cap // BLOCK
    if may_open and (tail is not None or not room or steps == 1 or not head.age or head.promoted):
        raise ValueError("Invalid pre-spawn eligibility")
    if any(type(k) is not int or k not in OFFERS or type(v) is not int
           or not 0 <= v <= (k if k < BLOCK else 0) for k, v in prefix_slots.items()):
        raise ValueError("Pass normalized prefix quotas from _prefix_policy")

    h = len(head.dirty)
    t = len(tail.dirty) if tail is not None else 0
    forced = ("promotion" if head.promoted else "bootstrap" if head.raw is None else
              "terminal_clock" if head.age == steps - 1 else "sparse_stop" if head.verify else None)
    required = None
    options = {}
    for offer in OFFERS:
        opens = offer == 2 * BLOCK and may_open
        samples_tail = opens or (offer == 2 * BLOCK and tail is not None
                                 and not tail.tentative and tail.age < steps - 1)
        capacity = max(0, offer - t)
        width, optional, reason = BLOCK, 0, None
        if forced == "bootstrap":
            phase = "bootstrap"
        elif forced:
            phase, reason = "verify", forced
        elif offer >= BLOCK:
            phase = "dense"
        elif capacity == 0 and not h:
            phase, width = "tail_dirty_drain", 0
        else:
            if required is None:
                # At full capacity and zero quota _choose returns dirty U unresolved.
                # One acceptance-mask read for the menu, irrespective of prefix mode.
                rows, _, end = choose(head, BLOCK, 0, "zero", seed=0, tick=0)
                required = len(rows)
                clean_prefix = end - sum(r < end for r in head.dirty)
            base = max(h, min(capacity, required))
            optional = min(prefix_slots.get(offer, 0), clean_prefix, max(0, capacity - base))
            width = base + optional
            if exchange_limit and offer == 128:
                optional += exchange_count(base-h, optional, clean_prefix, prefix_slots.get(128,0),
                                           exchange_limit, tick)
            phase = "sparse"
            if not width:
                phase, reason, width = "verify", "empty_selection", BLOCK
        options[offer] = Quote(offer, phase, reason, opens, width,
                               BLOCK if samples_tail else 0, 0 if samples_tail else t,
                               h + t, optional, BLOCK if width == BLOCK and room else 0)
    return options
