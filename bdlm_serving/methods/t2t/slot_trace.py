"""Request-scoped budget offers. A fixed trace is one possible external source."""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RoundInfo:
    tick: int
    block_start: int
    block_width: int
    masks: int
    mandatory: int


@dataclass(frozen=True)
class BudgetOffer:
    q: int
    prefix: int = 0
    selector: str = "right"

    def plan(self, width):
        if (type(self.q) is not int or type(self.prefix) is not int
                or not 0 <= self.prefix < self.q <= width
                or self.selector not in ("right", "random", "oldest")):
            raise ValueError("Require 0 <= prefix < Q <= physical width and a known selector")
        return self.q, self.prefix, ("hard_prefix_slack_" + self.selector
                                     if self.prefix else "hard_prefix_zero")


BudgetSource = Callable[[RoundInfo], int | BudgetOffer | None]


def request_source(slots, source, policy):
    if source is not None:
        if slots or policy != "hard_prefix_zero" or not callable(source):
            raise ValueError("Use one external budget source with the K0 fallback policy")
        return source
    return (lambda info: slots[info.tick % len(slots)]) if slots else None


def resolve(source, info, fallback):
    value = source(info)
    if value is None:
        value = fallback
    if type(value) is int:
        value = BudgetOffer(value)
    if not isinstance(value, BudgetOffer):
        raise ValueError("Budget source must return integer Q, BudgetOffer, or None for fallback")
    return value.plan(info.block_width)


def slots_for(policy, width, initial, fresh, mask):
    if not policy.startswith("slottrace_"):
        return ()
    fields = policy.removeprefix("slottrace_").split("_")
    if not 1 <= len(fields) <= 16 or not all(x.isascii() and x.isdecimal() for x in fields):
        raise ValueError("Expected slottrace_<Q>[_<Q>...] with up to 16 slots")
    slots = tuple(int(x) for x in fields)
    if (any(str(q) != x or not 1 <= q <= width for q,x in zip(slots, fields))
            or slots[0] != initial or fresh != 0 or mask != initial):
        raise ValueError("Single-block external slot trace must start at its declared K0 slot")
    return slots
