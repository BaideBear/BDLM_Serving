"""Request-scoped budget offers. A fixed trace is one possible external source."""
from dataclasses import dataclass
import re
from typing import Callable


NATIVE_FULL = "native_full"


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
    native_full: bool = False

    def plan(self, width):
        if (type(self.q) is not int or type(self.prefix) is not int
                or not 0 <= self.prefix < self.q <= width
                or self.selector not in ("right", "random", "oldest")):
            raise ValueError("Require 0 <= prefix < Q <= physical width and a known selector")
        if type(self.native_full) is not bool:
            raise ValueError("native_full must be boolean")
        if self.native_full:
            if (self.q, self.prefix, self.selector) != (width, 0, "right"):
                raise ValueError("Native offer requires a complete canvas with no prefix quota")
            return self.q, 0, NATIVE_FULL
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


def parse_offer(field, width):
    if re.fullmatch(r"[1-9][0-9]*n", field):
        value = BudgetOffer(int(field[:-1]), native_full=True)
        value.plan(width)
        return value
    match = re.fullmatch(r"([1-9][0-9]*)(?:([ruo])([1-9][0-9]*))?", field)
    if not match:
        raise ValueError("Expected Q or Q{r|u|o}K for right/uniform/oldest slack")
    q, mode, k = match.groups()
    value = BudgetOffer(int(q), int(k or 0), {None:"right","r":"right","u":"random","o":"oldest"}[mode])
    value.plan(width)
    return value


def slots_for(policy, width, initial, fresh, mask):
    if not policy.startswith("slottrace_"):
        return ()
    fields = policy.removeprefix("slottrace_").split("_")
    if not 1 <= len(fields) <= 16:
        raise ValueError("Expected slottrace_<Q>[_<Q>...] with up to 16 slots")
    slots = tuple(parse_offer(x,width) for x in fields)
    if slots[0].q != initial or fresh != 0 or mask != initial - slots[0].prefix:
        raise ValueError("Single-block trace must start at its declared Q and prefix quota")
    return slots
