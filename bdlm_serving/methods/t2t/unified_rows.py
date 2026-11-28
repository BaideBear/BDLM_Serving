"""Physical Q selection for an ordinary decoder call, independent of checkpoint."""
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Params:
    prefix_share: Fraction = Fraction(1, 8)
    open_at: Fraction = Fraction(1, 4)

    def __post_init__(self):
        for value in (self.prefix_share, self.open_at):
            if type(value) not in (Fraction, int) or not 0 <= value <= 1:
                raise ValueError("Expected a rational fraction in [0, 1]")


@dataclass(frozen=True)
class Block:
    start: int  # First valid generated row, not necessarily aligned to B.
    size: int
    masks: frozenset  # Unresolved positions, not necessarily a literal MASK ID.
    dirty: frozenset  # Published input changes not yet consumed by a decoder call.
    resident: frozenset

    def __post_init__(self):
        if type(self.start) is not int or self.start < 0 or type(self.size) is not int or self.size < 1:
            raise ValueError("Invalid valid-row interval")
        for key in ("masks", "dirty", "resident"):
            rows = frozenset(getattr(self, key))
            if any(type(i) is not int or not self.start <= i < self.start+self.size for i in rows):
                raise ValueError("Expected absolute row IDs inside the block")
            object.__setattr__(self, key, rows)


@dataclass(frozen=True)
class RowDecision:
    rows: tuple
    dirty: tuple
    mask: tuple
    prefix: tuple
    cold: tuple     # Subset of mask, not additional physical Q.
    sampled: tuple
    open_next: bool

    @property
    def drain_only(self):
        return not self.sampled


def plan_rows(q, b, head, next_block=None, *, opened=False, params=Params()):
    """Select actual paid rows. No logits, last offer, model ID or adapter-ranked lists.

    A supplied fresh successor is a candidate, not a request to open it. An
    admitted successor persists across calls, even on a downshift. Model-owned
    terminal verification/re-encoding is a separate paid phase.
    """
    if (type(b) is not int or b < 1 or type(q) is not int or not 0 < q <= 2*b
            or not isinstance(params, Params) or not isinstance(head, Block)
            or head.size > b or type(opened) is not bool):
        raise ValueError("Invalid ordinary decoder offer or head")
    if next_block is None:
        if opened:
            raise ValueError("An open successor must be supplied, including its debt")
    elif (not isinstance(next_block, Block) or next_block.size > b
          or next_block.start != head.start+head.size
          or (not opened and (next_block.dirty or next_block.resident
                              or len(next_block.masks) != next_block.size))):
        raise ValueError("Expected a consecutive successor, initially unvisited and unresolved")

    # Require warm head context for opportunistic early opening, but the 2B
    # offer always admits a valid successor if we can actually project it.
    eligible = next_block is not None and (
        opened or q == 2*b or (bool(head.masks) and len(head.resident) == head.size
                                and len(head.masks) <= params.open_at * head.size))
    blocks = (head, next_block) if eligible else (head,)
    debt = frozenset().union(*(block.dirty for block in blocks))
    d = tuple(sorted(debt)[:q])
    room = q-len(d)
    candidates, prefixes = [], []
    for block in blocks:
        # Head gets first access. Within a block, first visits precede revisits.
        candidates.extend(sorted(block.masks-debt, key=lambda i: (i in block.resident, i)))
        if block.masks:
            frontier = min(block.masks)
            prefixes.extend(sorted((i for i in block.resident-debt if i < frontier), reverse=True))
    n = min(len(prefixes), int(room * params.prefix_share), max(0, room-bool(candidates)))
    p = tuple(sorted(prefixes[:n]))
    m = tuple(sorted(candidates[:room-n]))
    sample = tuple(sorted((*m, *p)))
    rows = tuple(sorted((*d, *sample)))
    resident = frozenset().union(*(block.resident for block in blocks))
    cold = tuple(i for i in m if i not in resident)
    admitted = opened or bool(eligible and rows and rows[-1] >= next_block.start)
    return RowDecision(rows, d, m, p, cold, sample, admitted)
