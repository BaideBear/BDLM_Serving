"""A single, deterministic query-row policy. No tensors, logits, or future offers."""
from dataclasses import dataclass
from bdlm_serving.methods.t2t.prefix_coverage import spread_prefix


@dataclass(frozen=True)
class Law:
    num: int = 1
    den: int = 8

    def __post_init__(self):
        if type(self.num) is not int or type(self.den) is not int or not 0 <= self.num < self.den:
            raise ValueError("Expected a prefix share in [0,1)")


@dataclass(frozen=True)
class Plan:
    rows: tuple
    dirty: tuple
    cold: tuple
    mask: tuple
    prefix: tuple
    drain_only: bool


def choose(q, b, mandatory, unresolved, prefix, *, law=Law(), cold=(),
           prefix_identity="right"):
    """Plan one call on the adapter's eligible physical blocks, using absolute IDs.

    Adapters supply eligible unresolved/cold rows and rank clean prefix rows best
    first. Dirty debt is not required to fit after an external downshift. Pure
    drains never advance sampling. A prefix never creates a sample by itself.
    ``spread`` chooses equal-mass bin endpoints from the same eligible pool after
    the caller-owned dose has been computed; it does not widen visibility.
    """
    if type(q) is not int or type(b) is not int or not 0 < q <= 2*b or not isinstance(law, Law):
        raise ValueError("Invalid budget, block width or allocation law")
    if prefix_identity not in ("right", "spread"):
        raise ValueError("Invalid prefix identity")
    dirty, mask, cold, prefix = set(mandatory), set(unresolved), set(cold), tuple(prefix)
    pool = set(prefix)
    if (any(type(i) is not int or i < 0 for i in dirty | mask | cold | pool)
            or len(pool) != len(prefix) or pool & (dirty | mask | cold) or cold & dirty):
        raise ValueError("Invalid row identity or overlapping prefix/cold debt")
    ordered = lambda x: tuple(sorted(x))
    d = ordered(dirty)[:q]
    room = q-len(d)
    c, m, p = (), (), ()
    if room and cold:
        c = ordered(cold)[:room]
    elif room and (mask or dirty) and not cold:
        candidates = ordered(mask-dirty)
        cap = min(room, b)*law.num//law.den
        # Leave at least one clean unresolved row whenever there is one.
        n = min(len(prefix), cap, room-bool(candidates))
        if mask:  # No discretionary projection when only outstanding writes remain.
            p = (tuple(spread_prefix(prefix, n)) if prefix_identity == "spread"
                 else ordered(prefix[:n]))
        m = candidates[:room-len(p)]
    return Plan(ordered((*d, *c, *m, *p)), d, c, m, p, bool(d) and not (c or m or p)
                and (not room or not mask))
