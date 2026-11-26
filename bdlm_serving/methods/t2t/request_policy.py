"""Translate a scheduler's integer offers into an existing single-block request kernel."""
from .slot_trace import BudgetOffer


SERVICE_TIERS = {"llada21": (4,8,16,32,64), "gemma": (32,64,128,256,512)}
SLOTS = {m: tiers[:-1] for m, tiers in SERVICE_TIERS.items()}
PREFIX = {"llada21": BudgetOffer(16,1,"right"), "gemma": BudgetOffer(128,32,"random")}


def bind_request(model, initial, lookup, *, prefix=False, native_full=True):
    """Return a next_budget callback for exactly one request.

    lookup(RoundInfo) returns a nominal integer Q, or None for the initial slot.
    Full-width execution is explicitly native or sparse. Optional prefix rows
    only fill unused seats, never consume a MASK seat. The kernel owns state,
    mandatory refresh, physical work accounting and the request's selection seed.
    No inference across block boundaries and no mapping of dual-block offers.
    """
    if model not in SLOTS or type(initial) is not int or initial not in SLOTS[model]:
        raise ValueError("Unknown model or unsupported initial single-block slot")
    if type(prefix) is not bool or type(native_full) is not bool or not callable(lookup):
        raise ValueError("Expected booleans and a synchronous request-scoped lookup")
    width = SLOTS[model][-1]
    table = {q:BudgetOffer(q) for q in SLOTS[model]}
    if prefix:
        pref = PREFIX[model]
        table[pref.q] = pref
    if native_full:
        table[width] = BudgetOffer(width,native_full=True)

    def source(info):
        if info.block_width != width:
            raise ValueError("Offer was consumed by a different model width")
        q = lookup(info)
        if q is None:
            q = initial
        if type(q) is not int or q not in table:
            raise ValueError(f"Unsupported {model} single-block offer: {q!r}")
        return table[q]
    return source
