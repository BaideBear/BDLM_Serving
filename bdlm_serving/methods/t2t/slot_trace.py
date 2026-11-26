"""Repeat an external sequence of Q offers across one request's selection rounds."""


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
