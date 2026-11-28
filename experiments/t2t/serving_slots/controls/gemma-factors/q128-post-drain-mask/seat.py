"""Only the post-drain Live row selection is modified."""
from hashlib import sha256
import json
from weakref import ref

B, Q = 256, 128
FULL = frozenset(range(B))


def check(ok, why):
    if not ok:
        raise RuntimeError(why)


def digest(obj):
    return sha256(json.dumps(obj, separators=(",", ":")).encode()).hexdigest()


def choose(rows, sample, prefix, debt, unresolved, resident):
    """Inputs are local row sets on a fully built head."""
    donors = prefix - debt
    receivers = (unresolved & resident) - rows - debt
    if not donors or not receivers:
        return None
    frontier = min(unresolved)
    check(donors <= (sample & resident) and max(donors) < frontier, "Nonoptional prefix")
    if frontier not in sample:
        return None
    return min(donors), min(receivers)


def install(raw, enabled, swaps):
    apply = type(raw)._apply
    get = ref(raw)
    previous = None
    used = set()

    def wrapped(states, pos, sample, prefix, cold, offer, route):
        nonlocal previous
        req = get()
        head, = states
        h = head.ordinal
        base = req.origin + B*h
        local = lambda xs: {x-base for x in xs}
        rows, selected, pref = map(local, (pos, sample, prefix))
        debt = set(head.rows.dirty)
        warm = head.rows.initialized == FULL
        check(offer == Q and req.tail is None and not req.rebuild
              and rows <= FULL and debt <= rows, "Outside single-head Q128 contract")
        qualifying_drain = (warm and route == "drain" and head.can_sample
                            and len(rows) == Q and rows == debt and not sample and not prefix)
        key = (h, req.cache.get_seq_length())
        note = None
        if (previous == key and h not in used and warm and route == "sparse"
                and not cold and selected and pref):
            check(rows == selected and not debt, "Not the next ordinary warm sample")
            unresolved = FULL - head.rows.accepted
            visible = tuple(req.bank.live)
            check(set(visible) == {base+i for i in FULL} and key[1] == base
                  and req.bank.causal, "Changed KV domain or encoder epoch")
            pick = choose(rows, selected, pref, debt, unresolved, FULL)
            if pick:
                drop, take = pick
                used.add(h)
                trial = (rows - {drop}) | {take}
                check(len(trial) == len(rows) == Q and min(unresolved) in trial,
                      "Width or frontier changed")
                note = dict(tick=req.tick, head=h, previous_tick=req.tick-1, drop=drop, take=take,
                            frontier=min(unresolved), width=len(rows), encoder_end=key[1],
                            visible_start=visible[0], visible_end=visible[-1]+1,
                            stock_rightmost_query=max(rows), trial_rightmost_query=max(trial),
                            debt=sorted(debt), age=head.sampler.age,
                            state_sha256=digest([sorted(rows), sorted(pref), sorted(unresolved),
                                                head.sampler.ids[0].tolist(),
                                                head.sampler.rng.get_state().tolist()]))
                if enabled:
                    pos = tuple(base+i for i in sorted(trial))
                    sample = set(pos)
                    prefix = tuple(x for x in prefix if x != base+drop)
        event = apply(req, states, pos, sample, prefix, cold, offer, route)
        previous = key if qualifying_drain else None
        if note is not None:
            p, = event["parts"]
            check(tuple(req.bank.live) == visible and event["positions"] == list(pos)
                  and set(p["sampled"]) == local(sample) and set(p["prefix"]) == local(prefix)
                  and p["age_after"] == note["age"] + 1 and not event["encoded"], "Bad swap receipt")
            swaps.append(note)
        return event

    raw._apply = wrapped
