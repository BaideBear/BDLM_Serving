"""Physical decoder-row limits for the labelled Optimus control adaptation."""
import numpy as np


def parse_caps(text, allowed):
    parts = text.split(",")
    if not 1 <= len(parts) <= 16 or any(s not in {str(q) for q in allowed} for s in parts):
        raise ValueError(f"Invalid Optimus decoder-cap schedule: {text}")
    return tuple(map(int, parts))


def limited_rows(candidate, prior, dirty, cap):
    """Keep the upstream plan untouched unless it actually exceeds the offer."""
    required = np.flatnonzero(dirty)
    if not np.all(np.isin(required, candidate)) or not np.all(prior[required]):
        raise ValueError("An unwritten admission is outside the Optimus replay plan")
    if len(candidate) <= cap:
        return candidate
    if len(required) >= cap:
        return required[:cap]
    remaining = candidate[~dirty[candidate]]
    # Preserve the upstream progress guarantee, then replay its resolved suffix.
    masks = remaining[prior[remaining] == 0]
    order = np.concatenate((required, masks[:1], remaining[prior[remaining] != 0], masks[1:]))
    return np.sort(order[:cap])


def validate_trace(row, block=256):
    mode, tag, *slots = row["policy"].split("_")
    caps = parse_caps(",".join(slots), (4, 8, 16, 32, 64, 128, 256))
    if tag != "cap" or mode not in ("ibs", "obs"):
        raise ValueError("Unknown capped Optimus control")
    events = row["events"]
    trace = row["trace"]
    dec = [q for tag, q in events[1:] if tag == "decoder"]
    if (not trace or events[0][0] != "encoder" or len(trace) != len(dec)
            or any(not (tag == "decoder" or (tag == "encoder" and q == block))
                   for tag, q in events[1:])
            or row["generation_forwards"] != len(events)-1 or
            row["generation_query_rows"] != row["generation_physical_padded_rows"]
            or sum(q for _, q in events[1:]) != row["generation_query_rows"]
            or row["max_q"] != max(q for _, q in events[1:]) or row["max_decoder_q"] != max(dec)):
        raise ValueError("Capped trace and observed physical work disagree")
    dirty, wrote, initialized = set(), set(), set()
    for tick, (s, q) in enumerate(zip(trace, dec)):
        offer = caps[tick % len(caps)]
        picked, admitted = set(s["positions"]), set(s["admitted"])
        start = s["block"]*block
        end = start+block+(max(caps) if mode == "obs" else 0)
        if (s["tick"] != tick or s["offer"] != offer or not 0 < q <= offer
                or s["q"] != q or q != len(picked) or q != len(s["positions"])
                or min(picked) < start or max(picked) >= min(2048,end)
                or len(admitted) != s["accepted"] or len(s["admitted"]) != len(admitted)
                or not admitted <= picked or wrote & admitted or s["proposed_q"] < q
                or s["cross"] != sum(i >= start+block for i in picked)
                or (dirty-picked and (not picked <= dirty or admitted))):
            raise ValueError("Invalid decoder cap, admission, or mandatory replay")
        initialized |= picked
        if (s["visible_end"] != max(picked)+1 or
                not initialized.issuperset(range(start, s["visible_end"]))):
            raise ValueError("Invalid resident KV span")
        dirty = (dirty-picked) | admitted
        wrote |= admitted
        if any(i < start+s["ready"] for i in dirty):
            raise ValueError("Unrefreshed token moved behind the cache frontier")
    if not wrote.issuperset(range(start,start+block)) or any(i < start+block for i in dirty):
        raise ValueError("Unfinished physical block")
    return caps[0] if len(set(caps)) == 1 else None
