"""Count actual IBS decoder calls that omit previously initialized right-hand keys."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def measure(arm, budget, block_size=256):
    seen, resolved, debt = set(), set(), set()
    counts, events = Counter(), []
    block = None
    ticks = list(zip(arm["trace"], arm["decoder"], arm["states"], strict=True))
    assert len(ticks) == arm["bill"]["D"]
    for index, (tick, decoder, state) in enumerate(ticks):
        if tick["block"] != block:
            seen, resolved, debt = set(), set(), set()
            block = tick["block"]
        pos = set(tick["positions"])
        assert 0 < len(pos) == tick["q"] <= budget
        assert len(decoder["ids"]) == len(pos) and tick["visible_end"] == max(pos)+1
        assert all(r // block_size == block for r in pos)
        seen.update(pos)
        invisible = {r for r in seen if r >= tick["visible_end"]}
        kind = "sampling" if decoder["sample"] is not None else "payment"
        counts[kind] += 1
        if invisible:
            # Rejected Gemma rows may be re-noised without admission debt.
            # Only already resolved rows with paid IDs are classified as clean.
            clean = len(invisible & (resolved-debt))
            events.append(dict(tick=index, block=block, kind=kind, rows=len(invisible),
                               paid_resolved=clean, pending=len(invisible & debt),
                               unresolved=len(invisible-resolved), last_decoder=index==len(ticks)-1))
        resolved.update(tick["admitted"])
        debt = set(state["debt"])
    return counts, events


def audit(raw, budget):
    totals, events, cases, digest, done = Counter(), [], set(), hashlib.sha256(), False
    with raw.open("rb") as f:
        for line in f:
            digest.update(line)
            if not line.startswith(b"RECEIPT "):
                continue
            rec = json.loads(line[8:])
            if rec["kind"] == "complete":
                done = True
            if rec["kind"] != "case":
                continue
            i = rec["index"]
            if i in cases:
                raise ValueError("duplicate input")
            cases.add(i)
            c, e = measure(rec["arms"]["paid"], budget)
            totals.update(c)
            events.extend(dict(index=i, **event) for event in e)
    if cases != set(range(90)) or not done:
        raise ValueError("not a complete frozen90 source")
    return dict(fixed_q=budget, requests=len(cases), decoder_calls=dict(totals),
                affected_requests=len({e["index"] for e in events}), events=events,
                receipt_folder=raw.parent.name, sha256=digest.hexdigest())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args = p.parse_args()
    paths = {32:"gemma-ibs-certified-eos-a0-20261128T100436Z",
             64:"gemma-ibs-certified-eos-a0-q64-20261128T110532Z",
             128:"gemma-ibs-certified-eos-a0-q128-20261128T110532Z"}
    print(json.dumps([audit(args.root/name/"raw.log", q) for q,name in paths.items()],indent=2))
