"""Replay row identities at projection, sampling and sealing in recorded LLaDA W2."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from audit_raw import B, EOS, MASK, require, verify


def changes(slot):
    return {i: (a, b) for i, (a, b) in enumerate(zip(slot["ids"], slot["new"])) if a != b}


def kind(a, b):
    require(b != MASK and a != b, "unexpected revoke/no-op")
    return "M2T" if a == MASK else "T2T"


def account(r):
    base = verify(r)
    prior, visits, done = {}, Counter(), set()
    c = Counter()
    # pending values are absolute-position -> (input_id, sampled_id).
    retired, written = set(), Counter()
    for ti, tick in enumerate(r["ticks"]):
        cp = tick["commit"]
        slots = tick["slots"]
        require(tick["physical_width"] == tick["written_KV"] and
                tick["physical_width"] == (B*len(slots) if r["arm"] == "active_w2" else 2*B),
                "geometry differs from paid projections")
        if prior:
            require(all(tick["head"] <= p < tick["head"]+len(slots)*B for p in prior),
                    "unaccounted earlier write")
        next_pending = {}
        for i, s in enumerate(slots):
            g = cp+i
            require(g not in retired, "re-opened retired block")
            lo = tick["head"] + i*B
            for k, tok in enumerate(s["ids"]):
                debit = prior.pop(lo+k, None)
                if debit is not None:
                    require(debit[1] == tok, "paid a different token version")
                    c[f"paid_{kind(*debit)}"] += 1
                    c["paid_in_successor_slot" if i else "paid_in_head_slot"] += 1
            for k, debit in changes(s).items():
                written[kind(*debit)] += 1
                next_pending[lo+k] = debit
            visits[g] += 1
            if s["fin"] or visits[g] >= B+16+1:
                done.add(g)
        require(not prior, "previous write was never projected or retired")
        closed, eos = 0, False
        while closed < len(slots) and cp+closed in done:
            s = slots[closed]
            g, lo = cp+closed, tick["head"] + closed*B
            dirty = {p: next_pending.pop(p) for p in list(next_pending) if lo <= p < lo+B}
            for debit in dirty.values():
                c[f"sealed_unpaid_{kind(*debit)}"] += 1
            c["sealed_slots"] += 1
            c["sealed_slots_with_unpaid_writes"] += bool(dirty)
            retired.add(g)
            closed += 1
            eos = EOS in s["new"]
            if eos:
                break
        terminal = ti == len(r["ticks"])-1
        if closed:
            c["forwards_with_seal"] += 1
            c["forwards_with_seal_and_other_slot_write"] += bool(next_pending)
            c["forwards_with_seal_and_surviving_other_slot_write"] += bool(next_pending) and not terminal
            c["forwards_with_seal_and_discarded_other_slot_write"] += bool(next_pending) and terminal
        if not any(changes(s) for s in slots):
            c["forwards_without_any_write"] += 1
            c["forwards_without_any_write_with_seal"] += bool(closed)
            c["forwards_without_any_write_with_one_open"] += len(slots) == 1
        if terminal:
            require(closed > 0 and (eos or closed == len(slots)), "unaccounted terminal")
            for debit in next_pending.values():
                require(eos, "discard not explained by head EOS")
                c[f"discarded_unpaid_{kind(*debit)}"] += 1
            c["discarded_open_slots_after_eos"] += len(slots)-closed
        else:
            require(not eos and r["ticks"][ti+1]["commit"] == cp+closed, "wrong seal boundary")
        prior = {} if terminal else next_pending
    for k, ref in (("M2T","actual_m2t_writes"), ("T2T","actual_t2t_writes")):
        c[f"writes_{k}"] = written[k]
        require(written[k] == base[ref] ==
                c[f"paid_{k}"]+c[f"sealed_unpaid_{k}"]+c[f"discarded_unpaid_{k}"],
                "row-version balance does not close")
    c["requests"] = 1
    c["F"] = len(r["ticks"])
    c["Q"] = sum(t["physical_width"] for t in r["ticks"])
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    receipts = {x["index"]:x for x in map(json.loads,(directory/"compact.ndjson").read_bytes().splitlines())}
    require(set(receipts) == set(range(90)), "incomplete frozen cohort")
    output = {}
    for arm in ("flow_w2", "active_w2"):
        seal, cohorts = sha256(), {k: Counter() for k in ("original30","additional60","all90")}
        for i in range(90):
            raw = (directory/f"{i:02d}-{arm}.json").read_bytes()
            r = json.loads(raw)
            require((r["index"],r["arm"])==(i,arm), "request identity")
            seal.update(sha256(raw).digest())
            one, ref = account(r), receipts[i][arm]
            require((one["F"], one["Q"]) == (ref["F"], ref["Q"]), "forward bill changed")
            cohorts["original30" if i<30 else "additional60"].update(one)
            cohorts["all90"].update(one)
        output[arm] = dict(ordered_raw_hash=seal.hexdigest(), **cohorts)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
