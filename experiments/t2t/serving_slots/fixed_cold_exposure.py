"""Count first admissions before physical block residency and later edit opportunities."""
from collections import Counter
import csv
import json
import sys

from .fixed_family_report import DATA, report
from .llada_all_slots90 import require


def exposure(case, block=32, cap=2048):
    origin = case["prefill_Q"]
    end = origin + cap
    seen_end = origin
    first, reconsidered, changed, refreshed = {}, set(), set(), set()
    for e in case["trace"]:
        rows = e["rows"]
        width = len(rows)
        require(width == e["q"] <= case["q"] and len(set(rows)) == width and rows,
                "Bad physical projection")
        limit = max(seen_end, max(rows)+1)
        require(set(range(seen_end, limit)) == {p for p in rows if p >= seen_end} and
                limit == e["visible_kv"] <= end, "Unverified visible KV")
        seen_end = limit
        full = {p for p in rows if min(end, (p//block+1)*block) <= limit}
        sampled = set(e["sampled"])
        require(sampled <= set(rows) and set(e["admitted"]+e["edited"]+e["revoked"]) <= sampled,
                "Write without a sampled physical row")
        # Strictly later than first admission. Physical refresh alone cannot edit a token.
        old = first.keys() & full
        refreshed.update(old)
        reconsidered.update(old & sampled)
        changed.update(old.intersection(e["edited"]+e["revoked"]))
        for p in e["admitted"]:
            first.setdefault(p, p not in full)
    output = set(range(origin, origin+len(case["output_ids"])))
    require(output <= first.keys() and changed <= reconsidered <= refreshed,
            "Output without an observed admission, or edit without an opportunity")
    partial = {p for p in output if first[p]}
    return Counter(output_positions=len(output), first_before_full=len(partial),
                   later_full_projected=len(partial & refreshed),
                   later_full_sampled=len(partial & reconsidered),
                   later_full_changed=len(partial & changed))


def main():
    indices = (0, 10, 20)
    require(report(DATA, "llada21", "cpu")["protocol_smoke3"], "Incomplete pilot")
    root = DATA/"llada21-fixed-family-v1/cpu/law-1-8"
    records = []
    for q in (4, 8, 16, 32, 64):
        stats = [exposure(json.loads((root/f"{i:02}-q{q}.json").read_bytes())) for i in indices]
        records.append(dict(model="llada21", device="cpu", indices="0,10,20", budget=q,
                            **{k:sum(x[k] for x in stats) for k in stats[0]}))
    writer = csv.DictWriter(sys.stdout, fieldnames=records[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)


if __name__ == "__main__":
    main()
