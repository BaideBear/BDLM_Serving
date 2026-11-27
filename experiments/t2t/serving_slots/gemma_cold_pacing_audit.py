"""Independent replay of first visits, mandatory refreshes and phase costs. No torch."""
from collections import Counter

B = 256
TIERS = (32, 64, 128)
QUAL = (0, 10, 20)
PHASES = ("cold", "warm", "closing_drain", "verify")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def replay(trace, q, pace):
    all_rows = set(range(B))
    totals = Counter()
    blocks, head, seen, debt, age, verified = [], -1, set(), set(), 0, 0
    for tick, e in enumerate(trace):
        if e["ordinal"] != head:
            require(e["ordinal"] == head+1 and (head == -1 or verified == B and not debt),
                    "Unsealed predecessor")
            head, seen, debt, age, verified = head+1, set(), set(), 0, 0
            blocks.append(dict(ordinal=head, first_tick=tick, full_tick=None, init_clock=None,
                               first_width=None))
        require(e["tick"] == tick and e["offer"] == q and len(e["parts"]) == 1
                and e["opened"] is None and e["head_initialized_before"] == len(seen),
                "Wrong head, offer or initialization ledger")
        p = e["parts"][0]
        rows, sample = set(p["rows"]), set(p["sampled"])
        require(p["ordinal"] == head and len(rows) == e["decoder_Q"] == e["q"] <= q
                and sample <= rows <= all_rows and set(p["dirty_before"]) == debt
                and set(sorted(debt)[:q]) <= rows and p["age_before"] == age,
                "Missing physical payment")
        if seen != all_rows:
            phase = "cold"
            limit = q//2 if pace else q
            pay = set(sorted(debt)[:q])
            first = set(sorted(all_rows-seen)[:min(limit, q-len(pay))])
            require(sample == first and rows == pay | first and p["prefix"] == []
                    and e["route"] == ("sparse" if first else "drain"),
                    "Cold row order, sample or width drift")
            require(not pace or first, "Paced initialization starved")
            totals.update(first_Q=len(first), cold_paid_Q=len(pay), cold_idle=q-len(rows),
                          overlap_F=int(bool(first and pay)), cold_drain_F=int(not first),
                          cold_sample_F=int(bool(first)), cap_binding_F=int(
                              pace and len(all_rows-seen) > limit and q-len(pay) > limit))
            if not seen:
                blocks[-1]["first_width"] = len(first)
            seen |= first
            if seen == all_rows:
                blocks[-1].update(full_tick=tick, init_clock=age+1)
                require(not pace or age+1 == 2*B//q, "Unexpected paced cold clocks")
        elif p["phase"] == "verify":
            phase = "verify"
            require(not sample and p["rows"] == list(range(verified, verified+len(rows)))
                    and ((e["route"] == "verify_paid" and not verified and debt and debt <= rows)
                         or (e["route"] == "verify" and not debt)), "Invalid terminal check")
            verified += len(rows)
            totals["fused_F"] += e["route"] == "verify_paid"
        else:
            closing = e["head_mask_before"] == 0 or age == 48
            phase = "closing_drain" if closing else "warm"
            require(verified == 0 and e["route"] in ("sparse", "drain")
                    and (not closing or not sample), "Late or ineligible warm sample")
            if phase == "warm":
                totals["warm_drain_F"] += int(not sample)
        flips = set(p["admitted"]) | set(p["revoked"])
        written = flips | set(p["input_changed"])
        age += int(bool(sample))
        debt = (debt-rows) | written
        require(written <= sample and age == p["age_after"] <= 48 and debt == set(p["dirty_after"]),
                "Unmeasured write, second clock advance or unpaid debt")
        totals[phase+"_F"] += 1
        totals[phase+"_Q"] += len(rows)
        totals["prefix_Q"] += len(p["prefix"])
    require(blocks and verified == B and not debt and all(b["full_tick"] is not None for b in blocks)
            and totals["first_Q"] == B*len(blocks), "Incomplete cold or terminal coverage")
    fields = [p+"_"+m for p in PHASES for m in ("F","Q")] + [
        "first_Q","cold_paid_Q","cold_idle","overlap_F","cold_drain_F","cold_sample_F",
        "cap_binding_F","fused_F","warm_drain_F","prefix_Q"]
    return {k:totals[k] for k in fields}, blocks
