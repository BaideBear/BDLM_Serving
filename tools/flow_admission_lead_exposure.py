"""Audit a one-sample-lookahead admission flag on unmodified compact Flow W2 tapes."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path


def lead(previous, current, threshold=.8):
    """Requires two actual adjacent observations on the same sole open head."""
    return current < threshold <= current + max(0., current - previous)


def measure(r, indices):
    count = Counter()
    rates = defaultdict(Counter)
    exposed = set()
    for idx in indices:
        row = json.loads((r/f"{idx:02}-active_w2.json").read_bytes())
        ticks = row["ticks"]
        B = 32
        n = len(row["prompt_ids"])
        maxblocks = (n + row["cap"] + B-1)//B - n//B
        last = {}
        for t in ticks:
            for j, slot in enumerate(t["slots"], t["commit"]):
                last[j] = slot
        prior = None
        visits, blocks = Counter(), set()
        for i, t in enumerate(ticks):
            h = t["commit"]
            for j in range(h, t["right"]):
                visits[j] += 1
            s = t["slots"][0]
            allowed = (t["open"] == 1 and h+1 < maxblocks and not s["fin"] and
                       visits[h] < B+16+1 and prior is not None and prior[0] == h)
            if allowed and s["frontier"] < .8:
                nxt = ticks[i+1]
                assert nxt["commit"] == h and nxt["open"] == 1
                flag = lead(prior[1], s["frontier"])
                bucket = rates[s["frontier"]]
                bucket[flag, "n"] += 1
                bucket[flag, "yes"] += nxt["slots"][0]["frontier"] >= .8
            if allowed and lead(prior[1], s["frontier"]):
                count["trigger_ticks"] += 1
                if h not in blocks:
                    blocks.add(h)
                    exposed.add(idx)
                    # From here on, future stock states are diagnostics, never inputs to lead().
                    assert i+1 < len(ticks)
                    nxt = ticks[i+1]
                    assert nxt["commit"] == h and nxt["open"] == 1
                    count["first_triggers"] += 1
                    if nxt["slots"][0]["frontier"] >= .8:
                        count["original_progress_ge_gate_after_one_more_tick"] += 1
                    if any(v == 156892 and not pm for v, pm in zip(s["new"], s["prompt"])):
                        count["visible_eos_at_first_trigger"] += 1
                    tail_seen = h+1 in last
                    count["successor_actually_visited_some_time"] += tail_seen
                    head_ended = any(v == 156892 and not pm for v, pm in zip(last[h]["new"], last[h]["prompt"]))
                    count["head_eventually_terminal_eos"] += head_ended
                    count["head_not_terminal_and_successor_visited"] += not head_ended and tail_seen
            prior = (h, s["frontier"]) if t["open"] == 1 and not s["fin"] else None
    # Descriptive overlap only: match exact current progress, not task identity.
    # Weights use the number of flagged risk-set ticks at each matched progress.
    weight, on, off = 0, 0, 0.
    for x in rates.values():
        if x[True, "n"] and x[False, "n"]:
            weight += x[True, "n"]
            on += x[True, "yes"]
            off += x[True, "n"] * x[False, "yes"] / x[False, "n"]
    return dict(requests=len(indices), exposed_requests=len(exposed), **count,
                exact_current_progress_matched_ticks=dict(flagged_in_overlap=weight,
                    next_gate_rate_flagged=on/weight if weight else None,
                    next_gate_rate_unflagged_standardized=off/weight if weight else None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    raw = (args.root/"compact.ndjson").read_bytes()
    assert __debug__ and len(raw.splitlines()) == 90
    pairs = [json.loads(line) for line in raw.splitlines()]
    assert sorted(x["index"] for x in pairs) == list(range(90))
    canonical = json.dumps(pairs, sort_keys=True, separators=(",", ":"), allow_nan=False)
    done = json.loads((args.root/"complete.json").read_bytes())
    assert sha256(canonical.encode()).hexdigest() == done["pairs_sha256"] and done["pairs"] == 90
    print(json.dumps(dict(scope="archived-control exposure only; not treatment outcomes",
                          registration=done["reg"], sealed_pairs_sha256=done["pairs_sha256"],
                          compact_bytes_sha256=sha256(raw).hexdigest(),
                          **{k:measure(args.root, list(indices)) for k, indices in
                             (("original30", range(30)), ("added60", range(30,90)),
                              ("frozen90", range(90)))}), indent=2))


if __name__ == "__main__":
    main()
