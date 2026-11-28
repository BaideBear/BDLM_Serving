"""Validate the A3 recorder gate and export counts without token IDs."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

from recorder import B, Q, check, exposures

CASES = [0, 10, 20]
GPU = "GPU-94c61e79-d80b-cae2-d85f-6bd2a2c99a1c"


def verify_case(case):
    bare, noop = case["bare"], case["noop"]
    check(case["controls_identical"] and bare == noop and not bare["error"], "Control mismatch")
    tape, events = case["tape"], bare["events"]
    check(len(tape) == len(events) == bare["D"] == len(bare["physical_positions"])
          and bare["projection_widths_verified"] == 85, "Incomplete physical records")
    previous = None
    for i, (row, event) in enumerate(zip(tape, events)):
        p, = event["parts"]
        base = row["origin"] + B*row["head"]
        absolute = lambda xs: sorted(base+x for x in xs)
        missing = set(range(base, base+B))
        masks = set(row["masks_before"])
        sample = set(row["sampled"])
        sel = set(row["rows"])
        debt = set(row["dirty_before"])
        initialized = set(row["initialized"])
        check(row["tick"] == event["tick"] == i and row["head"] == event["ordinal"] == p["ordinal"]
              and row["encoder_end"] == base == row["block"] and row["route"] == event["route"]
              and row["width"] == event["decoder_Q"] == len(sel) <= Q
              and row["layer_physical_q"] == [row["width"]]*85
              and row["rows"] == event["positions"] == bare["physical_positions"][i]
              and initialized == set(row["resident"]) <= missing
              and len(initialized) == event["head_initialized_before"]
              and row["visible_end"] == max(initialized | sel)+1
              and masks == missing - set(row["accepted_before"])
              and len(masks) == event["head_mask_before"]
              and row["frontier"] == min(masks, default=None)
              and debt <= sel and row["dirty_before"] == event["dirty_before_absolute"]
              and row["prefix"] == absolute(p["prefix"])
              and set(row["cold"]) <= sel-initialized
              and row["sampled"] == event["sampled_absolute"]
              and row["written_absolute"] == event["written_absolute"]
              and row["dirty_after"] == event["dirty_after_absolute"],
              (case["index"], i, "physical or frontier invariant"))
        added, revoked = set(absolute(p["admitted"])), set(absolute(p["revoked"]))
        after = (set(row["accepted_before"]) | added) - revoked
        proposals = row["proposals"]
        check([a["position"] for a in proposals] == row["sampled"]
              and all(a["accepted"] == (a["position"] in after)
                      and a["admitted"] == (a["position"] in added)
                      and 0 <= a["confidence"] <= 1 for a in proposals)
              and set(row["accepted_after"]) == after and set(row["masks_after"]) == missing - after
              and sample <= sel and set(row["dirty_after"]) == (debt-sel) | set(row["written_absolute"]),
              (case["index"], i, "proposal or write invariant"))
        if previous and row["head"] == previous["head"]:
            check(initialized == set(previous["initialized"]) | set(previous["rows"])
                  and row["accepted_before"] == previous["accepted_after"]
                  and row["dirty_before"] == previous["dirty_after"], (case["index"], i, "continuity"))
        else:
            check(not initialized and not row["accepted_before"] and not debt, "New head not cold")
        previous = row
    counts, swaps = exposures(tape, set(case["special"]))
    check(counts == case["counts"] and swaps == case["swaps"], "Exposure recomputation mismatch")
    return dict(index=case["index"], counts=counts, swaps=swaps, exact_bare_noop=True)


def main(root):
    raw = (root/"raw.log").read_bytes()
    check((root/"pipeline.exit").read_text().split() == ["0", "0"], "Invalid exit")
    start, *cases, done = [json.loads(l[8:]) for l in raw.splitlines() if l.startswith(b"RECEIPT ")]
    check(start["kind"] == "start" and start["gpu"] == GPU and start["q"] == Q
          and start["profile"] == "dense_stop_joint_verified_eos" and start["cases"] == CASES
          and all(c["kind"] == "case" for c in cases) and [c["index"] for c in cases] == CASES
          and done == dict(kind="complete", requests=len(CASES)), "Wrong run")
    release, = [l.decode() for l in raw.splitlines() if l.startswith(b"RELEASE ")]
    release_check = (root/"release-check.txt").read_text().strip()
    check(all(s in release for s in ("rc=0", f"state={GPU}\\,\\ 0\\,", "apps=''", "lock=closed"))
          and all(s in release_check for s in (f"state={GPU}\\,\\ 0\\,\\ 0", "apps=''",
                                               "worker=absent", "scratch=absent", "lock=free")),
          "Missing clean A3 release")
    rows = [verify_case(c) for c in cases]
    totals = Counter()
    for r in rows:
        totals.update(r["counts"])
    out = dict(schema="gemma-live-q32-lastmiss-a3-gate-v1", gpu=GPU,
               raw_sha256=sha256(raw).hexdigest(), cases=CASES, rows=rows, totals=dict(totals),
               decision="eligible_for_separate_registration" if totals["legal_exchanges"] else "no_go",
               launcher_release=release, independent_release=release_check)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
