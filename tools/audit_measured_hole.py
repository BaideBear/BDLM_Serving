"""Count blocked/retained extra writes on existing frozen GPU observations."""
import argparse
from collections import Counter
import json
from pathlib import Path

from bdlm_serving.methods.t2t.frontier_agreement import FrontierEvidenceLedger, before_measured_hole
from tools.audit_agreement_dose import run, selection


def counts(steps, excluded, *, full_tape):
    ledger = FrontierEvidenceLedger() if full_tape else None
    out = Counter()
    previous_tick = -1
    for step in steps:
        _, extras, _ = selection(step, excluded)
        if step["tick"] <= previous_tick:
            raise ValueError("physical sampling ticks are not strictly ordered")
        previous_tick = step["tick"]
        ordered = tuple(p for p in step["positions"] if p in extras)
        kept = before_measured_hole(step["positions"], step["native"], ordered)
        if ledger is not None:
            actual = ledger.observe(**{k: step[k] for k in
                                    ("block", "tick", "positions", "proposals", "probabilities", "native")},
                                    excluded_ids=excluded)
            if kept != actual:
                raise ValueError("full measured tape ledger disagrees with replay")
        if ordered:
            out.update(v1_steps=1, v1_rows=len(ordered), kept_rows=len(kept),
                       blocked_rows=len(ordered)-len(kept), any_kept_steps=int(bool(kept)),
                       any_blocked_steps=int(kept != ordered))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result_root", type=Path)
    print(json.dumps(run(p.parse_args().result_root, summarize=counts), indent=2, sort_keys=True))
