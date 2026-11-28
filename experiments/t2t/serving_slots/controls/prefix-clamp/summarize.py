"""Reconcile the registered six GPU receipts; this does not execute inference."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(root):
    indices = (7, 13, 25, 33, 55, 71)
    summary = json.loads((root / "summary.json").read_bytes())
    if (summary["cases"] != list(indices) or summary["probe"] not in {"paid-prefix", "native-prefix"}
            or summary["gate"] != "passed" or summary["failed"]):
        raise ValueError("Incomplete or invalid run")
    cases, digests = [], {}
    for i, reg in zip(indices, summary["per_case"], strict=True):
        path = root / f"case-{i:02}.json"
        payload = path.read_bytes()
        digests[path.name] = hashlib.sha256(payload).hexdigest()
        r = json.loads(payload)
        arm = r["arms"]["probe"]
        f = arm["fresh"]
        if (r["index"] != i or reg["index"] != i or reg["fresh"] != f or r["error"]
                or not r["stock_observer_probe_identical"]
                or any(arm[k] != r["arms"][other][k]
                       for other in ("stock", "observer") for k in ("N", "bill", "correct", "rng_after"))):
            raise ValueError(f"Unmatched control at {i}")
        row = dict(index=i, N=arm["N"], bill=arm["bill"], correct=arm["correct"], exposed=f is not None)
        if f is not None:
            pref, masks = set(f["refreshed_prefix"]), set(f["measured_mask"])
            native = summary["probe"] == "native-prefix"
            expected_f = 4 if native else 3
            expected_q = f["visible_width"] + 3*f["selected"] if native else 3*f["visible_width"]
            if (not f["identical_replay"] or f["extra_F"] != expected_f or f["extra_Q"] != expected_q
                    or not 0 < f["selected"] <= min(f["visible_width"], 64) or not masks or not pref
                    or not all(0 <= x < f["ready"] for x in pref)
                    or not all(f["ready"] <= x < f["visible_width"] for x in masks)):
                raise ValueError(f"Invalid intervention at {i}")
            row.update(tick=f["tick"], projected=f["selected"], visible=f["visible_width"],
                       prefix=len(pref), masks=len(masks), control_top1=len(f["control_vs_resident_top1"]),
                       control_accept=len(f["control_vs_resident_accept"]), top1=len(f["top1_flips"]),
                       p9=len(f["p9_flips"]), admissions=len(f["take_flips"]),
                       accept_up=sorted(set(f["fresh_take"])-set(f["clamped_take"])),
                       accept_down=sorted(set(f["clamped_take"])-set(f["fresh_take"])),
                       extra_F=f["extra_F"], extra_Q=f["extra_Q"])
        cases.append(row)
    for name in ("summary.json", "raw.log"):
        digests[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return dict(source=root.name, probe=summary["probe"], sha256=digests, cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
