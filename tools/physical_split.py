"""Partition observed hard-policy query rows, without inferring a split from the quota."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from bdlm_serving.evaluation.frozen import MANIFEST
from bdlm_serving.methods.t2t.protocol import verify_work


def request_split(row):
    verify_work(row)
    c = Counter()
    for event in row["events"]:
        if event["stage"] != "selected":
            continue
        ll = row["model"] == "llada21"
        positions = event["absolute_positions" if ll else "positions"]
        q = event["physical_q" if ll else "q"]
        fresh = event["mandatory_fresh"]
        if not ll and fresh != event["selected_fresh"]:
            raise ValueError("A mandatory fresh row was skipped")
        masks = event["selected_masked_before" if ll else "selected_mask"]
        extra = len(event["reserved_extra_edit_positions"])
        backfill = q - fresh - masks - extra
        if (not 0 <= backfill <= q or q != len(set(positions)) or
                (ll and any(n != q for n in event["layer_physical_q"]))):
            raise ValueError("Invalid physical split")
        c.update(selected_F=1, selected_Q=q, mandatory_Q=fresh, mask_Q=masks,
                 reserved_old_Q=extra, backfill_old_Q=backfill,
                 overflow_Q=event["overflow"])
    phases = row["q_by_phase"]
    if c["selected_Q"] != phases.get("selected", 0) or sum(phases.values()) != row["generation_physical_padded_rows"]:
        raise ValueError("Observed physical rows disagree with phase totals")
    c.update(total_Q=sum(phases.values()), total_F=row["generation_forwards"],
             other_Q=sum(phases.values())-c["selected_Q"])
    return c


def generate(runs, filters, out):
    manifest = json.loads(MANIFEST.read_text())["samples"]
    result = []
    for run in runs:
        config = json.loads((run/"run.json").read_text())
        offset = config.get("seed_offset", 0)
        for arm, policy in zip(config["arms"], config["policies"], strict=True):
            if arm["selection"] not in filters:
                continue
            if json.loads((run/f"complete-{policy}.json").read_text()) != dict(requests=30, policy=policy):
                raise ValueError(f"Incomplete {policy}")
            c = Counter()
            for m in manifest:
                row = json.loads((run/f"{m['index']:02d}-{policy}.json").read_text())
                if (any(row[k] != m[k] for k in ("index", "dataset", "task_id")) or
                        row["seed"] != m["seed"] + offset or row["policy"] != policy or
                        row["model"] != config["model"]):
                    raise ValueError("Arm identity mismatch")
                c.update(request_split(row))
            result.append(dict(model=config["model"], seed_offset=offset, budget=arm["budget"],
                policy=policy, **{k:c[k] for k in ("total_F", "total_Q", "selected_F",
                "selected_Q", "mandatory_Q", "mask_Q", "reserved_old_Q", "backfill_old_Q",
                "other_Q", "overflow_Q")}))
    if not result:
        raise ValueError("No matching complete hard-policy arms")
    keys = [(r["model"], r["seed_offset"], r["policy"]) for r in result]
    if len(set(keys)) != len(keys):
        raise ValueError("Overlapping arms")
    result.sort(key=lambda r:(r["model"], r["seed_offset"], r["budget"], r["policy"]))
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, result[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, action="append")
    p.add_argument("--mode", required=True, action="append")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    generate(args.run, set(args.mode), args.out)
