"""Count actionable prefix identity changes on sealed real Q8 right-edge tapes."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from bdlm_serving.methods.t2t.prefix_coverage import spread_prefix
from tools.audit_ibs_spare import check, inspect


def exposure(event, prompt):
    pool = event["prefix_refresh_eligible"]
    base = event["prefix_refresh_original_absolute"]
    right = event["prefix_refresh_absolute"]
    lower = max(prompt, 32 * event["block"])
    end = min(32 * event["block"] + event["cached_prefix_before"], max(base))
    check(pool == sorted(set(pool)) and all(lower <= r < end for r in pool)
          and not (set(pool) & (set(base) | set(event["dirty_before"]))), "invalid prefix pool")
    check(len(base) + event["prefix_refresh_slack"] == 8 and event["prefix_refresh_quota"] == 2,
          "not the registered parent")
    k = min(2, 8-len(base), len(pool))
    check(right == (pool[-k:] if k else []) and sorted(base + right) == event["absolute_positions"],
          "logged right arm differs from physical rows")
    spread = spread_prefix(pool, k)
    moved = len(set(spread) - set(right))
    return Counter(ticks=1, eligible_ticks=int(k > 0), two_seat_ticks=int(k == 2),
                   right_prefix_q=k, changed_ticks=int(moved > 0), replaced_q=moved)


def audit(root, manifest, frozen):
    anchor = json.loads(frozen.read_bytes())
    run = json.loads((root/"run.json").read_bytes())
    done = json.loads((root/"complete.json").read_bytes())
    returned = json.loads((root/"returned.json").read_bytes())
    panel = json.loads(manifest.read_bytes())
    check(anchor["run_sha256"] == done["study"] == sha256((root/"run.json").read_bytes()).hexdigest()
          and returned["restored"] and returned["complete"] == sha256((root/"complete.json").read_bytes()).hexdigest()
          and anchor["samples_sha256"] == run["samples_sha256"] == panel["samples_sha256"]
          and run["q"] == 8 and run["indices"] == done["indices"] == list(range(90)), "unsealed cohort")
    joint = sha256()
    counts = {}
    for task, pair in zip(panel["samples"], done["pairs"], strict=True):
        i = task["index"]
        check(pair["index"] == i and i not in counts, "invalid case")
        for quota, name in [(0, "control"), (2, "refresh")]:
            blob = (root/f"{i:02}-{run['arms'][str(quota)]}.json").read_bytes()
            joint.update(sha256(blob).digest())
            if not quota:
                continue
            r = json.loads(blob)
            check(all(r[k] == task[k] for k in ("index", "seed", "task_id", "dataset"))
                  and r["policy"] == run["arms"]["2"], "mismatched task")
            _, bill = inspect(r, quota)
            check(bill == pair[name], "different physical bill")
            c = Counter()
            for e in r["events"]:
                if e["phase"] != "prefill":
                    c.update(exposure(e, len(r["prompt_ids"])))
            check(c["ticks"] == bill["F"] and c["right_prefix_q"] == bill["prefix_Q"], "lost calls or seats")
            c.update(requests=1, exposed_requests=int(c["changed_ticks"] > 0))
            counts[i] = c
    check(joint.hexdigest() == anchor["pairs_sha256"] and set(counts) == set(range(90)), "different trajectory")
    return {"observation_only": True, "parent": "historical right-edge Q8; original full-block stop",
            "paired_source_sha256": joint.hexdigest(),
            "cohorts": {name: dict(sum((counts[i] for i in group), Counter())) for name, group in
                        (("full90", range(90)), ("original30", range(30)), ("new60", range(30, 90)),
                         ("six", [7, 13, 25, 33, 55, 71]))}}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "manifest", "frozen"):
        p.add_argument(name, type=Path)
    print(json.dumps(audit(**vars(p.parse_args())), indent=2, sort_keys=True))
