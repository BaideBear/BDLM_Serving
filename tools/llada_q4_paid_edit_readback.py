"""Read archived paired edits and IBS trajectories on n164. No model or GPU."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path("/data1/project_data/gx-erqw-20261123")
SNAP = ROOT / "archive/a800-gs-20261128/snapshot/results"
IBS = ROOT / "runtime/t2t-native-1125/prefix90/llada21-optimus-hardcap-low"
IBS_SHA = "444ebcf67fa934e7bda89e864e758e5c8112b4802d89da6f42588421a07902d5"


def check(ok, label):
    if not ok:
        raise ValueError(label)


def audit_pairs(directory, q, indices):
    run = json.loads((directory / "run.json").read_bytes())
    check((run["q"], run["mode"], run["indices"], run["protocol"]) ==
          (q, "head", indices, "llada-t2t-dirty-single-factor-v1"), "run identity")
    files = sorted((directory / "rows").glob("*.json"))
    check([int(p.stem) for p in files] == indices, "case denominator")
    count, seal = Counter(), sha256()
    for path in files:
        blob = path.read_bytes()
        seal.update(sha256(blob).digest())
        pair = json.loads(blob)
        old, new = pair["old"], pair["new"]
        check(all(pair[k] == old[k] == new[k] for k in ("index", "task_id", "dataset", "seed"))
              and pair["index"] == int(path.stem) and old["input_sha256"] == new["input_sha256"],
              "unmatched pair")
        c = pair["compare"]
        check(c["passed"] and all(v is True for v in c["checks"].values()), "first opportunity")
        for name, arm in (("off", old), ("on", new)):
            trace = arm["trace"]
            check(len(trace) == arm["F"] and sum(e["q"] for e in trace) == arm["Q"] ==
                  arm["decoder_Q"] and arm["encoder_Q"] == 0, "physical bill")
            debt = set()
            for e in trace:
                rows = set(e["rows"])
                check(len(rows) == len(e["rows"]) == e["q"] <= q and
                      e["layer_physical_q"] == [len(rows)] * 20, "physical cap")
                check(set(e["dirty_before"]) == debt and set(e["dirty"]) == debt & rows,
                      "debt payment")
                writes = set(e["admitted"] + e["edited"] + e["revoked"])
                check(writes == set(e["written_absolute"]) and writes <= set(e["sampled"]) <= rows,
                      "write eligibility")
                debt = (debt - rows) | writes
                check(debt == set(e["dirty_after"]), "writeback")
                if name == "on":
                    extra = set(e.get("t2t_dirty_rows", ()))
                    if extra:
                        check(extra <= set(e["dirty"]) & set(e["sampled"]) and
                              set(e["cold"]) & set(e["sampled"]) and
                              set(e["dirty_before"]) <= rows, "exposure")
                        count.update(opportunity_F=1, extra_rows=len(extra),
                                     edits=len(extra & set(e["edited"])),
                                     revokes=len(extra & set(e["revoked"])))
            check(not debt and not arm["failure"], "incomplete request")
            for k in ("N", "F", "Q", "correct"):
                count[f"{name}_{k}"] += arm[k]
            count[f"{name}_sum_TPF"] += arm["N"] / arm["F"]
        count.update(pairs=1, equal_ids=int(old["output_ids"] == new["output_ids"]),
                     rescues=int(new["correct"] and not old["correct"]),
                     losses=int(old["correct"] and not new["correct"]))
    return dict(counts=dict(count), rows_sha256=seal.hexdigest())


def ibs_edit_phase():
    """Classify the actual edited dirty rows, not all rows offered for editing."""
    seal, phases = sha256(), Counter()
    for i in range(90):
        blob = (IBS / f"{i:02d}-optcap_ibs_4.json").read_bytes()
        seal.update(sha256(blob).digest())
        item = json.loads(blob)
        check(item["index"] == i, "IBS identity")
        resident = set()
        for e in item["events"]:
            pos = e["absolute_positions"]
            if e["phase"] != "prefill":
                masks = {p for p, m in zip(pos, e["selected_masked"], strict=True) if m}
                dirty = set(e["dirty_before"])
                key = ("with_cold" if masks - resident else "warm_only" if masks
                       else "all_rows_dirty" if dirty == set(pos) else "no_mask_other")
                phases[key] += len(dirty & set(e["edited_positions"]))
            resident.update(pos)
    check(seal.hexdigest() == IBS_SHA, "IBS archive digest")
    return dict(phases)


if __name__ == "__main__":
    print(json.dumps({
        "C4": audit_pairs(SNAP / "llada-t2t-dirty-c4-pilot-20261128-062040", 4, [0, 10, 20]),
        "C8": audit_pairs(SNAP / "llada-t2t-dirty-c8-pilot-20261128-062445", 8, list(range(30))),
        "IBS4_edits_on_paid_debt_by_call": ibs_edit_phase(),
    }, indent=2, sort_keys=True))
