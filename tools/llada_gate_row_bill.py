"""Audit paid rows from the sealed LLaDA Q64 gate and warm-head ablations."""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(p):
    return json.loads(p.read_bytes())


def bill(arm):
    """Disjoint physical roles; sampling, tail activity and edits are overlays."""
    out, debt = Counter(), set()
    for tick, e in enumerate(arm["trace"]):
        rows, samples = set(e["rows"]), set(e["sampled"])
        n = len(rows)
        require(e["tick"] == tick and 0 < n == len(e["rows"]) == e["q"] <= e["offer"] == 64
                and e["layer_physical_q"] == [n]*20 and e["prefix_kv_length"] > 0,
                f"Unverified physical/history Q at {tick}")
        require(debt == set(e["dirty_before"]) and samples <= rows, "Missing debt or projection")
        roles = {k: set(e[k]) for k in ("dirty", "cold", "mask", "prefix")}
        named = set.union(*roles.values())
        require(named <= rows and len(named) == sum(map(len, roles.values()))
                and roles["dirty"] == debt & rows, "Invalid row classes")
        roles["other"] = rows-named
        edits, writes = set(e["edited"]), set(e["written_absolute"])
        require(edits | set(e["admitted"]) | set(e["revoked"]) == writes <= samples,
                "Unmeasured write")
        debt = (debt - rows) | writes
        require(debt == set(e["dirty_after"]), "Uncarried debt")
        for k, ps in roles.items():
            out[k+"_Q"] += len(ps)
            out[k+"_T2T"] += len(ps & edits)
        out.update(F=1, history_Q=n, tail_Q=e["tail_q"], sampled_Q=len(samples),
                   idle_seats=64-n, edits=len(edits))
    require(not debt and not arm["failure"] and out["F"] == arm["F"]
            and out["history_Q"] == arm["Q"] == arm["decoder_Q"]
            and not arm["encoder_Q"], "Unsealed trajectory or incorrect total")
    out.update(n=1, N=arm["N"], correct=int(arm["correct"]))
    return out


def report(control, gate, warm_pilot, warm_rest, manifest):
    data = defaultdict(Counter)
    tasks = load(manifest)["samples"]
    require([t["index"] for t in tasks] == list(range(90)), "Expected frozen90")
    for t in tasks:
        i = t["index"]
        if i < 60:
            half, quarter = (load(control/"rows"/f"c64-full-{g}-{i:02d}.json") for g in ("half", "quarter"))
        else:
            g = load(gate/"rows"/f"{i:02d}.json")
            half, quarter = g["old"], g["new"]
        warm = load((warm_pilot if i in (0,10,20) else warm_rest)/"rows"/f"{i:02d}.json")
        require(all(half[k] == warm["old"][k] for k in ("output_ids","Q","F","N","correct","input_sha256"))
                and bill(half) == bill(warm["old"]), "Distinct half controls")
        for tag, arm in (("half", half), ("quarter", quarter), ("half_warm_sample", warm["new"])):
            require(all(arm[k] == t[k] for k in ("index","dataset","task_id","seed"))
                    and arm["input_sha256"] == half["input_sha256"]
                    and arm["coverage"] == "head" and arm["q"] == 64 and not arm["borrow"]
                    and arm["cross_open"] == ("quarter" if tag == "quarter" else "half"),
                    "Mixed registered identities")
            counts = bill(arm)
            for group in ("original30" if i < 30 else "new60", "frozen90"):
                data[group,tag].update(counts)
    return [dict(cohort=k[0], arm=k[1], **v) for k,v in data.items()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("control","gate","warm-pilot","warm-rest","manifest"):
        ap.add_argument("--"+name, type=Path, required=True)
    rows = report(**vars(ap.parse_args()))
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
