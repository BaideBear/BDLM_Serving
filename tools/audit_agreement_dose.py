"""Compare two choices of k extra admissions on each unchanged GPU observer step."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path


SIX = (7, 13, 25, 33, 55, 71)
# AutoProcessor.tokenizer.all_special_ids union generation_config eos and pad,
# from the exact A3 snapshot /gs/models/diffusiongemma-cd40195.
GEMMA_EXCLUDED = {0, 1, 2, 3, 4, 46, 47, 48, 49, 50, 51, 52, 98, 100, 101, 105,
                  106, 255999, 256000, 258880, 258881, 258882, 258883, 258884}


def selection(step, excluded):
    rows = list(zip(step["positions"], step["proposals"], step["probabilities"],
                    step["native"], strict=True))
    if len({r[0] for r in rows}) != len(rows) or any(not isfinite(r[2]) for r in rows):
        raise ValueError("invalid observation")
    pool = {pos: (token, prob) for pos, token, prob, take in rows
            if not take and token not in excluded and prob > .5}
    announced = step["eligible"] if "eligible" in step else step["extra"]
    votes = set(announced)
    if not votes <= pool.keys() or len(votes) != len(announced):
        raise ValueError("unpaid, native, duplicated, or ineligible extra admission")
    top = set(sorted(pool, key=lambda i: (-pool[i][1], i))[:len(votes)])
    return pool, votes, top


def tally(steps, excluded, *, full_tape):
    out, next_admission = Counter(), {}
    ticks = [s["tick"] for s in steps]
    if sorted(set(ticks)) != ticks:
        raise ValueError("steps not strictly ordered")
    for step in reversed(steps):
        pool, votes, top = selection(step, excluded)
        if votes:
            out.update(exposed_steps=1, dose=len(votes), shared_rows=len(votes & top),
                       replaced_rows=len(votes - top), different_set_steps=int(votes != top))
        if full_tape:
            for label, subset in (("repeat_only", votes-top), ("current_only", top-votes)):
                for pos in subset:
                    later = next_admission.get((step["block"], pos))
                    status = ("unobserved" if later is None else
                              "match" if pool[pos][0] == later else "different")
                    out[f"{label}_{status}"] += 1
            # This label is the next ordinary admission, not the final answer or a T2T.
            for pos, token, accepted in zip(step["positions"], step["proposals"],
                                            step["native"], strict=True):
                if accepted:
                    next_admission[step["block"], pos] = token
    return out


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def run(root, *, summarize=tally):
    llada = root / "llada-ibs-q16-paid-agreement-h0-v1"
    gemma = root / "gemma-paid-agreement-q128-01"
    reg, reg_hash = read(llada/"registration.json")
    notes = {r["index"]: r for r in map(json.loads, (llada/"receipts.ndjson").read_text().splitlines())}
    summary, ghash = read(gemma/"summary.json")
    if sorted(notes) != list(SIX) or summary["cases"] != list(SIX) or summary["gate"] != "passed":
        raise ValueError("wrong cohort")
    out = {"source_registration": {"llada": reg_hash, "gemma_summary": ghash}, "models": {}}
    for model in ("llada", "gemma"):
        total, cases = Counter(), []
        for index in SIX:
            if model == "llada":
                d, digest = read(llada/"pairs"/f"{index:02}.json")
                tracked = d["arms"]["tracked"]
                if (d["arms"]["stock"] != tracked or tracked["index"] != index
                        or digest != notes[index]["raw_sha"]):
                    raise ValueError("LLaDA observer drift")
                steps, excluded = d["tapes"]["tracked"]["ticks"], set(reg["excluded"])
            else:
                d, digest = read(gemma/f"case-{index:02}.json")
                if (not d["stock_observer_identical"] or d["index"] != index
                        or d != next(r for r in summary["per_case"] if r["index"] == index)):
                    raise ValueError("Gemma observer drift")
                steps = d["arms"]["observer"]["agreement"]["extra_events"]
                excluded = GEMMA_EXCLUDED
            c = summarize(steps, excluded, full_tape=model == "llada")
            total.update(c)
            cases.append({"index": index, "source_sha256": digest, **c})
        out["models"][model] = {"total": total, "per_request": cases}
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result_root", type=Path)
    print(json.dumps(run(p.parse_args().result_root), indent=2, sort_keys=True))
