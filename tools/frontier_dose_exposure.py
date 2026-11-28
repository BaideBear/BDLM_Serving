"""Replay pre-plan prefix-dose decisions on the sealed Gemma Q128 Live parent tape."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from bdlm_serving.methods.t2t.budget_family import Law, choose
from bdlm_serving.methods.t2t.frontier_dose import progress_quota


RAW = "a7d85ee3e36cdc59e100ca76794da35752cab1cebc059f693c64fb4d55542ae4"
MANIFEST = "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed"
B, Q = 256, 128
FULL = set(range(B))
SIX = (7, 13, 25, 33, 55, 71)


def require(ok, evidence):
    if not ok:
        raise ValueError(evidence)


def inspect(case):
    i, live = case["index"], case["live"]
    require(not live["error"] and case["observer_identical"], (i, "unqualified parent"))
    accepted, seen, debt = set(), set(), set()
    previous, ordinal = None, 0
    origin = live["physical"][0][1]
    c = Counter()
    for t, (e, phys) in enumerate(zip(live["events"], live["physical_positions"], strict=True)):
        key = (i, t)
        p, = e["parts"]
        rows = set(p["rows"])
        require(e["ordinal"] == ordinal == p["ordinal"] and e["tick"] == t and e["offer"] == Q
                and len(rows) == e["decoder_Q"] and e["decoder_F"] == 1 and len(rows) <= Q
                and phys == e["positions"] == [origin + B*ordinal + j for j in sorted(rows)]
                and set(p["dirty_before"]) == debt and debt <= rows
                and e["head_initialized_before"] == len(seen)
                and e["head_mask_before"] == B-len(accepted), (key, "pre-state"))
        c["D"] += 1
        c["Q"] += len(rows)
        if e["route"] == "sparse":
            unresolved, cold = FULL - accepted, FULL - seen
            pool = [j for j in reversed(range(min(unresolved, default=B))) if j in seen-debt]
            frontier = min(FULL - ((accepted & seen) - debt), default=B)
            args = (Q, B, debt, unresolved, pool)
            parent = choose(*args, law=Law(1, 4), cold=cold)
            require(tuple(p["rows"]) == parent.rows and tuple(p["prefix"]) == parent.prefix
                    and tuple(p["sampled"]) == (parent.cold or parent.rows)
                    and not parent.drain_only, (key, "parent plan mismatch"))
            # Cold sampling is unchanged but records its pre-plan frontier in both arms.
            growth = previous is not None and frontier > previous
            keep = progress_quota(previous, frontier, 1)
            alternate = choose(*args, law=Law(keep, 4), cold=cold)
            removed = set(parent.rows) - set(alternate.rows)
            added = set(alternate.rows) - set(parent.rows)
            require(removed <= set(parent.prefix) and added <= (unresolved & seen) - debt
                    and len(added) <= len(removed) and alternate.dirty == parent.dirty
                    and alternate.cold == parent.cold and (not cold or parent == alternate),
                    (key, "non-prefix intervention"))
            c["sample_plans"] += 1
            c["growth_plans"] += growth
            c["plans_with_row_change"] += bool(removed)
            c["removed_prefix_rows"] += len(removed)
            c["replacement_mask_rows"] += len(added)
            c["seats_left_free"] += len(removed)-len(added)
            previous = frontier
        else:
            require(e["route"] in ("drain", "verify", "verify_paid") and not p["sampled"]
                    and not p["prefix"], (key, "not an ordinary sampling tick"))
        writes = set(p["admitted"]) | set(p["revoked"]) | set(p["input_changed"])
        accepted = (accepted | set(p["admitted"])) - set(p["revoked"])
        seen.update(rows)
        debt = (debt-rows) | writes
        require(set(p["dirty_after"]) == debt, (key, "post-state"))
        if e["encoded"]:
            require(not debt, (key, "epoch changed with dirty KV"))
            ordinal += 1
            previous = None
            accepted.clear()
            seen.clear()
    require((c["D"], c["Q"], ordinal) == (live["D"], live["Q"], live["E"])
            and not debt, (i, "physical bill"))
    c["requests_with_row_change"] = int(c["plans_with_row_change"] > 0)
    c["requests"] = 1
    c["E"] = ordinal
    return c


def audit(raw, manifest, token_reference):
    require(hashlib.sha256(raw).hexdigest() == RAW and manifest["samples_sha256"] == MANIFEST,
            "wrong input")
    meta = {p["index"]: p for p in manifest["samples"]}
    token_hash = hashlib.sha256(token_reference).hexdigest()
    prompts = {r["index"]: r for r in json.loads(token_reference)["rows"]}
    rows, complete, verified_inputs = {}, False, False
    for line in raw.splitlines():
        if not line.startswith(b"RECEIPT "):
            continue
        p = json.loads(line[8:])
        if p["kind"] == "complete":
            complete = p["requests"] == 90
        if p["kind"] == "prompt_preflight":
            verified_inputs = (p["manifest_sha256"] == token_hash and
                               p["matched_actual_encoder"] == p["matched_historical_live"] == 90)
        if p["kind"] != "case":
            continue
        i = p["index"]
        require(i not in rows and all(p[k] == meta[i][k] == prompts[i][k]
                                      for k in ("task_id", "dataset", "seed"))
                and p["prompt_sha256"] == prompts[i]["prompt_sha256"], (i, "frozen input"))
        rows[i] = inspect(p)
    require(complete and verified_inputs and set(rows) == set(meta) == set(prompts) == set(range(90)),
            "unverified inputs or incomplete tape")
    scopes = dict(fixed6=SIX, original30=range(30), new60=range(30, 90), frozen90=range(90))
    def aggregate(ids):
        total = Counter()
        for i in ids:
            total.update(rows[i])
        return dict(total)
    return dict(scope="same-state exposure only; not a treatment rollout",
                source_sha256=RAW, manifest_samples_sha256=MANIFEST, token_manifest_sha256=token_hash,
                cohorts={k: aggregate(v) for k, v in scopes.items()},
                requests={i: dict(rows[i]) for i in sorted(rows)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw", type=Path)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("token_reference", type=Path)
    ns = ap.parse_args()
    print(json.dumps(audit(ns.raw.read_bytes(), json.loads(ns.manifest.read_bytes()),
                           ns.token_reference.read_bytes()), indent=2))
