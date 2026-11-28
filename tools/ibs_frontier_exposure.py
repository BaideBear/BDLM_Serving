"""Can one clean replay seat buy an already proposed first MASK? No inference."""
import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path


def check(ok, message):
    if not ok:
        raise ValueError(message)


def swap_if_safe(pos, masks, dirty, initialized, resolved, prompt, origin, q):
    """Use only information already available before the real forward."""
    if len(pos) != q or not masks or any(i not in resolved for i in pos):
        return None
    donors = [i for i in pos if i >= prompt and i in initialized and i not in dirty]
    if not donors:
        return None
    drop, take = max(donors), masks[0]
    new = (set(pos) - {drop}) | {take}
    if not all(i in initialized or i in new for i in range(origin, max(new)+1)):
        return None
    return drop, take


def inspect_request(r, q):
    prompt = len(r["prompt_ids"])
    initialized, resolved = set(), set(range(prompt))
    counts = Counter()
    last = False
    due = set()
    for e in r["events"]:
        pos = e["absolute_positions"]
        if e["phase"] == "prefill":
            check(len(initialized) == 0, "unexpected prefill")
            initialized.update(pos)
            continue
        origin = 32 * e["block"]
        cand = [origin + j for j in e["candidate_local"]]
        masks = [i for i in cand if i not in resolved]
        dirty = set(e["dirty_before"])
        check(pos == sorted(set(pos)) and len(cand) == len(set(cand)) and
              all(origin <= i < origin+32 for i in cand) and
              dirty == due and dirty <= set(pos) <= set(cand) and len(pos) <= q and
              e["selected_masked"] == [i not in resolved for i in pos],
              "incoming debt, position, or resolution ledger mismatch")
        last = False
        counts["generation_F"] += 1
        counts["generation_Q"] += len(pos)
        if masks and not any(e["selected_masked"]) and len(pos) == q:
            counts["no_mask_despite_proposal"] += 1
        swap = swap_if_safe(pos, masks, dirty, initialized, resolved, prompt, origin, q)
        written, edited = set(e["written_absolute"]), set(e["edited_positions"])
        if swap:
            drop, take = swap
            counts["eligible_calls"] += 1
            counts["donor_edited_on_original_call"] += drop in edited
            counts["incoming_mask_previously_projected"] += take in initialized
            counts["other_edits_on_original_call"] += len(edited - {drop})
            last = True
        check((written | edited) <= set(pos), "unmeasured write")
        resolved.difference_update(edited - written)
        resolved.update(written)
        due = written
        initialized.update(pos)
    check((counts["generation_F"], counts["generation_Q"]) ==
          (r["generation_forwards"], r["generation_query_rows"]), "F/Q mismatch")
    counts["eligible_on_final_call"] = int(last)
    return counts


def summarize(root, q):
    digest = sha256()
    groups = {}
    for i in range(90):
        blob = (root / f"{i:02d}-optcap_ibs_{q}.json").read_bytes()
        digest.update(sha256(blob).digest())
        r = json.loads(blob)
        check(r["index"] == i and r["policy"] == f"optcap_ibs_{q}" and
              r["status"] in ("success", "generation_failed"), "unregistered source")
        c = inspect_request(r, q)
        for name in ("all", r["status"], "original30" if i < 30 else "new60"):
            g = groups.setdefault(name, Counter())
            g.update(c)
            g["requests"] += 1
            g["exposed_requests"] += bool(c["eligible_calls"])
            g["requests_with_donor_edit"] += bool(c["donor_edited_on_original_call"])
    return dict(q=q, records_sha256=digest.hexdigest(), groups=groups)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--q", type=int, choices=(4, 8, 16, 32), default=4)
    args = p.parse_args()
    print(json.dumps(summarize(args.root, args.q), indent=2, sort_keys=True))
