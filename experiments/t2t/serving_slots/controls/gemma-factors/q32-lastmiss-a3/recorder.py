"""Observe the actual selected-row entropy decision, without another RNG draw."""
from collections import Counter
from unittest.mock import patch
from weakref import ref

Q, B = 32, 256


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def install(raw, tape, entropy, torch):
    get = ref(raw)
    apply = type(raw)._apply
    accept = entropy.accept_canvas

    def wrapped(states, pos, sample, prefix, cold, offer, route):
        req = get()
        head, = states
        base = req.origin + B*head.ordinal
        absolute = lambda rows: sorted(base+x for x in rows)
        sampled = sorted(sample)
        accepted_before = set(head.rows.accepted)
        initialized = absolute(head.rows.initialized)
        resident = list(req.bank.live)
        sc = absolute(head.sampler.sc)
        masks = absolute(set(range(B))-accepted_before)
        before = dict(tick=req.tick, head=head.ordinal, origin=req.origin, block=base,
                      route=route, phase=head.sampler.phase, width=len(pos), rows=list(pos),
                      sampled=sampled, prefix=sorted(prefix), cold=sorted(cold), initialized=initialized,
                      resident=resident, sc=sc, accepted_before=absolute(accepted_before),
                      masks_before=masks, dirty_before=absolute(head.rows.dirty),
                      frontier=masks[0] if masks else None, encoder_end=req.cache.get_seq_length(),
                      visible_end=max(set(resident) | set(pos))+1)
        proposals = []
        invocations = 0

        def observed(sampler, ids, draft, scores, clock):
            nonlocal invocations
            result = accept(sampler, ids, draft, scores, clock)
            invocations += 1
            confidence = scores.softmax(dim=-1, dtype=torch.float32).gather(-1, draft[..., None]).squeeze(-1)
            check(not proposals and len(sampled) == ids.shape[1], "Wrong proposal shape")
            proposals.extend(dict(position=p, old_id=old, draft_id=d, confidence=c, accepted=ok)
                             for p, old, d, c, ok in zip(sampled, ids[0].tolist(), draft[0].tolist(),
                                     confidence[0].tolist(), sampler.accepted_token_mask[0].tolist()))
            return result

        with patch.object(entropy, "accept_canvas", observed):
            e = apply(req, states, pos, sample, prefix, cold, offer, route)
        p, = e["parts"]
        check(offer == Q and invocations == bool(sampled) and sampled == absolute(p["sampled"])
              and e["positions"] == list(pos) and p["dirty_before"] ==
              [x-base for x in before["dirty_before"]], "Recorder disagrees with actual routing")
        admitted = set(p["admitted"])
        accepted = (accepted_before | admitted)-set(p["revoked"])
        for proposal in proposals:
            proposal["admitted"] = proposal["position"]-base in admitted
            check(proposal["accepted"] == (proposal["position"]-base in accepted), "Acceptance drift")
        before.update(proposals=proposals, admitted=absolute(admitted), dirty_after=absolute(p["dirty_after"]),
                      accepted_after=absolute(accepted), masks_after=absolute(set(range(B))-accepted),
                      written_absolute=absolute(set(p["admitted"])|set(p["revoked"])|set(p["input_changed"])),
                      edited=absolute(p["edited"]), revoked=absolute(p["revoked"]),
                      input_changed=absolute(p["input_changed"]))
        tape.append(before)
        return e

    raw._apply = wrapped


def exposures(tape, special):
    counts = Counter()
    swaps = []
    used = set()
    previous = None
    for cur in tape:
        prev, previous = previous, cur
        if cur["route"] != "sparse" or not cur["sampled"] or len(cur["initialized"]) != B:
            continue
        counts["warm_sample_ticks"] += 1
        if (prev is None or prev["tick"]+1 != cur["tick"] or prev["head"] != cur["head"]
                or prev["encoder_end"] != cur["encoder_end"] or prev["route"] != "sparse"
                or not prev["sampled"]):
            continue
        counts["adjacent_sample_ticks"] += 1
        masks = set(cur["masks_before"])
        draft = {p["position"] for p in prev["proposals"] if not p["accepted"] and not p["admitted"]
                 and p["draft_id"] not in special}
        rejected = draft & set(prev["sampled"]) & set(prev["masks_before"]) & masks
        clean = rejected - set(prev["written_absolute"]) - set(cur["dirty_before"])
        rows = set(cur["rows"])
        warm = set(cur["resident"]) & set(cur["initialized"]) & set(cur["sc"])
        target = (clean & warm) - rows - {cur["frontier"]}
        protected = set(cur["prefix"]) | set(cur["dirty_before"]) | {cur["frontier"], max(rows)}
        donors = (masks & rows & warm & set(cur["sampled"])) - protected
        counts.update(rejected_rows=len(rejected), clean_rejected_rows=len(clean),
                      clean_rejected_omitted_rows=len(target), donor_ticks=bool(donors))
        if not target or not donors:
            continue
        counts["legal_ticks_before_once_cap"] += 1
        if cur["head"] in used:
            continue
        drop = next(p for p in reversed(cur["rows"]) if p in donors)
        take = min(target)
        alt = (rows - {drop}) | {take}
        alt_sample = (set(cur["sampled"]) - {drop}) | {take}
        check(len(alt) == cur["width"] and len(alt_sample) == len(cur["sampled"])
              and (set(cur["resident"]) | alt) == (set(cur["resident"]) | rows)
              and protected <= alt, "Illegal counterfactual")
        used.add(cur["head"])
        counts.update(legal_exchanges=1, query_max_unchanged=max(alt)==max(rows))
        swaps.append(dict(tick=cur["tick"], head=cur["head"], drop=drop, take=take,
                          previous_cold=take in prev["cold"], width=cur["width"],
                          sampled_count=len(alt_sample), visible_end=cur["visible_end"],
                          rightmost_before=max(rows), rightmost_after=max(alt)))
    return dict(counts), swaps
