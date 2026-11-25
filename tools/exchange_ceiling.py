"""Exact fixed-width old-edit/MASK-admission tradeoff on native LLaDA B32 traces."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from bdlm_serving.evaluation.frozen import MANIFEST
from bdlm_serving.methods.t2t.exchange import MODES, choose, position_keys, read_rates
from bdlm_serving.methods.t2t.hazard import features, read_weights
from tools.exchange_study import states
from tools.t2t_results import table


BUDGETS = ((4, 1), (8, 1), (16, 2))
FROZEN = "hard_exchange_edit4"
LOCKED = ("locked_candidate_oracle", "locked_victim_oracle", "locked_both_oracle", "locked_mask_rate")
POSITIONAL = "locked_candidate_positional"


def rerank_entrants(state, base, ranked, weights):
    """Score only eligible entrants. No current labels; gate and victims stay fixed."""
    k = len(ranked - base)
    if not k:
        return set(ranked)
    valid, resolved, fresh, strict, writes, clock = state
    masks = [i for i in valid if not resolved[i]]
    pool = [i for i in valid if resolved[i] and i not in base and i not in fresh]
    def key(i):
        x = features(i, masks, fresh, strict, writes, clock)
        if len(x) != len(weights):
            raise ValueError("Feature/weight mismatch")
        return -sum(a*b for a,b in zip(x,weights)), i
    positions = (ranked & base) | set(sorted(pool,key=key)[:k])
    assert base - positions == base - ranked and len(positions) == len(ranked)
    return positions


def lock_gate(state, rates, base, ranked, targets, admissions, mode):
    """Keep accepted swaps AND victim types. Only replace candidates/victims within that stratum."""
    if mode not in LOCKED:
        raise ValueError(mode)
    changed = ranked-base
    if not changed:
        return ranked
    valid, resolved, fresh, strict, writes, clock = state
    victims = base-ranked
    first_mask = next((i for i in valid if i in base and not resolved[i]),None)
    mask_v = {i for i in victims if not resolved[i]}
    old_v = victims-mask_v
    if first_mask in victims or victims & fresh:
        raise ValueError("Ranked selector did not preserve protected rows")
    if mode in ("locked_candidate_oracle","locked_both_oracle"):
        pool = [i for i in valid if i not in base and resolved[i] and i not in fresh]
        changed = set(sorted(pool,key=lambda i:(i not in targets,i not in changed,i))[:len(changed)])
    if mode in ("locked_victim_oracle","locked_both_oracle","locked_mask_rate"):
        masks = [i for i in valid if i in base and not resolved[i] and i != first_mask]
        if mode == "locked_mask_rate":
            keys = position_keys(valid,resolved,fresh,strict,writes,clock)
            order = sorted(masks,key=lambda i:(rates.get(keys[i],rates[("mask",)]),i not in mask_v,-i))
        else:
            order = sorted(masks,key=lambda i:(i in admissions,i not in mask_v,i))
            olds = [i for i in valid if i in base and resolved[i] and i not in fresh]
            old_v = set(sorted(olds,key=lambda i:(i in targets,i not in old_v,i))[:len(old_v)])
        victims = old_v | set(order[:len(mask_v)])
    picked = (base-victims)|changed
    assert len(picked)==len(base) and fresh<=picked and len(changed)==len(ranked-base)
    return picked


def ceiling(base, fresh, masks, old_targets, admissions, *, cap=None, max_lost=None):
    """Return an exact optimum; masks are ordered as in the zero-reserve selector.

    Among max-hit sets, minimize lost admissions, then swaps, then raw MASK
    displacement. The only allowed entrants are old positions outside base.
    """
    protected = next((i for i in masks if i in base), None)
    masks = set(masks)
    entrants = sorted(old_targets - base)
    # A swap bringing no hit can be undone without harm. If a swap evicts a
    # base hit, undoing it preserves coverage and improves the swap tie-break.
    victims = base - fresh - old_targets - {protected}
    free_old = sorted(victims - masks)
    free_mask = sorted((victims & masks) - admissions)
    paid_mask = sorted(victims & admissions)
    order = free_old + free_mask + paid_mask[:max_lost]
    n = min(len(entrants[:cap]), len(order))
    return (base - set(order[:n])) | set(entrants[:n])


def counts(base, picked, masks, old_targets, admissions):
    added, displaced = picked - base, base - picked
    assert len(added) == len(displaced) and not (added & masks)
    already = len(base & old_targets)
    new, dropped = len(added & old_targets), len(displaced & old_targets)
    raw_mask, lost = len(displaced & masks), len(displaced & admissions)
    return dict(
        old_targets=len(old_targets), base_old_hits=already,
        residual_old_targets=len(old_targets) - already,
        selected_old_hits=already + new - dropped, new_old_hits=new,
        lost_old_hits=dropped, net_new_old_hits=new - dropped,
        natural_mask_admissions=len(admissions), base_mask_admissions=len(base & admissions),
        selected_mask_admissions=len(picked & admissions),
        displaced_masks=raw_mask, lost_mask_admissions=lost,
        displaced_nonadmitting_masks=raw_mask - lost, swaps=len(added),
    )


def request_counts(folder, task, rates, weights=None):
    totals = {}
    endpoints = (("oracle_zero_loss", 0), ("oracle_unconstrained_loss", None))
    for q, cap in BUDGETS:
        variants = LOCKED + ((POSITIONAL,) if weights is not None else ())
        for label, limit in [("zero_reserve", 0), (FROZEN, cap)] + [(s,cap) for s in variants] + [
                (label, limit) for limit in (cap, None) for label, _ in endpoints]:
            totals[q, label, limit] = Counter(
                states=0, physical_q=0, mandatory_fresh=0, fresh_overflow_rows=0,
                fresh_strict_hits=0, **counts(set(), set(), set(), set(), set()))
    for state, admissions, edits in states(folder, task):
        valid, resolved, fresh, _, _, _ = state
        masks = [i for i in valid if not resolved[i]]
        mask_set = set(masks)
        old_targets = edits - fresh
        for q, cap in BUDGETS:
            base = set(choose(*state, q, 0, rates, 1.)[0])
            frozen, reserved, _ = choose(*state, q, cap, rates, MODES[FROZEN])
            frozen = set(frozen)
            assert set(reserved) == frozen - base and len(reserved) <= cap
            first = next((i for i in masks if i in base), None)
            required = fresh | ({first} if first is not None else set())
            assert len(base) == min(len(valid), max(q, len(fresh)))
            points = [("zero_reserve", 0, base), (FROZEN, cap, frozen)]
            points.extend((s,cap,lock_gate(state,rates,base,frozen,old_targets,admissions,s)) for s in LOCKED)
            if weights is not None:
                points.append((POSITIONAL,cap,rerank_entrants(state,base,frozen,weights)))
            for limit in (cap, None):
                for label, cost in endpoints:
                    picked = ceiling(base, fresh, masks, old_targets, admissions,
                                     cap=limit, max_lost=cost)
                    points.append((label, limit, picked))
            for label, limit, picked in points:
                assert required <= picked <= set(valid)
                assert limit is None or len(picked - base) <= limit
                record = totals[q, label, limit]
                record.update(states=1, physical_q=len(picked), mandatory_fresh=len(fresh),
                              fresh_overflow_rows=max(0, len(fresh) - q),
                              fresh_strict_hits=len(edits & fresh))
                record.update(counts(base, picked, mask_set, old_targets, admissions))
    return totals


def evaluate(folder, manifest, rates, weights=None):
    verified = json.loads((folder / "verified.json").read_text())
    if verified != dict(model="llada21", matched_requests=len(manifest)):
        raise ValueError("Expected verified native LLaDA traces for the complete manifest")
    requests, summary = [], defaultdict(Counter)
    for task in manifest:
        identity = {key: task[key] for key in ("index", "dataset", "task_id", "seed")}
        for key, record in request_counts(folder, task, rates, weights).items():
            requests.append(dict(**identity, **result_row(key, record)))
            summary[key].update(record)
    return requests, [result_row(key, record) for key, record in summary.items()]


def result_row(key, record):
    q, policy, cap = key
    return dict(q=q, policy=policy, max_new_seats="uncapped" if cap is None else cap, **record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True, help="verified native trace directory")
    parser.add_argument("--rates", type=Path, required=True, help="frozen exchange rates.json")
    parser.add_argument("--hazard-weights", type=Path, help="also rerank entrants with the frozen calibration24 positional fit")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())["samples"]
    weights = read_weights(args.hazard_weights) if args.hazard_weights else None
    requests, summary = evaluate(args.dir, manifest, read_rates(args.rates), weights)
    args.out.mkdir(parents=True, exist_ok=True)
    table(args.out / "requests.csv", requests)
    table(args.out / "summary.csv", summary)


if __name__ == "__main__":
    main()
