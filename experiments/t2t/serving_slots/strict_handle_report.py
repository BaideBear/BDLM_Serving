"""Read compact receipts from the two real-GPU strict-handle qualifications."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .strict_handle_qualify import JOINT_PROFILES, PROMOTION_PROFILES, STOP_PROFILES, single


def summarize(root, model, manifest):
    if model not in ("gemma", "llada"):
        raise ValueError("unknown model")
    tiers = (32,64,128,256,512) if model == "gemma" else (4,8,16,32,64)
    groups = {s:[] for s in [*(f"fixed{q}" for q in tiers),"switch25","park"]}
    ids = {r["index"]:r for r in manifest["samples"]}
    for p in sorted(root.glob("*.json")):
        r = json.loads(p.read_bytes())
        s, i = r["suite"], r["index"]
        if model=='gemma' and r.get('profile') in STOP_PROFILES:
            groups.setdefault('drop_after_stop',[])
            if r['profile'] in PROMOTION_PROFILES:
                groups.setdefault('drop_after_promotion',[])
            if r['profile'] in JOINT_PROFILES:
                groups.setdefault('drop_at_joint',[])
            if r['profile']==single.PROFILE:
                for name in single.BOUNDARIES:
                    groups.setdefault(name,[])
        if (s not in groups or i not in (0,10,20) or p.name!=f"{s}-{i:02}.json"
                or r["schema"]!=f"strict-handle-{model}-qualification-v1"
                or r["samples_sha256"]!=manifest["samples_sha256"]
                or not all(r[k]==ids[i][k] for k in ("seed","dataset","task_id"))
                or r["F"]!=r["decoder_F"]+r["encoder_F"] or r["encoder_Q"]!=tiers[-1]//2*r["encoder_F"]
                or (model=="llada" and r["encoder_F"]!=0)
                or r["decoder_projection_modules"]!=(85 if model=="gemma" else 20)
                or any(type(v) is not int or v<=0 for v in r["directed_pairs"].values())
                or sum(r["directed_pairs"].values())!=r["decoder_F"]-1):
            raise ValueError(f"invalid qualification: {p.name}")
        groups[s].append(r)
    arms = {}
    profiles = {r.get("profile","fixed90_candidate") for rows in groups.values() for r in rows}
    if len(profiles)>1:
        raise ValueError("mixed profiles in qualification")
    candidate = profiles=={single.PROFILE}
    top = {r["index"]:r for r in groups["fixed512"]} if candidate else {}
    actionable = {i for i,r in top.items() if r["stop_checks"].get("top_actionable",0)}
    for suite, rows in groups.items():
        needed = actionable if suite in single.BOUNDARIES else {0,10,20}
        if not needed or {r["index"] for r in rows}!=needed:
            raise ValueError(f"incomplete qualification: {suite}")
        changes = Counter()
        for r in rows:
            changes.update(r["directed_pairs"])
        if set(changes)-{f"{a}>{b}" for a in tiers for b in tiers}:
            raise ValueError(f"unregistered transition: {suite}")
        if suite == "switch25" and len(changes)!=len(tiers)**2:
            raise ValueError("did not exercise every directed pair")
        fixed = suite.startswith("fixed")
        if fixed and (set(changes)!={f"{suite[5:]}>{suite[5:]}"} or
                      not all(r.get("identical") is True for r in rows)):
            raise ValueError(f"fixed parity failure: {suite}")
        arms[suite] = dict(requests=len(rows), directed_pairs=dict(sorted(changes.items())),
                           **{k:sum(r[k] for r in rows) for k in
                              ("F","decoder_F","encoder_F","decoder_Q","encoder_Q","downshifts_with_debt",
                               "parked_rows_paid_on_downshift")})
        if fixed:
            arms[suite]["exact_replays"] = len(rows)
            if model=="gemma":
                arms[suite]["reference_sha256"]={r["index"]:r["reference_sha256"] for r in rows}
        if profiles and profiles <= set(STOP_PROFILES):
            counts=Counter()
            for r in rows:
                counts.update(r['stop_checks'])
            if (counts['latches']>counts['eligible_pairs'] or
                    fixed and not (suite=='fixed256' or candidate and suite=='fixed512')
                    and (counts['eligible_pairs'] or counts['latches']) or
                    suite=='drop_after_stop' and not (counts['latches'] and counts['post_stop_sub_B_ticks'] and changes['256>32'])):
                raise ValueError('unqualified native stop scope or missing post-latch downshift')
            if profiles <= set(PROMOTION_PROFILES) and (
                    fixed and suite!='fixed512' and counts['sampled_promotions']
                    or suite=='fixed512' and not counts['sampled_promotions']
                    or suite=='drop_after_promotion' and not (
                        counts['post_promotion_downshifts_with_debt'] and changes['512>32'])):
                raise ValueError('unqualified promotion scope or missing debt-carrying downshift')
            if profiles<=set(JOINT_PROFILES) and (fixed and suite!='fixed512' and counts['joint_calls'] or
                    suite=='fixed512' and not (counts['joint_calls'] and counts['tail_dirty_Q_in_joint']) or
                    suite=='drop_at_joint' and not (counts['top_to_small_at_joint'] and counts['small_verify_chunks']
                                                   and not counts['joint_calls'] and changes['512>32'])):
                raise ValueError('unqualified joint verification or missing fallback')
            arms[suite]['stop_checks']=dict(counts)
        if candidate and (suite=="fixed512" or suite in single.BOUNDARIES):
            for r in rows:
                i, b = r["index"], r["top_boundary"]
                if b["ledger_sha256"]!=top[i]["top_boundary"]["ledger_sha256"]:
                    raise ValueError("Boundary used a different historical ledger")
                if suite=="fixed512":
                    if b["prefix_ticks"]!=r["decoder_F"]:
                        raise ValueError("Fixed physical tape incomplete")
                elif not b.get("events") or b["prefix_ticks"]!=b["fixed_actionable_tick"]+int(suite=="after_top_stop"):
                    raise ValueError("Missing aligned boundary")
            arms[suite]["by_index"]={r["index"]:dict(**r["top_boundary"],stop_checks=r["stop_checks"])
                                     for r in rows}
    return dict(model=model, indices=[0,10,20], samples_sha256=manifest["samples_sha256"],
                **({"profile":profiles.pop()} if model=="gemma" else {}),
                metric="functional GPU qualification; no mixed-budget quality ranking",
                gpu=sorted({r["gpu"] for v in groups.values() for r in v}), arms=arms)


if __name__=="__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model",choices=("gemma","llada"))
    ap.add_argument("receipts",type=Path)
    ap.add_argument("manifest",type=Path)
    a=ap.parse_args()
    print(json.dumps(summarize(a.receipts,a.model,json.loads(a.manifest.read_bytes())),indent=2,sort_keys=True))
