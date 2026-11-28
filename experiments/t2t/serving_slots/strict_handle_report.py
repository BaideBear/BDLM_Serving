"""Read compact receipts from the two real-GPU strict-handle qualifications."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .strict_handle_qualify import JOINT_PROFILES, PROMOTION_PROFILES, STOP_PROFILES, single


def eos_summary(root,manifest):
    from hashlib import sha256
    from .gemma_verified_eos_qualify import PROFILE, QUAL, TIERS, check
    data=json.loads((root/"complete.json").read_bytes())
    names=[*(f"fixed{q}" for q in TIERS),"up","down","switch25","park"]
    expected={f"{name}-{i:02}.json" for name in names for i in QUAL}
    check(set(data["inputs"])==expected and set(p.name for p in root.glob("*.json"))==expected|{"complete.json"}
          and data["profile"]==PROFILE and data["suites"]==27 and data["fixed"]==15 and data["mixed"]==12,
          "Incomplete EOS qualification")
    report=dict(profile=PROFILE,fixed=15,mixed=12,source=str(root),samples_sha256=manifest["samples_sha256"],
                complete_sha256=sha256((root/"complete.json").read_bytes()).hexdigest(),suites={})
    for name in names:
        changes=Counter()
        cases=[]
        for i in QUAL:
            p=root/f"{name}-{i:02}.json"
            blob=p.read_bytes()
            r=json.loads(blob)
            t=manifest["samples"][i]
            check(sha256(blob).hexdigest()==data["inputs"][p.name] and r["identical"] is True and
                  r["samples_sha256"]==report["samples_sha256"] and r["profile"]==PROFILE
                  and r["suite"]==name and all(r[k]==t[k] for k in ("index","seed","dataset","task_id"))
                  and r["decoder_projection_modules"]==85 and r["F"]==r["decoder_F"]+r["encoder_F"] and
                  r["encoder_Q"]==256*r["encoder_F"] and sum(r["directed_pairs"].values())==r["decoder_F"]-1,
                  "Invalid sealed replay")
            changes.update(r["directed_pairs"])
            cases.append({k:r[k] for k in ("index","early","parked_rows_paid_on_downshift","downshifts_with_debt",
                                           "saved_F","saved_Q","top_boundary","reference_sha256") if k in r})
        report["suites"][name]=dict(directed_pairs=dict(changes),cases=cases)
    return report


def summarize(root, model, manifest):
    if model=="gemma-eos":
        return eos_summary(root,manifest)
    if model not in ("gemma", "llada"):
        raise ValueError("unknown model")
    tiers = (32,64,128,256,512) if model == "gemma" else (4,8,16,32,64)
    groups = {s:[] for s in [*(f"fixed{q}" for q in tiers),"switch25","park"]}
    ids = {r["index"]:r for r in manifest["samples"]}
    for p in sorted(root.glob("*.json")):
        r = json.loads(p.read_bytes())
        s, i = r["suite"], r["index"]
        if s == "park16":
            groups.setdefault("park16", [])
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
    paid16_profile = profiles=={"quarter_window_repeat64_paid16"}
    if paid16_profile and "park16" not in groups:
        raise ValueError("missing park16 suite for quarter_window_repeat64_paid16")
    if not paid16_profile and "park16" in groups:
        raise ValueError("unexpected park16 suite for this profile")
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
        if model=="llada" and profiles in ({"quarter_fullwrite"},{"quarter_edit_warm"},{"quarter_edit_window"},
                                           {"quarter_window_repeat64"},{"quarter_window_repeat64_paid16"}):
            counts=Counter()
            paid16=profiles=={"quarter_window_repeat64_paid16"}
            repeat=profiles in ({"quarter_window_repeat64"},{"quarter_window_repeat64_paid16"})
            window=repeat or profiles=={"quarter_edit_window"}
            editable=window or profiles=={"quarter_edit_warm"}
            allowed={"window_F" if window else "borrow_Q", "quarter_open_F"}|({"edit_once_F","terminal_writes","warm_sample_rows"}
                                                   if editable else {"full_write_F"})
            if paid16:
                allowed.add("paid_mask_rows")
            if window and suite in ("park","park16"):
                allowed.add("parked_dirty_Q")
            if repeat:
                allowed.update(("repeat_Q","discard_on_downshift"))
            for r in rows:
                c=r["policy_checks"]
                if set(c)-allowed or any(type(v) is not int or v<0 for v in c.values()):
                    raise ValueError("invalid LLaDA action counters")
                counts.update(c)
            if fixed:
                middle={"edit_once_F","terminal_writes"} if editable else {"full_write_F"}
                top_events={"quarter_open_F","warm_sample_rows"} if editable else {"quarter_open_F"}
                if repeat:
                    top_events.add("repeat_Q")
                expected={"fixed16":{"borrow_Q"},"fixed32":middle,"fixed64":top_events}.get(suite,set())
                if window and suite in ("fixed4","fixed8","fixed16"):
                    expected={"window_F"}
                    if paid16 and suite=="fixed16":
                        expected.add("paid_mask_rows")
                if {k for k in allowed if counts[k]} != expected:
                    raise ValueError(f"wrong fixed action exposure: {suite}")
            if suite=="switch25" and not arms[suite]["downshifts_with_debt"]:
                raise ValueError("no real debt on budget transitions")
            if paid16 and suite=="switch25" and not counts["paid_mask_rows"]:
                raise ValueError("no paid Q16 warm MASK exposure in the mixed cycle")
            if window and suite=="park" and not (counts["parked_dirty_Q"] and changes["64>4"] and changes["4>64"]):
                raise ValueError("no real parked-tail repayment and return")
            if window and suite=="park16" and not (counts["parked_dirty_Q"] and changes["64>16"] and changes["16>64"]):
                raise ValueError("no real parked-tail repayment and return")
            arms[suite]["policy_checks"]=dict(counts)
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
    if model=="llada" and profiles in ({"quarter_window_repeat64"},{"quarter_window_repeat64_paid16"}) and not any(
            arms[s]["policy_checks"].get("discard_on_downshift",0) for s in ("switch25","park")):
        raise ValueError("No live vote discarded on downshift")
    return dict(model=model, indices=[0,10,20], samples_sha256=manifest["samples_sha256"],
                **({"profile":profiles.pop()} if model=="gemma" or profiles in ({"quarter_fullwrite"},{"quarter_edit_warm"},{"quarter_edit_window"},{"quarter_window_repeat64"},{"quarter_window_repeat64_paid16"}) else {}),
                metric="functional GPU qualification; no mixed-budget quality ranking",
                gpu=sorted({r["gpu"] for v in groups.values() for r in v}), arms=arms)


if __name__=="__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model",choices=("gemma","llada","gemma-eos"))
    ap.add_argument("receipts",type=Path)
    ap.add_argument("manifest",type=Path)
    a=ap.parse_args()
    print(json.dumps(summarize(a.receipts,a.model,json.loads(a.manifest.read_bytes())),indent=2,sort_keys=True))
