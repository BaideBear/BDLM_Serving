"""Publish the fixed K0 reference without selecting a best arm at each budget."""
import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fixed_rows(source):
    for model, slots in (("llada21", (4, 8, 16, 32)), ("gemma", (16, 32, 64, 128, 256))):
        for q in slots:
            path = source / f"{model}-q{q}-k1-right.csv"
            with path.open() as f:
                rows = [r for r in csv.DictReader(f) if r["scope"] == "all"]
            if ([int(r["requests"]) for r in rows] != [30, 60, 90] or any(
                    r["model"] != model or int(r["nominal_q"]) != q or
                    r["baseline_policy"] != f"t2t_q{q}_f0_m{q}_hard_prefix_zero" for r in rows)):
                raise ValueError(f"Unexpected fixed control: {path}")
            for field in ("correct","N","F","physical_Q"):
                vals = [int(r["baseline_"+field]) for r in rows]
                if sum(vals[:2]) != vals[2]:
                    raise ValueError(f"Disjoint cohorts disagree: {path} {field}")
            for r in rows:
                yield dict(model=model, slot=q, role="diagnostic" if (model,q)==("gemma",16)
                           else "primary", cohort=r["cohort"], requests=r["requests"],
                           **{k:r["baseline_"+k] for k in ("correct","macro_tpf","pooled_token_efficiency",
                                                            "N","F","physical_Q","q_per_f")})


def adjacent(data, private, source):
    from bdlm_serving.evaluation.frozen import sha
    from experiments.t2t.prefix_window.score_run import snapshot
    from tools.prefix_cohorts import load_cases, load_request_set, select_window_cases, _window_row

    cases = load_cases(data/"samples.json")
    digest = sha(data/"samples.json")
    fixed = {(r["model"], r["slot"], r["cohort"]): r for r in fixed_rows(source)}
    for model, dirs in (
        ("llada21", {4:"llada21-4-matched", 8:"llada-q8", 16:"llada21-16-matched", 32:"llada21-32-matched"}),
        ("gemma", {16:"gemma-grid-k1", 32:"gemma-q32", 64:"gemma-grid-k1",
                   128:"gemma-grid-k1", 256:"gemma-grid-k1"}),
    ):
        prev = None
        for q, name in dirs.items():
            label = f"t2t_q{q}_f0_m{q}_hard_prefix_zero"
            d = snapshot(data/name, label, digest, private/".inputs")
            rows, m, p, budget = load_request_set(d, cases, "K0", label)
            if (m,p,budget) != (model,label,q):
                raise ValueError("Fixed control identity mismatch")
            current = {r["index"]:r for r in rows}
            if prev is not None:
                lo, low_label, old = prev
                for window in (30,60,90):
                    w = _window_row(select_window_cases(cases,window), old,current,window,"all",
                                    model,low_label,label,None)
                    for side, width in (("baseline",lo),("arm",q)):
                        ref = fixed[model,width,w["cohort"]]
                        if any(float(ref[k]) != w[f"{side}_{k}"]
                               for k in ("correct","macro_tpf","N","F","physical_Q")):
                            raise ValueError(f"Re-scoring disagrees with fixed control: {model}/{width}")
                    yield dict(model=model,low=lo,high=q,cohort=w["cohort"],requests=window,
                               **{k:w[k] for k in ("baseline_correct","arm_correct","rescued","lost",
                                   "quality_delta","quality_delta_ci_low","quality_delta_ci_high",
                                   "macro_tpf_delta","macro_tpf_delta_ci_low","macro_tpf_delta_ci_high",
                                   "baseline_q_per_f","arm_q_per_f","delta_F","delta_Q")})
            prev = q,label,current


def write_rows(path, rows):
    rows = list(rows)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def true64_controls(data):
    import json
    from bdlm_serving.evaluation.frozen import sha
    from tools.prefix_cohorts import load_cases, load_request_set, select_window_cases, _window_row

    cases_path=data/"samples.json"
    cases=load_cases(cases_path)
    target=data/"llada21-rolling-dual64"
    def read(folder,file):
        return json.loads((folder/file).read_text())
    def load(folder,p):
        meta=read(folder,"run.json")
        if (meta["samples_sha256"]!=sha(cases_path) or meta["model"]!="llada21" or
                meta["suite"]!="frozen90"):
            raise ValueError("Wrong LLaDA suite")
        if p=="fullwidth":
            if read(folder,"qualified.json")!=dict(model="llada21",full_width=32,matched=90):
                raise ValueError("Missing native endpoint proof")
        elif read(folder,"complete-"+p+".json")!=dict(policy=p,requests=90):
            raise ValueError("Incomplete LLaDA control")
        rows,m,label,_=load_request_set(folder,cases,"cross-slot",p)
        if (m,label)!=("llada21",p):
            raise ValueError("Wrong LLaDA policy")
        return {r["index"]:r for r in rows}
    proof=read(target,"reproduced30-dual64_noedit.json")
    if proof["matched"]!=30 or proof["policy"]!="dual64_noedit":
        raise ValueError("Missing true64 historical reproduction")
    dual=load(target,"dual64_noedit")
    for n,p in ((16,"t2t_q16_f0_m16_hard_prefix_zero"),(32,"fullwidth")):
        baseline=load(data/f"llada21-{n}-matched",p)
        for w in (30,60,90):
            yield dict(_window_row(select_window_cases(cases,w),baseline,dual,w,"all",
                                   "llada21",p,"dual64_noedit",None),
                       baseline_nominal_q=n,arm_nominal_q=64)


def unified_fixed_row(raw, case, arm, budget):
    from bdlm_serving.evaluation.scoring import score
    if (raw["model"] != "diffusiongemma" or raw["policy"] != arm or raw["status"] != "success"
            or any(raw[k] != case[k] for k in ("index", "dataset", "seed", "task_id"))
            or raw["decoder_layers"] != 30 or raw["projection_modules"] != 85):
        raise ValueError("Wrong unified frozen request or projection coverage")
    t = raw["trace"]
    if not t or any(e["tick"] != i or e["offer"] != budget or not 0 < e["q"] <= 512 or
            not 0 <= e["mandatory_q"] <= e["q"] or not 0 <= e["prefix"] <= e["q"]-e["mandatory_q"]
            or e["overflow"] != max(0, e["q"]-budget) for i,e in enumerate(t)):
        raise ValueError("Invalid fixed-offer trace")
    enc = sum(len(e["encoded"]) for e in t)
    f, q, n = len(t)+enc, sum(e["q"] for e in t)+256*enc, raw["output_tokens"]
    if ((q,f,enc*256,q-enc*256) != (raw["generation_physical_padded_rows"],raw["generation_forwards"],
            raw["encoder_q"],raw["decoder_q"]) or raw["tpf"] != n/f or raw["token_efficiency"] != n/q
            or n < 0 or type(raw["truncated"]) is not bool):
        raise ValueError("Physical work disagrees with compact trace")
    stages = ("bootstrap","verify","sparse","dense","tail_dirty_drain")
    if any(e["phase"] not in stages for e in t):
        raise ValueError("Unregistered decoder phase")
    sparse = [e for e in t if e["phase"]=="sparse"]
    if any(e["parked"] or e["opportunity"] for e in sparse):
        raise ValueError("Cannot infer mask-only rows on a multi-block sparse step")
    sq = sum(e["q"] for e in sparse)
    fresh = sum(e["mandatory_q"] for e in sparse)
    prefix = sum(e["prefix"] for e in sparse)
    breakdown = {"excess_"+phase: sum(e["overflow"] for e in t if e["phase"] == phase) for phase in stages}
    return dict(**{k:case[k] for k in ("index","dataset","seed","task_id")}, **breakdown,
                N=n, F=f, Q=q, physical_Q=q, max_q=max(e["q"] for e in t),
                tpf=n/f, failed=0, truncated=raw["truncated"], correct=int(score(raw["output_text"],case)["correct"]),
                prefix_rows=sum(e["prefix"] for e in t), prefix_ticks=sum(e["prefix"]>0 for e in t),
                decoder_f=len(t), encoder_f=enc, overflow_ticks=sum(e["overflow"]>0 for e in t),
                overflow_rows=sum(e["overflow"] for e in t), strict_edits=sum(e["strict_edits"] for e in t),
                sparse_q=sq,sparse_mandatory_q=fresh,sparse_prefix_q=prefix,sparse_mask_only_q=sq-fresh-prefix)


def rank_evidence(a, b, quota):
    first = a["first_rank_change"]
    if (a["input_sha256"] != b["input_sha256"] or b["first_rank_change"] != first
            or any(e["prefix"] > quota for r in (a,b) for e in r["trace"])):
        raise ValueError("Unpaired rank witness or quota exceeded")
    if first is None:
        if any(a[k]!=b[k] for k in ("output_ids","trace","output_tokens","generation_forwards","generation_physical_padded_rows")):
            raise ValueError("Unchanged selections diverged")
        return False
    if type(first) != int or not 0 <= first < min(len(r["trace"]) for r in (a,b)):
        raise ValueError("Invalid first intervention")
    x,y=(r["trace"][first] for r in (a,b))
    if (a["trace"][:first] != b["trace"][:first] or x["phase"] != "sparse" or not x["prefix"]
            or any(x[k]!=y[k] for k in ("q","offer","phase","tick","head","mandatory_q","prefix","overflow"))):
        raise ValueError("First intervention not matched in cost and state")
    return True


def exchange_evidence(control, trial):
    first=trial["first_exchange_tick"]
    if control["first_exchange_tick"]!=first or control["input_sha256"]!=trial["input_sha256"]:
        raise ValueError("Different exchange witnesses")
    for r in (control,trial):
        if (r["oldest_reproduced"] is not True or r["quote_verified_decoder_calls"]!=len(r["trace"])
                or len(r["trace"])!=len(r["exchange_by_tick"])):
            raise ValueError("Missing quote or historical control")
        for n,t in zip(r["exchange_by_tick"],r["trace"]):
            if (type(n) is not int or not 0<=n<=min(4,t["prefix"]) or n and (t["tick"]%2 or t["phase"]!="sparse"
                    or t["q"]-t["prefix"]-t["mandatory_q"]<3*n)):
                raise ValueError("Move outside the paid unresolved budget")
        if sum(r["exchange_by_tick"])!=r["total_exchanged_rows"]:
            raise ValueError("Per-tick exchange count mismatch")
    if (control["total_exchanged_rows"]!=0 or
            first!=next((i for i,n in enumerate(trial["exchange_by_tick"]) if n),None)):
        raise ValueError("Wrong first physical intervention")
    if first is None:
        if (control["trace"], control["output_ids"])!=(trial["trace"],trial["output_ids"]):
            raise ValueError("Changed a request without exchanging rows")
    else:
        if type(first) is not int or not 0<=first<min(len(r["trace"]) for r in (control,trial)):
            raise ValueError("Not a matched intervention")
        a,b=(r["trace"][first] for r in (control,trial))
        if (control["trace"][:first]!=trial["trace"][:first] or
                any(a[k]!=b[k] for k in ("tick","offer","q","mandatory_q","phase","head"))
                or b["prefix"]-a["prefix"]!=trial["exchange_by_tick"][first]):
            raise ValueError("Not an equal-width exchange")
    return first is not None


def unified_fixed(data, budgets, *, rank_quota=None, exchange=False, spawn=False):
    from bdlm_serving.evaluation.frozen import sha
    from experiments.t2t.serving_slots import unified90 as u, gemma_fixed_caps as caps
    from experiments.t2t.serving_slots.external_report import cost_gaps
    from experiments.t2t.serving_slots.mask_pair_report import efficiency_interval
    from tools.prefix_cohorts import canonical_model, load_cases, normalize_request, select_window_cases, _window_row
    if not budgets or len(set(budgets)) != len(budgets) or not set(budgets) <= set(u.SLOTS):
        raise ValueError("Give distinct registered Gemma slots")
    if exchange and (list(budgets)!=[128] or rank_quota!=32):
        raise ValueError("Only Q128/K32 has a registered exchange")
    if spawn and (list(budgets)!=[512] or rank_quota is not None or exchange):
        raise ValueError("Earlier tail admission only applies to Q512")
    samples = data/"samples.json"
    if sha(samples) != caps.SAMPLES_SHA:
        raise ValueError("Wrong samples")
    cases = load_cases(samples)
    for b in sorted(budgets):
        primary = f"q{b}"
        base_group = next(g for g in u.GROUPS if primary in g)
        group = base_group if rank_quota is None else (f"q{b}_k{rank_quota}_random",f"q{b}_k{rank_quota}_oldest")
        if exchange:
            group=u.EXCHANGE
        if spawn:
            group=u.SPAWN
        if group not in u.GROUPS:
            raise ValueError("Unregistered rank comparison")
        path = data/("gemma-unified90-"+"-".join(group))
        m = u.read(path/"run.json")
        if (m["suite"] != "frozen90" or m["device"] != "cuda:0" or m["samples_sha256"] != caps.SAMPLES_SHA
                or m["model"] != "diffusiongemma" or m["indices"] != list(range(90))
                or set(m["arms"]) != set(group) or u.read(path/"returned.json") != {"gpu":0,"restored":True}
                or any(u.read(path/f"complete-{p}.json") != {"policy":p,"requests":90} for p in group)):
            raise ValueError("Missing qualification for unified arm")
        refs = []
        if b <= 256:
            capped = data/caps.OUTPUT if b <= 64 else data.parent/"private-optimus-cap90-v1/gemma-optimus-cap90-v1"
            refs.extend((capped, f"{mode}_cap_{b}") for mode in ("ibs","obs"))
        if b >= 256:
            refs.append((data/"gemma-wave-reference90", f"flow_w{b//256}"))
        if b == 512:
            refs.append((data/"gemma-fixed512-active-w2","active_w2"))
        base = {}
        raw_random = {}
        intervention = {}
        if spawn:
            if (m.get("spawn_thresholds")!={primary:.8,group[1]:.5} or m.get("quote_costs") is not True
                    or m["arms"]!={p:[[512],True,{}] for p in group}
                    or m["wave_config"]["flowblock_spawn_threshold"]!=.8):
                raise ValueError("Not the registered two-threshold causal pair")
            references={}
            for q in (256,512):
                p=f"q{q}"
                old=data/("gemma-unified90-"+"-".join(next(g for g in u.GROUPS if p in g)))
                oldmeta=u.read(old/"run.json")
                if (any(oldmeta[k]!=m[k] for k in ("revision","model","suite","indices","samples_sha256","wave_config",
                        "native_config","torch","transformers","rng","launcher_sha256","device"))
                        or oldmeta["arms"][p]!=[[q],True,{}]
                        or u.read(old/"returned.json")!={"gpu":0,"restored":True}
                        or u.read(old/f"complete-{p}.json")!={"policy":p,"requests":90}):
                    raise ValueError("Missing frozen reference for the spawn study")
                references[p]={c["index"]:u.read(old/f"{c['index']:02d}-{p}.json") for c in cases}
                base[p]={c["index"]:unified_fixed_row(references[p][c["index"]],c,p,q) for c in cases}
        if rank_quota is not None:
            refpath=data/("gemma-unified90-"+"-".join(base_group))
            refmeta=u.read(refpath/"run.json")
            policy_ok=(m.get("exchange")==u.EX_SPEC and m.get("quote_costs") is True if exchange
                       else m["prefix_rank"]=={p:u.RANKS[p] for p in group})
            if (not policy_ok or
                    any(refmeta[k]!=m[k] for k in ("revision","suite","model","indices","samples_sha256","wave_config",
                    "native_config","torch","transformers","rng","launcher_sha256","device")) or
                    refmeta["arms"][primary]!=[[b],True,{}] or u.read(refpath/"returned.json")!={"gpu":0,"restored":True}
                    or u.read(refpath/f"complete-{primary}.json")!={"policy":primary,"requests":90}):
                raise ValueError("Wrong rank or zero-prefix control")
            # Different engine revisions: both use identical registered inputs, checked per request below.
            base_raw={c["index"]:u.read(refpath/f"{c['index']:02d}-{primary}.json") for c in cases}
            base[primary]={c["index"]:unified_fixed_row(base_raw[c["index"]],c,primary,b) for c in cases}
            if exchange:
                rankpath=data/"gemma-unified90-q128_k32_random-q128_k32_oldest"
                if (u.read(rankpath/"returned.json")!={"gpu":0,"restored":True}
                        or u.read(rankpath/f"complete-{group[0]}.json")!={"policy":group[0],"requests":90}):
                    raise ValueError("Missing official oldest control")
                random="q128_slack_random"
                base[random]={}
                for c in cases:
                    i=c["index"]
                    raw=u.read(refpath/f"{i:02d}-{random}.json")
                    if raw["input_sha256"]!=base_raw[i]["input_sha256"]:
                        raise ValueError("Wrong independent random control")
                    base[random][i]=unified_fixed_row(raw,c,random,b)
        for folder,label in refs:
            meta=u.read(folder/"run.json")
            if (meta["samples_sha256"] != caps.SAMPLES_SHA or meta["suite"] != "frozen90"
                    or label not in meta["policies"] or u.read(folder/f"complete-{label}.json") !=
                    dict(policy=label,requests=90)):
                raise ValueError("Incomplete baseline")
            if "_cap_" in label:
                if (not meta["hard_q_cap"] or meta["cap_excludes"] != "encoder" or
                        u.read(folder/"qualified-logits.json")["first_step_logits"] != "bit_identical"):
                    raise ValueError("Not a qualified decoder-cap arm")
            elif (meta["block_size"],meta["flowblock_spawn_threshold"]) != (256,0.8):
                raise ValueError("Wrong FlowBlock config")
            base[label] = {}
            for c in cases:
                i=c["index"]
                raw=u.read(folder/f"{i:02d}-{label}.json")
                if raw["policy"] != label or canonical_model(raw["model"]) != "gemma":
                    raise ValueError("Wrong baseline endpoint")
                r=normalize_request(raw,c,"official fixed control")
                if r["Q"] != r["physical_Q"]:
                    raise ValueError("Physical/raw mismatch in Gemma reference")
                base[label][i]=r
        for arm in (a for a in group if not a.endswith("_open") and not (spawn and a==primary)):
            if m["arms"][arm] != [[b],True,{str(k):v for k,v in u.ARMS[arm][2].items()}]:
                raise ValueError("Wrong causal mode, prefix quota or fixed slot")
            ours={}
            for c in cases:
                i=c["index"]
                raw=u.read(path/f"{i:02d}-{arm}.json")
                ours[i]=unified_fixed_row(raw,c,arm,b)
                if spawn:
                    control=u.read(path/f"{i:02d}-{primary}.json")
                    twin=references[primary][i]
                    if (any(control[k]!=twin[k] for k in (*u.KEYS,"model","policy","input_sha256","output_ids","output_text",
                                "output_tokens","trace","generation_forwards","generation_physical_padded_rows",
                                "first_geometry_sha256","decoder_layers","projection_modules","truncated"))
                            or references["q256"][i]["input_sha256"]!=control["input_sha256"]):
                        raise ValueError("Original control failed frozen Q512 replay or different Q256 input")
                    intervention[i]=u.spawn_witness(control,raw) is not None
                    ours[i].update(dual_ticks=raw["dual_ticks"],base_dual_ticks=control["dual_ticks"])
                if rank_quota is not None:
                    if (raw["input_sha256"]!=base_raw[i]["input_sha256"] or
                            raw["first_opportunity"] is not None or any(e["prefix"]>rank_quota for e in raw["trace"])):
                        raise ValueError("Wrong rank inputs, cross-block execution or quota")
                    if arm==group[0]:
                        raw_random[i]=raw
                        if exchange:
                            twin=u.read(rankpath/f"{i:02d}-{arm}.json")
                            if any(twin[k]!=raw[k] for k in (*u.KEYS,"input_sha256","output_ids","trace",
                                                             "generation_forwards","generation_physical_padded_rows")):
                                raise ValueError("Oldest control replay failed")
                    else:
                        intervention[i]=(exchange_evidence(raw_random[i],raw) if exchange
                                         else rank_evidence(raw_random[i],raw,rank_quota))
                    if exchange:
                        ours[i]["swaps"]=raw["total_exchanged_rows"]
                        ours[i]["swap_ticks"]=sum(bool(n) for n in raw["exchange_by_tick"])
                    if not exchange and b==128 and rank_quota==32 and arm==group[0]:
                        twin=u.read(refpath/f"{i:02d}-q128_slack_random.json")
                        if any(twin[k]!=raw[k] for k in ("input_sha256","output_ids","trace","generation_forwards","generation_physical_padded_rows")):
                            raise ValueError("Registered Q128/K32 random did not replicate")
            for w in (30,60,90):
                sub = select_window_cases(cases,w)
                ids = [c["index"] for c in sub]
                for label, rows in base.items():
                    r = _window_row(sub,rows,ours,w,"all","gemma",label,arm,b)
                    r.update(cost_gaps([rows[i]["physical_Q"] for i in ids],[ours[i]["Q"] for i in ids]))
                    r["efficiency_ci_low"],r["efficiency_ci_high"] = efficiency_interval(sub,rows,ours)
                    for k in ("prefix_ticks","prefix_rows","encoder_f","decoder_f","overflow_ticks","overflow_rows","strict_edits",
                              "excess_bootstrap","excess_verify","excess_sparse","excess_dense","excess_tail_dirty_drain"):
                        r["ours_"+k] = sum(ours[i][k] for i in ids)
                    r["ours_prefix_requests"] = sum(ours[i]["prefix_ticks"]>0 for i in ids)
                    if spawn:
                        r.update(earlier_spawn_requests=sum(intervention[i] for i in ids),
                                 ours_dual_ticks=sum(ours[i]["dual_ticks"] for i in ids),
                                 q512_control_dual_ticks=sum(ours[i]["base_dual_ticks"] for i in ids))
                    if rank_quota is not None:
                        r["prefix_quota"]=rank_quota
                        r["exchange_exposed_requests" if exchange else "ranking_changed_requests"]=(sum(intervention[i] for i in ids)
                            if label==group[0] and arm==group[1] else "")
                        if exchange:
                            r.update(exchanged_rows=sum(ours[i]["swaps"] for i in ids),
                                     exchange_ticks=sum(ours[i]["swap_ticks"] for i in ids))
                        for k in ("sparse_q","sparse_mandatory_q","sparse_prefix_q","sparse_mask_only_q"):
                            r["ours_"+k]=sum(ours[i][k] for i in ids)
                    yield r
            if ((arm == primary and any("slack" in a for a in group)) or
                    (rank_quota is not None and arm==group[0])):
                base[arm]=ours


def unified_adjacent(data):
    """Paired fixed endpoints on the same executable. These are not live switches."""
    from math import isclose
    from bdlm_serving.evaluation.frozen import sha
    from experiments.t2t.serving_slots import unified90 as u
    from experiments.t2t.serving_slots.mask_pair_report import efficiency_interval
    from experiments.t2t.serving_slots.external_report import cost_gaps
    from tools.prefix_cohorts import load_cases, select_window_cases, _window_row

    if sha(data/"samples.json")!=u.caps.SAMPLES_SHA:
        raise ValueError("Wrong frozen suite")
    cases=load_cases(data/"samples.json")
    prev=None
    common=None
    fingerprints=None
    for b in u.SLOTS:
        arm=f"q{b}"
        group=next(g for g in u.GROUPS if arm in g)
        folder=data/("gemma-unified90-"+"-".join(group))
        m=u.read(folder/"run.json")
        signature={k:m[k] for k in ("revision","model","device","suite","wave_config","native_config",
                                    "samples_sha256","indices","sources","torch","transformers","rng","launcher_sha256")}
        if (m["indices"]!=list(range(90)) or m["samples_sha256"]!=u.caps.SAMPLES_SHA
                or m["suite"]!="frozen90" or m["device"]!="cuda:0" or m["model"]!="diffusiongemma"
                or m["arms"][arm]!=[[b],True,{}] or u.read(folder/f"complete-{arm}.json")!={"policy":arm,"requests":90}
                or u.read(folder/"returned.json")!={"gpu":0,"restored":True} or common is not None and signature!=common):
            raise ValueError("Endpoint run manifests differ in more than the offer")
        common=signature
        rows={}
        digests={}
        for c in cases:
            i=c["index"]
            raw=u.read(folder/f"{i:02d}-{arm}.json")
            rows[i]=unified_fixed_row(raw,c,arm,b)
            digests[i]=raw["input_sha256"]
        if fingerprints is not None and fingerprints!=digests:
            raise ValueError("Paired prompt fingerprint mismatch")
        fingerprints=digests
        # Independently reconcile against the published fixed-slot table at each endpoint.
        with (HERE/f"controls/gemma-unified-q{b}-vs-baselines.csv").open() as f:
            published={int(r["window"]):r for r in csv.DictReader(f)
                       if r["arm_policy"]==arm and r["scope"]=="all"}
        for w in (30,60,90):
            sub=select_window_cases(cases,w)
            items=[rows[c["index"]] for c in sub]
            ref=published[w]
            if (any(sum(r[k] for r in items)!=int(ref["arm_"+k]) for k in ("N","F","physical_Q","correct"))
                    or not isclose(sum(r["tpf"] for r in items)/w,float(ref["arm_macro_tpf"]),rel_tol=1e-12)):
                raise ValueError("Published endpoint differs from the re-scored requests")
            if prev is None:
                continue
            lo,old=prev
            r=_window_row(sub,old,rows,w,"all","gemma",f"q{lo}",arm,None)
            ids=[c["index"] for c in sub]
            q0,q1=([side[i]["physical_Q"] for i in ids] for side in (old,rows))
            yield dict(r, low=lo, high=b, **cost_gaps(q0,q1),
                       requests_less_Q=sum(y<x for x,y in zip(q0,q1)),
                       requests_equal_Q=sum(y==x for x,y in zip(q0,q1)),
                       requests_more_Q=sum(y>x for x,y in zip(q0,q1)),
                       efficiency_ci_low=(eff:=efficiency_interval(sub,old,rows))[0],efficiency_ci_high=eff[1])
        prev=b,rows


def gemma_capped_controls(data):
    from bdlm_serving.evaluation.frozen import sha
    from experiments.t2t.prefix_window.score_run import snapshot, link_requests
    from experiments.t2t.serving_slots import gemma_fixed_caps as caps
    from experiments.t2t.serving_slots.mask_pair_report import efficiency_interval
    from tools.prefix_cohorts import load_cases, load_request_set, select_window_cases, _window_row

    samples = data/"samples.json"
    cases = load_cases(samples)
    target = data/caps.OUTPUT
    digest = sha(samples)
    receipt = caps.read(target/"protected-complete.json")
    if (digest != caps.SAMPLES_SHA or receipt != dict(status="complete",model="gemma",
            policies=list(caps.POLICIES),requests=90,gpu=0,holder_restored=True,
            samples_sha256=digest,run=target.name)):
        raise ValueError("Not the qualified GPU0 fixed-cap matrix")
    caps.verify_run(target,data,cases)
    cache = data/"fixed-slot-scores"/".inputs"
    fixed = {(r["slot"],r["cohort"]):r for r in fixed_rows(HERE.parent/"prefix_window")
             if r["model"] == "gemma"}
    for q, suffix in ((32,"gemma-q32"),(64,"gemma-grid-k1")):
        ours = f"t2t_q{q}_f0_m{q}_hard_prefix_zero"
        def table(path, label):
            rows, model, policy, nominal = load_request_set(path,cases,"comparison",label)
            if (model,policy,nominal) != ("gemma",label,q):
                raise ValueError("Wrong candidate or baseline identity")
            return {r["index"]:r for r in rows}
        our = table(snapshot(data/suffix,ours,digest,cache),ours)
        for mode in ("ibs","obs"):
            label = f"{mode}_cap_{q}"
            base = table(link_requests(target,label,cache),label)
            for w in (30,60,90):
                sub = select_window_cases(cases,w)
                r = _window_row(sub,base,our,w,"all","gemma",label,ours,q)
                if any(float(fixed[q,r["cohort"]][k]) != r["arm_"+k]
                       for k in ("correct","N","F","physical_Q","macro_tpf")):
                    raise ValueError("Frozen K0 endpoint not reproduced")
                r["efficiency_ci_low"],r["efficiency_ci_high"] = efficiency_interval(sub,base,our)
                yield r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="also compute adjacent-slot pairs from private runs")
    parser.add_argument("--true64",type=Path,help="Publish cross-budget comparisons for the completed true two-block run")
    parser.add_argument("--gemma-fixed-caps",type=Path,help="Compare Q32/Q64 K0 with decoder-capped IBS/OBS")
    parser.add_argument("--unified-slots",type=int,nargs="+",
                        help="Rebuild per-slot tables from completed GPU90 arms; use with --data")
    parser.add_argument("--unified-adjacent",action="store_true",
                        help="Compare adjacent fixed Gemma offers on the identical measured executor")
    parser.add_argument("--exchange",action="store_true",help="Compare the paired Q128/K32 equal-width exchange")
    parser.add_argument("--spawn50",action="store_true",help="Validate and score earlier Q512 tail admission vs Q256/Q512/W2")
    parser.add_argument("--rank-groups",nargs="+",metavar="Q:K",
                        help="Compare ranked prefixes against each other, K0 and capped baselines")
    args = parser.parse_args()
    if args.spawn50:
        if not args.data:
            parser.error("--spawn50 requires --data")
        write_rows(HERE/"controls/gemma-unified-q512-spawn50.csv",unified_fixed(args.data,[512],spawn=True))
        raise SystemExit(0)
    if args.exchange:
        if not args.data:
            parser.error("--exchange requires --data")
        write_rows(HERE/"controls/gemma-unified-q128-k32-exchange.csv",unified_fixed(args.data,[128],rank_quota=32,exchange=True))
        raise SystemExit(0)
    if args.unified_adjacent:
        if not args.data:
            parser.error("--unified-adjacent requires --data")
        write_rows(HERE/"controls/gemma-unified-adjacent.csv", unified_adjacent(args.data))
        raise SystemExit(0)
    if args.rank_groups:
        if not args.data:
            parser.error("--rank-groups requires --data")
        for spec in args.rank_groups:
            b,k=map(int,spec.split(":"))
            write_rows(HERE/f"controls/gemma-unified-q{b}-k{k}-rank.csv",
                       unified_fixed(args.data,[b],rank_quota=k))
        raise SystemExit(0)
    if args.unified_slots:
        if not args.data:
            parser.error("--unified-slots requires --data")
        for b in args.unified_slots:
            write_rows(HERE/f"controls/gemma-unified-q{b}-vs-baselines.csv", unified_fixed(args.data,[b]))
        raise SystemExit(0)
    if args.gemma_fixed_caps:
        write_rows(HERE/"controls/gemma-fixed-caps-vs-k0.csv", gemma_capped_controls(args.gemma_fixed_caps))
        raise SystemExit(0)
    if args.true64:
        write_rows(HERE/"controls/llada21-true64.csv", true64_controls(args.true64))
        raise SystemExit(0)
    source = HERE.parent/"prefix_window"
    write_rows(HERE/"fixed-k0.csv", fixed_rows(source))
    if args.data:
        write_rows(HERE/"adjacent-k0.csv", adjacent(args.data,args.data/"fixed-slot-scores",source))
