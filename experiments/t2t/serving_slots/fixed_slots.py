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
    breakdown = {"excess_"+phase: sum(e["overflow"] for e in t if e["phase"] == phase) for phase in stages}
    return dict(**{k:case[k] for k in ("index","dataset","seed","task_id")}, **breakdown,
                N=n, F=f, Q=q, physical_Q=q, max_q=max(e["q"] for e in t),
                tpf=n/f, failed=0, truncated=raw["truncated"], correct=int(score(raw["output_text"],case)["correct"]),
                prefix_rows=sum(e["prefix"] for e in t), prefix_ticks=sum(e["prefix"]>0 for e in t),
                decoder_f=len(t), encoder_f=enc, overflow_ticks=sum(e["overflow"]>0 for e in t),
                overflow_rows=sum(e["overflow"] for e in t), strict_edits=sum(e["strict_edits"] for e in t))


def unified_fixed(data, budgets):
    from bdlm_serving.evaluation.frozen import sha
    from experiments.t2t.serving_slots import unified90 as u, gemma_fixed_caps as caps
    from experiments.t2t.serving_slots.external_report import cost_gaps
    from experiments.t2t.serving_slots.mask_pair_report import efficiency_interval
    from tools.prefix_cohorts import canonical_model, load_cases, normalize_request, select_window_cases, _window_row
    if not budgets or len(set(budgets)) != len(budgets) or not set(budgets) <= set(u.SLOTS):
        raise ValueError("Give distinct registered Gemma slots")
    samples = data/"samples.json"
    if sha(samples) != caps.SAMPLES_SHA:
        raise ValueError("Wrong samples")
    cases = load_cases(samples)
    for b in sorted(budgets):
        primary = f"q{b}"
        group = next(g for g in u.GROUPS if primary in g)
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
        for arm in (a for a in group if not a.endswith("_open")):
            if m["arms"][arm] != [[b],True,u.ARMS[arm][2]]:
                raise ValueError("Wrong causal mode, prefix quota or fixed slot")
            ours = {c["index"]: unified_fixed_row(u.read(path/f"{c['index']:02d}-{arm}.json"),c,arm,b) for c in cases}
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
                    yield r
            if arm == primary and len(group)>1 and any("slack" in a for a in group):
                base[primary]=ours


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
    args = parser.parse_args()
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
