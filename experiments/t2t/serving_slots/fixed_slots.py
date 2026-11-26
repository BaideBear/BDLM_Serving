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
    args = parser.parse_args()
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
