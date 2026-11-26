"""Rebuild the fixed-seat context comparison from scored, token-free tables."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from tools.budget_index import (paired_uncertainty, request_key, request_metrics,
                                stratified_draws)
from tools.render_hard_study import (ROOT, Point, all_rows, point, read_baselines, render,
                                     window_controls)
from tools.t2t_results import table


def scored_rows(folder, policy):
    summaries = [s for s in all_rows(folder / "llada21-online-summary.csv") if s["policy"] == policy]
    if len(summaries) != 1:
        raise ValueError(f"Expected one dataset=all summary for {policy}, found {len(summaries)}")
    with (folder / "llada21-online-requests.csv").open() as f:
        requests = [r for r in csv.DictReader(f) if r["policy"] == policy]
    return summaries[0], requests


def bundle(folder, policy, keys):
    summary, requests = scored_rows(folder, policy)
    return point(summary, "token_efficiency"), request_metrics(requests, keys, summary)


def report(out, root=ROOT):
    exp = root / "experiments/t2t"
    keys = sorted(request_key(r) for r in json.loads(
        (root / "benchmarks/frozen30/manifest.json").read_text())["samples"])
    draws = stratified_draws(keys)
    base = read_baselines(root / "experiments/baselines/summary.csv")
    groups, contrasts = [], []
    for q, cap in ((4, 1), (8, 1), (16, 2)):
        by = {
            "No edit reserve": bundle(exp / "hard", f"t2t_q{q}_f0_m{q}_hard_noedit", keys),
            "Position only": bundle(exp / "calibrated" if q == 16 else out,
                                    f"t2t_q{q}_f0_m{q-cap}_hard_hazard_fixed", keys),
            "Position + input context": bundle(out, f"t2t_q{q}_f0_m{q-cap}_hard_context_fixed", keys),
        }
        rows = [(k, style, by[k][0]) for k, style in zip(by, ("hard", "adaptive", "geometry"), strict=True)]
        groups.append((q, rows + window_controls("llada21", q, 32, base)))
        p, x = by["Position + input context"]
        for name in ("Position only", "No edit reserve"):
            b, y = by[name]
            contrasts.append(dict(q=q, cap=cap, reference=name,
                correct=p.correct, control_correct=b.correct, tpf=p.tpf, control_tpf=b.tpf,
                efficiency=p.efficiency, control_efficiency=b.efficiency, physical_Q=int(x[:, 3].sum()),
                control_physical_Q=int(y[:, 3].sum()), delta_correct=p.correct-b.correct,
                delta_tpf=p.tpf-b.tpf, delta_eff_pp=100*(p.efficiency-b.efficiency),
                **paired_uncertainty(x, y, draws)))
    table(out / "matched.csv", contrasts)
    render(out / "rollout.svg", "llada21", 32, groups, adaptive=True,
           study_title="Does input-context ranking improve delivery at fixed seats?")
    primary = [r for r in contrasts if r["reference"] == "Position only"]
    return sum(r["delta_correct"] > 0 and r["delta_tpf"] >= 0 for r in primary) >= 2 and all(
        r["delta_correct"] >= 0 for r in primary)


def gated_arm(folder, mode, keys, source, *, scored=None):
    mask = 8 if mode == "noedit" else 7
    policy = f"t2t_q8_f0_m{mask}_hard_{mode}"
    s, requests = scored if scored is not None else scored_rows(folder, policy)
    for r in [s, *requests]:
        if r["model"] != "llada21" or tuple(int(r[k]) for k in ("nominal_q", "fresh", "mask", "edit")) != (8, 0, mask, 8-mask):
            raise ValueError(f"Model or quota mismatch for {policy}")
    x = request_metrics(requests, keys, s)
    check = point(s, "token_efficiency")
    for r in requests:
        if (not 0 < int(r["Q"]) <= int(r["physical_Q"]) or int(r["N"]) < 0 or
                any(int(r[k]) not in (0, 1) for k in ("correct", "failed", "truncated")) or
                int(r["correct"]) + int(r["failed"]) > 1):
            raise ValueError(f"Invalid request work or outcome in {policy}")
    n, f, q, physical = (int(s[k]) for k in ("N", "F", "Q", "physical_Q"))
    if (any(sum(int(r[k]) for r in requests) != int(s[k]) for k in ("Q", "failed", "truncated")) or
            not math.isclose(check.efficiency, n/q, rel_tol=1e-12) or
            not math.isclose(check.q_per_f, q/f, rel_tol=1e-12)):
        raise ValueError(f"Summary disagrees with request totals for {policy}")
    # Use the same physical-Q denominator as the paired bootstrap, even if Q is unpadded.
    p = Point(mean(int(r["N"]) / int(r["F"]) for r in requests), check.correct, n/physical, physical/f)
    arm = dict(model="llada21", nominal_q=8, policy=policy, source=source, requests=len(x),
               correct=p.correct, failed=int(s["failed"]), truncated=int(s["truncated"]), N=n, F=f,
               Q=q, physical_Q=physical, macro_tpf=p.tpf, physical_efficiency=p.efficiency,
               physical_q_per_f=p.q_per_f)
    return arm, p, x


def gated_report(out, root=ROOT, *, crowded=False):
    exp = root / "experiments/t2t"
    fixed = exp / "calibrated/embedding_context/rollout"
    keys = sorted(request_key(r) for r in json.loads(
        (root / "benchmarks/frozen30/manifest.json").read_text())["samples"])
    draws = stratified_draws(keys)
    arms, plot_rows = {}, []
    zero = ("noedit", exp / "hard", "archive_zero", "Zero reserve", "hard")
    if crowded:
        base, rank, rand = ("context_once_" + s for s in ("base", "rank", "random"))
        configs = [(base, out, "measured", "One switch: base", "hard"),
                   (rank, out, "measured", "One switch: ranked", "geometry"),
                   (rand, out, "measured", "One switch: random", "placebo"),
                   (*zero[:3], "No edit (historical)", "hard"),
                   ("context_gate", fixed.parent / "gated", "archive_gated", "Repeated gate (directed)", "adaptive")]
        comparisons = {rank: [(base, "primary_feasibility"), (rand, "same_gate_placement")],
                       rand: [(base, "descriptive")]}
    else:
        configs = [zero,
            ("context_fixed", fixed, "archive_fixed", "Fixed embedding", "adaptive"),
            ("context_slack", out, "measured", "Slack: ranked", "geometry"),
            ("context_slack_random", out, "measured", "Slack: random", "placebo"),
            ("context_gate", out, "measured", "Gate: ranked", "geometry"),
            ("context_gate_random", out, "measured", "Gate: random", "placebo")]
        comparisons = {m: [(m + "_random", "same_gate_placement"), ("noedit", "primary_feasibility"),
                           ("context_fixed", "fixed_embedding")] for m in ("context_slack", "context_gate")}
        comparisons["context_gate"].append(("context_slack", "ranked_vs_ranked"))
    snapshots = {}
    for mode, folder, source, label, style in configs:
        scored = None
        if crowded and mode in ("noedit", base):
            mask = 8 if mode == "noedit" else 7
            scored = scored_rows(folder, f"t2t_q8_f0_m{mask}_hard_{mode}")
            snapshots[mode] = {request_key(r): r for r in scored[1]}
        arms[mode] = gated_arm(folder, mode, keys, source, scored=scored)
        plot_rows.append((label, style, arms[mode][1]))
    if crowded:
        for key in keys:
            for field in ("N", "F", "Q", "physical_Q", "max_q", "correct", "failed", "truncated"):
                if int(snapshots[base][key][field]) != int(snapshots["noedit"][key][field]):
                    raise ValueError(f"One-switch base differs from noedit: request {key[0]}, {field}")
    baselines = read_baselines(root / "experiments/baselines/summary.csv")
    for kind in ("ibs", "obs"):
        if ("llada21", 8, kind) not in baselines:
            raise ValueError(f"Missing historical Optimus {kind.upper()}8")
        plot_rows.append((f"Optimus {kind.upper()}8 (historical)", kind, baselines["llada21", 8, kind]))

    contrasts, per_request, feasibility = [], [], {}
    for ranked, refs in comparisons.items():
        a, p, x = arms[ranked]
        for ref, purpose in refs:
            b, control, y = arms[ref]
            passed = ""
            if purpose == "primary_feasibility":
                passed = p.correct > control.correct and p.tpf >= control.tpf
                feasibility[a["policy"]] = passed
            # Symmetric two-factor identity for N*(1/F); descriptive, not causal.
            length = (x[:, 1] - y[:, 1]) * (1/x[:, 2] + 1/y[:, 2]) / 2
            forwards = (x[:, 1] + y[:, 1]) * (1/x[:, 2] - 1/y[:, 2]) / 2
            for key, xi, yi, dn, df in zip(keys, x, y, length, forwards, strict=True):
                obs = xi[1]/xi[2] - yi[1]/yi[2]
                if not math.isclose(dn + df, obs, abs_tol=1e-10, rel_tol=1e-10):
                    raise ValueError("TPF decomposition does not match paired request")
                per_request.append(dict(policy=a["policy"], reference=b["policy"], index=key[0],
                                        dataset=key[1], delta_correct=int(xi[0]-yi[0]),
                                        delta_N=int(xi[1]-yi[1]), delta_F=int(xi[2]-yi[2]),
                                        delta_physical_Q=int(xi[3]-yi[3]), delta_TPF=obs,
                                        length_component=dn, forwards_component=df))
            contrasts.append(dict(model="llada21", nominal_q=8, requests=len(x), purpose=purpose,
                policy=a["policy"], reference=b["policy"], primary_feasible=passed,
                delta_correct=p.correct-control.correct, delta_TPF=p.tpf-control.tpf,
                length_component=length.mean(), forwards_component=forwards.mean(),
                delta_eff_pp=100*(p.efficiency-control.efficiency),
                **{f"delta_{k}": a[k]-b[k] for k in ("N", "F", "physical_Q")},
                **paired_uncertainty(x, y, draws)))
    prefix = "crowded" if crowded else "gated"
    if crowded:
        dd = []
        for other, control in ((rand, base), ("context_gate", "noedit")):
            # (rank - base) - (other - control): the two base terms cancel
            # on EVERY paired draw, including across archives after exact replay parity.
            a, p, x = arms[rank]
            b, ref, y = arms[other]
            ci = paired_uncertainty(x, y, draws)
            dd.append(dict(policy=a["policy"], base=arms[base][0]["policy"], other=b["policy"],
                           other_base=arms[control][0]["policy"], requests=len(x),
                           delta_delta_correct=p.correct-ref.correct, delta_delta_TPF=p.tpf-ref.tpf,
                           **{f"delta_{k}": ci[k] for k in ("delta_correct_lo", "delta_correct_hi",
                                                           "delta_TPF_lo", "delta_TPF_hi")}))
        table(out / "crowded-delta-delta.csv", dd)
    render(out / f"{prefix}.svg", "llada21", 32, [(8, plot_rows)], adaptive=True,
           study_title="Crowded seats: one switch at Q8" if crowded else "Gated embedding placement at Q8")
    table(out / f"{prefix}-arms.csv", [a for a, _, _ in arms.values()])
    table(out / f"{prefix}-matched.csv", contrasts)
    table(out / f"{prefix}-request-deltas.csv", per_request)
    return feasibility


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, help="scored tables and destination (default: the study variant's directory)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--gated", action="store_true", help="Q8 same-gate randomization and zero-reserve feasibility")
    mode.add_argument("--crowded", action="store_true", help="Q8 isolated one-switch comparison")
    args = p.parse_args(argv)
    out = args.out or Path(__file__).resolve().parent / ("crowded" if args.crowded else "gated" if args.gated else "rollout")
    if args.gated or args.crowded:
        for policy, feasible in gated_report(out, crowded=args.crowded).items():
            reference = "one-switch base" if args.crowded else "zero reserve"
            print(f"{policy} feasibility vs {reference}: {'PASS' if feasible else 'FAIL'}")
    else:
        print("Passes matched-position preregistration:", report(out))


if __name__ == "__main__":
    main()
