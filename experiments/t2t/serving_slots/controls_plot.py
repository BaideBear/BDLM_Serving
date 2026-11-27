"""Rebuild the matched-cohort Optimus comparison from the public aggregate CSVs."""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = ("correct", "macro_tpf", "pooled_token_efficiency", "N", "F", "physical_Q", "q_per_f")


def comparison(root):
    out = {}
    for q in (32, 64, 128, 256):
        for ref in ("ibs", "obs"):
            for label in (("zero", "slack") if q in (32,128) else ("zero",)):
                path = root / f"gemma-{ref}_{q}-{label}.csv"
                with path.open() as f:
                    rows = [r for r in csv.DictReader(f) if r["scope"] == "all"]
                if [int(r["requests"]) for r in rows] != [30, 60, 90]:
                    raise ValueError(f"Invalid cohorts: {path}")
                for r in rows:
                    if (int(r["nominal_q"]) != q or r["baseline_policy"] != f"{ref}_{q}"
                            or r["model"] != "gemma"):
                        raise ValueError(f"Wrong source: {path}")
                    for side, method in (("baseline", ref.upper()), ("arm", label)):
                        key = (q, method, int(r["requests"]))
                        row = dict(q=q, method=method, cohort=r["cohort"], n=r["requests"],
                                   **{m:r[f"{side}_{m}"] for m in METRICS})
                        if out.setdefault(key, row) != row:
                            raise ValueError(f"Inconsistent duplicate for {key}")
    return out


def mixed_plot(source, target, matched=False, fixed_low=False, unified=False, rank=False, dose_q=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from itertools import groupby

    metrics = ("correct", "macro_tpf", "pooled_token_efficiency", "physical_Q")
    points = {}
    sources = ([p for q in (32,64,128,256,512) if (p := source/f"gemma-unified-q{q}-vs-baselines.csv").is_file()]
               if unified else [source])
    if dose_q is not None:
        if not rank or not source.is_dir():
            raise ValueError("Dose plot needs the directory of published rank tables")
        sources=sorted(source.glob(f"gemma-unified-q{dose_q}-k*-rank.csv"),
                       key=lambda p:int(p.stem.split("-")[3][1:]))
    for path in sources:
      with path.open() as f:
        for row in csv.DictReader(f):
            if row["window"] != "90" or row["scope"] != "all":
                continue
            if matched and "_cap_" not in row["baseline_policy"]:
                continue
            for side in ("arm", "baseline"):
                key = row[side+"_policy"]
                values = tuple(float(row[f"{side}_{m}"]) for m in metrics)
                if points.setdefault(key, values) != values:
                    raise ValueError(f"Inconsistent {key} result in {source}")
    title = {"flow_w1":"FlowBlock W1", "flow_w2":"FlowBlock W2", "active_w2":"Active-only W2",
             "replay":"K0 on all offers", "replay_q128_prefix32":"Q128 spare seats: K32"}
    def compact(slots):
        return " / ".join(q+(f" x{len(g)}" if len(g)>1 else "")
                          for q,group in groupby(slots.split("_")) for g in [list(group)])
    def label(name):
        if name in title:
            return title[name]
        if rank and name.startswith("q"):
            return name.upper().replace("_", " ") + (" (no prefix)" if "_" not in name else "")
        if name.startswith("q") and name[1:].split("_")[0].isdecimal():
            return "Unified "+name.upper().replace("_SLACK_RANDOM"," (random K32)")
        if name.startswith("t2t_q"):
            return "Ours K0 Q"+name.split("_")[1][1:]
        if name.startswith("slottrace_"):
            return "Ours: "+compact(name.removeprefix("slottrace_"))
        mode, slot = name.split("_", 1)
        cap = slot.startswith("cap_")
        return ("Capped " if cap else "Original ")+f"{mode.upper()} "+compact(slot.removeprefix("cap_"))
    if not points:
        raise ValueError("Missing frozen90 records")
    if dose_q is not None:
        order=[f"ibs_cap_{dose_q}", f"obs_cap_{dose_q}", f"q{dose_q}"]
        for p in sources:
            k=p.stem.split("-")[3]
            order.extend(f"q{dose_q}_{k}_{mode}" for mode in ("random","oldest"))
        if set(order)!=set(points):
            raise ValueError("Incomplete paired prefix dose table")
        points = {name:points[name] for name in order}
    if fixed_low:
        order = [name for q in (32,64) for name in (
            f"t2t_q{q}_f0_m{q}_hard_prefix_zero",f"ibs_cap_{q}",f"obs_cap_{q}")]
        if set(order) != set(points):
            raise ValueError("Missing or unexpected fixed-budget arm")
        points = {p:points[p] for p in order}
    elif matched:
        order=[]
        for plan in ("128","256","128_128_256_256","256_256_128_128"):
            own="slottrace_"+"_".join("256n" if x=="256" else x for x in plan.split("_"))
            order.extend([own,"ibs_cap_"+plan,"obs_cap_"+plan])
        if set(order)!=set(points):
            raise ValueError("Require the complete matched matrix")
        points={p:points[p] for p in order}
    n = len(points)
    profile=source.stem.rsplit("-profile-",1)[-1]
    if profile in ("transfer","dwell"):
        policy={"transfer":"replay","dwell":"dwell"}[profile]
        pair=(policy,policy+"_q128_prefix32")
        if set(points)!=set(pair):
            raise ValueError("Incomplete profile comparison")
        title.update({pair[0]:"K0 on all offers",pair[1]:"Q128 spare seats: K32"})
        points={p:points[p] for p in pair}
    fig, axes = plt.subplots(1, 4, figsize=(14, 1.8+n*.47), layout="constrained")
    colors = ["#116f74" if p.startswith(("slottrace","t2t_q","q","replay_q","dwell_q")) else
              "#6174a3" if p.startswith(("flow","active_w")) else "#bb713c" if "_cap_" in p else "#8c929a"
              for p in points]
    for j, (ax, heading) in enumerate(zip(axes, ("Correct / 90", "Tokens / forward",
                                                  "Token efficiency (%)", "Physical Q (thousands)"))):
        vals = [v[j] * (1, 1, 100, .001)[j] for v in points.values()]
        bars = ax.barh(range(n), vals, color=colors)
        ax.bar_label(bars, labels=[f"{v:.{(0,3,2,1)[j]}f}" for v in vals], padding=3, fontsize=9)
        ax.set_yticks(range(n), [label(p) for p in points] if not j else [])
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals)*1.34)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_title(heading, fontsize=11)
    detail="matched nominal offers vs capped Optimus" if matched else "high-first candidate vs Optimus and FlowBlock"
    if fixed_low:
        detail="Q32 and Q64: K0 vs decoder-capped Optimus"
    if unified:
        detail="fixed unified offers vs same-slot baselines; each row has 90 requests"
    if rank:
        slot=f"Q{dose_q}" if dose_q is not None else source.stem.split("-")[2].upper()
        detail="fixed "+slot+" | slack-only prefix selection vs capped Optimus"
        if source.stem.endswith("-exchange"):
            detail="fixed Q128 | oldest prefix: spare seats vs up to four MASK replacements on even ticks"
    if source.stem.endswith("-spawn50"):
        detail="fixed Q512 | 50% vs 80% tail admission, Q256 and FlowBlock W2"
    if source.stem.endswith("-profile-transfer"):
        detail="same five-slot external replay | economy vs Q128 spare-prefix candidate"
    if source.stem.endswith("-profile-dwell"):
        detail="external replay with consecutive Q128 offers | K0 vs Q128 spare-prefix candidate"
    fig.suptitle("DiffusionGemma | "+detail+" | frozen90")
    fig.savefig(target, dpi=150)
    plt.close(fig)


def llada_capped_plot(q, root, target):
    """Reconcile shared arms in the four paired comparisons before plotting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    controls = {}
    for mode in ("ibs", "obs"):
        expected = (
            (f"llada21-q{q}-{mode}-original-vs-capped.csv", f"optimus_{mode}_c{q}", f"optcap_{mode}_{q}"),
            (f"llada21-q{q}-vs-capped-{mode}.csv", f"optcap_{mode}_{q}", f"t2t_q{q}_f0_m{q}_hard_prefix_zero"),
        )
        for filename, baseline, arm in expected:
            with (root / filename).open() as f:
                record, = (r for r in csv.DictReader(f) if r["window"] == "90" and r["scope"] == "all")
            if (record["baseline_policy"], record["arm_policy"], record["model"], record["requests"]) != (
                baseline, arm, "llada21", "90"
            ):
                raise ValueError(f"Unexpected frozen90 arm: {filename}")
            for side, policy in [("baseline", baseline), ("arm", arm)]:
                values = tuple(float(record[f"{side}_{k}"]) for k in METRICS)
                if controls.setdefault(policy, values) != values:
                    raise ValueError(f"Disagreement between controls for {policy}")
    order = [f"optimus_ibs_c{q}", f"optcap_ibs_{q}", f"optimus_obs_c{q}", f"optcap_obs_{q}",
             f"t2t_q{q}_f0_m{q}_hard_prefix_zero"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.25), layout="constrained")
    labels = ["IBS original", "IBS capped", "OBS original", "OBS capped", "Ours, K0"]
    for i, (axis, col, factor, digits, title) in enumerate(zip(
        axes, ("correct", "macro_tpf", "pooled_token_efficiency", "q_per_f"),
        (1, 1, 100, 1), (0, 3, 2, 2),
        ("Correct / 90", "Tokens / forward", "Token efficiency (%)", "Physical Q / forward"),
    )):
        values = [controls[p][METRICS.index(col)] * factor for p in order]
        bars = axis.barh(range(5), values, color=["#9ba9b5", "#546f83", "#aaa4ab", "#756579", "#147a6b"])
        axis.bar_label(bars, labels=[f"{v:.{digits}f}" for v in values], padding=3, fontsize=9)
        axis.set_yticks(range(5), labels if i == 0 else [])
        axis.invert_yaxis()
        axis.set_xlim(0, max(values) * 1.29)
        axis.set_title(title, fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"LLaDA2.1 | nominal Q{q}, different physical costs | frozen90")
    fig.savefig(target, dpi=150)
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = comparison(HERE/"controls")
    with (HERE/"gemma-optimus.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(rows.values()))), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows.values())
    fig, axes = plt.subplots(4, 3, figsize=(11, 11.2), layout="constrained")
    for axrow, q in zip(axes, (32, 64, 128, 256)):
        order = ["IBS","OBS","zero"] + (["slack"] if q in (32,128) else [])
        for ax, col, desc in zip(axrow, METRICS, ("Correct / 90", "Tokens / forward", "Token efficiency (%)")):
            vals = [float(rows[q, method, 90][col]) for method in order]
            if col == "pooled_token_efficiency":
                vals = [v*100 for v in vals]
            bars = ax.bar(["IBS", "OBS", "K0", "Slack"][:len(order)], vals,
                          color=["#827b8b", "#afa7b5", "#34839b", "#d58636"])
            ax.bar_label(bars, labels=[f"{v:.0f}" if col == "correct" else f"{v:.2f}" for v in vals])
            ax.set_ylim(0, 96 if col == "correct" else max(vals)*1.20)
            ax.set_title(f"Q{q}: {desc}", fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("DiffusionGemma | same 90 requests, same nominal slot", fontsize=14)
    fig.savefig(HERE/"gemma-optimus.png", dpi=150)
    plt.close(fig)
    mixed_plot(HERE/"controls/gemma-mixed-vs-external.csv", HERE/"gemma-mixed-vs-external.png")
    matches=HERE/"controls/gemma-matched-offers-vs-external.csv"
    if matches.exists():
        with matches.open() as f:
            if any("_cap_" in r["baseline_policy"] for r in csv.DictReader(f)):
                mixed_plot(matches,HERE/"gemma-matched-offers.png",matched=True)
    low = HERE/"controls/gemma-fixed-caps-vs-k0.csv"
    if low.exists():
        mixed_plot(low,HERE/"gemma-q32-q64-capped.png",fixed_low=True)
    if list((HERE/"controls").glob("gemma-unified-q*-vs-baselines.csv")):
        mixed_plot(HERE/"controls",HERE/"gemma-unified-fixed.png",unified=True)
    for p in sorted((HERE/"controls").glob("gemma-unified-q*-k*-rank.csv")):
        mixed_plot(p, HERE/(p.stem+".png"), rank=True)
    for p in sorted((HERE/"controls").glob("gemma-unified-q*-k*-exchange.csv")):
        mixed_plot(p, HERE/(p.stem+".png"), rank=True)
    if list((HERE/"controls").glob("gemma-unified-q128-k*-rank.csv")):
        mixed_plot(HERE/"controls", HERE/"gemma-unified-q128-prefix-doses.png", rank=True,dose_q=128)
    spawn=HERE/"controls/gemma-unified-q512-spawn50.csv"
    if spawn.is_file():
        mixed_plot(spawn,HERE/(spawn.stem+".png"))
    for name in ("transfer","dwell"):
        path=HERE/f"controls/gemma-unified-profile-{name}.csv"
        if path.is_file():
            mixed_plot(path,HERE/(path.stem+".png"))
    for q in (4, 8, 16, 32):
        if all((HERE/"controls"/f"llada21-q{q}-{name}.csv").is_file()
               for m in ("ibs", "obs") for name in (f"vs-capped-{m}", f"{m}-original-vs-capped")):
            llada_capped_plot(q, HERE/"controls", HERE/f"llada21-q{q}-optimus-capped.png")


if __name__ == "__main__":
    main()
