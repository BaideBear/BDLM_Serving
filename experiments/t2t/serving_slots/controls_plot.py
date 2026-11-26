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


def mixed_plot(source, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ("correct", "macro_tpf", "pooled_token_efficiency", "physical_Q")
    points = {}
    with source.open() as f:
        for row in csv.DictReader(f):
            if row["window"] != "90" or row["scope"] != "all":
                continue
            for side in ("arm", "baseline"):
                key = row[side+"_policy"]
                values = tuple(float(row[f"{side}_{m}"]) for m in metrics)
                if points.setdefault(key, values) != values:
                    raise ValueError(f"Inconsistent {key} result in {source}")
    title = {"slottrace_256n_256n_128_128":"Ours: 256n x2, 128 x2",
             "flow_w1":"FlowBlock W1", "flow_w2":"FlowBlock W2"}
    def label(name):
        if name in title:
            return title[name]
        mode, slot = name.split("_", 1)
        cap = slot.startswith("cap_")
        slots = slot.removeprefix("cap_").replace("_", "/")
        return f"Optimus {mode.upper()} {slots}" + (" (capped)" if cap else "")
    if not points:
        raise ValueError("Missing frozen90 records")
    n = len(points)
    fig, axes = plt.subplots(1, 4, figsize=(14, 1.8+n*.47), layout="constrained")
    colors = ["#116f74" if p in title and p.startswith("slottrace") else
              "#6174a3" if p.startswith("flow") else "#bb713c" if "_cap_" in p else "#8c929a"
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
    fig.suptitle("DiffusionGemma | mixed slots vs Optimus and FlowBlock | frozen90")
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


if __name__ == "__main__":
    main()
