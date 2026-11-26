"""Rebuild the matched-cohort Optimus comparison from the public aggregate CSVs."""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = ("correct", "macro_tpf", "pooled_token_efficiency", "N", "F", "physical_Q", "q_per_f")


def comparison(root):
    out = {}
    for q in (32, 128):
        for ref in ("ibs", "obs"):
            for label in ("zero", "slack"):
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


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = comparison(HERE/"controls")
    with (HERE/"gemma-optimus.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(rows.values()))), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows.values())
    order = ("IBS", "OBS", "zero", "slack")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), layout="constrained")
    for axrow, q in zip(axes, (32, 128)):
        for ax, col, desc in zip(axrow, METRICS, ("Correct / 90", "Tokens / forward", "Token efficiency (%)")):
            vals = [float(rows[q, method, 90][col]) for method in order]
            if col == "pooled_token_efficiency":
                vals = [v*100 for v in vals]
            bars = ax.bar(["IBS", "OBS", "K0", "Slack"], vals,
                          color=["#827b8b", "#afa7b5", "#34839b", "#d58636"])
            ax.bar_label(bars, labels=[f"{v:.0f}" if col == "correct" else f"{v:.2f}" for v in vals])
            ax.set_ylim(0, 96 if col == "correct" else max(vals)*1.20)
            ax.set_title(f"Q{q}: {desc}", fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("DiffusionGemma | same 90 requests, same nominal slot", fontsize=14)
    fig.savefig(HERE/"gemma-optimus.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
