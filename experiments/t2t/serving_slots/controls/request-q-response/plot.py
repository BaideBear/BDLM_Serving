"""Plot paired GPU labels only, not model predictions. No weights or raw logs needed."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(path, out):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({(r["model"], r["executor_id"], r["continuation_q"]) for r in rows}) != 1:
        raise ValueError("Choose nonempty observations from one executor and continuation")
    ncols = min(2, len(rows))
    nrows = math.ceil(len(rows)/ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(5.7*ncols, 3.4*nrows+1.1),
                             constrained_layout=True, sharey=True)
    fig.suptitle(f"{rows[0]['model']}  |  One budget decision, then Q{rows[0]['continuation_q']}\n"
                 "Measured suffix TPF on matched GPU replays", ha="center", fontsize=15)
    for ax, r in zip(axes.flat, rows):
        arms = sorted(r["arms"], key=lambda a: a["q"])
        y = [(a["final_n"]-r["pre_delivered_n"])/(a["suffix_decoder_f"]+a["suffix_encoder_f"]) for a in arms]
        s = r["state"]
        xs = range(len(arms))
        ax.plot(xs, y, "o-", color="#17647A", linewidth=2, zorder=2)
        for x, val in zip(xs, y):
            ax.annotate(f"{val:.3f}", (x, val), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=9, color="#18353F")
        ax.margins(x=.09)
        ax.set_ylim(bottom=0)
        ax.set_xticks(list(xs), [f"{a['q']}\n({a['first_step_decoder_q']})" for a in arms])
        ax.set(xlabel="Offered Q   (actual first-step decoder Q)", ylabel="Effective tokens / suffix F")
        ax.set_title(f"{r['request_id']}  |  {s['head_mask_rows']} MASK, {s['dirty_rows']} dirty\n"
                     f"after {r['history_steps']} steps at Q{r['history_q']} · "
                     f"{sum(a['correct'] for a in arms)}/{len(arms)} arms correct", fontsize=10)
        ax.grid(axis="y", color="#D5DDE1", linewidth=.65)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(rows):]:
        ax.set_visible(False)
    upper = max((a["final_n"]-r["pre_delivered_n"])/(a["suffix_decoder_f"]+a["suffix_encoder_f"])
                for r in rows for a in r["arms"])
    axes.flat[0].set_ylim(0, max(.1, upper*1.18))
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=root/"llada-first-wave.jsonl")
    ap.add_argument("--out", type=Path, default=root/"llada-first-wave.png")
    a = ap.parse_args()
    plot(a.input, a.out)
