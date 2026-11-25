"""Partition observed hard-policy query rows, without inferring a split from the quota."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from bdlm_serving.evaluation.frozen import MANIFEST
from bdlm_serving.methods.t2t.protocol import verify_work


def request_split(row):
    verify_work(row)
    c = Counter()
    for event in row["events"]:
        if event["stage"] != "selected":
            continue
        ll = row["model"] == "llada21"
        positions = event["absolute_positions" if ll else "positions"]
        q = event["physical_q" if ll else "q"]
        fresh = event["mandatory_fresh"]
        if not ll and fresh != event["selected_fresh"]:
            raise ValueError("A mandatory fresh row was skipped")
        masks = event["selected_masked_before" if ll else "selected_mask"]
        extra = len(event["reserved_extra_edit_positions"])
        backfill = q - fresh - masks - extra
        if (not 0 <= backfill <= q or q != len(set(positions)) or
                (ll and any(n != q for n in event["layer_physical_q"]))):
            raise ValueError("Invalid physical split")
        c.update(selected_F=1, selected_Q=q, mandatory_Q=fresh, mask_Q=masks,
                 reserved_old_Q=extra, backfill_old_Q=backfill,
                 overflow_Q=event["overflow"])
    phases = row["q_by_phase"]
    if c["selected_Q"] != phases.get("selected", 0) or sum(phases.values()) != row["generation_physical_padded_rows"]:
        raise ValueError("Observed physical rows disagree with phase totals")
    c.update(total_Q=sum(phases.values()), total_F=row["generation_forwards"],
             other_Q=sum(phases.values())-c["selected_Q"])
    return c


def generate(runs, filters, out):
    manifest = json.loads(MANIFEST.read_text())["samples"]
    result = []
    for run in runs:
        config = json.loads((run/"run.json").read_text())
        offset = config.get("seed_offset", 0)
        for arm, policy in zip(config["arms"], config["policies"], strict=True):
            if arm["selection"] not in filters:
                continue
            if json.loads((run/f"complete-{policy}.json").read_text()) != dict(requests=30, policy=policy):
                raise ValueError(f"Incomplete {policy}")
            c = Counter()
            for m in manifest:
                row = json.loads((run/f"{m['index']:02d}-{policy}.json").read_text())
                if (any(row[k] != m[k] for k in ("index", "dataset", "task_id")) or
                        row["seed"] != m["seed"] + offset or row["policy"] != policy or
                        row["model"] != config["model"]):
                    raise ValueError("Arm identity mismatch")
                c.update(request_split(row))
            result.append(dict(model=config["model"], seed_offset=offset, budget=arm["budget"],
                policy=policy, **{k:c[k] for k in ("total_F", "total_Q", "selected_F",
                "selected_Q", "mandatory_Q", "mask_Q", "reserved_old_Q", "backfill_old_Q",
                "other_Q", "overflow_Q")}))
    if not result:
        raise ValueError("No matching complete hard-policy arms")
    keys = [(r["model"], r["seed_offset"], r["policy"]) for r in result]
    if len(set(keys)) != len(keys):
        raise ValueError("Overlapping arms")
    result.sort(key=lambda r:(r["model"], r["seed_offset"], r["budget"], r["policy"]))
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, result[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)


def render(source, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with source.open() as stream:
        rows = [r for r in csv.DictReader(stream) if int(r["selected_F"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, .37*len(rows)+1.9)),
                              gridspec_kw={"width_ratios":[1.7, 1]}, layout="constrained")
    palette = ["#287a70", "#68a5d2", "#d47e30", "#9c9aa6"]
    labels = ["Mandatory fresh", "MASK", "Reserved old", "Other old"]
    ticks = []
    for j,r in enumerate(rows):
        prefix = f"t2t_q{r['budget']}_f0_"
        mode = r["policy"].removeprefix(prefix).split("_hard_", 1)[1]
        ticks.append(f"{r['model']}  Q{r['budget']}  {mode}")
        start = 0
        for k,color in zip(("mandatory_Q","mask_Q","reserved_old_Q","backfill_old_Q"), palette):
            portion = 100*int(r[k])/int(r["selected_Q"])
            axes[0].barh(j, portion, left=start, color=color, height=.72)
            start += portion
        fraction = 100*int(r["other_Q"])/int(r["total_Q"])
        axes[1].barh(j, fraction, color="#67647b", height=.72)
        axes[1].text(fraction+1, j, f"{fraction:.1f}%", va="center", fontsize=9)
    for ax in axes:
        ax.set_xlim(0, 100)
        ax.set_ylim(len(rows)-.4, -.6)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Share of physical Q (%)")
    axes[0].set_yticks(range(len(rows)), ticks, fontsize=9)
    axes[1].set_yticks([])
    axes[0].set_title("Within sparse selection steps", loc="left")
    axes[1].set_title("Non-selection share of all generation Q", loc="left")
    axes[0].legend([matplotlib.patches.Patch(color=c) for c in palette], labels,
                   ncol=2, loc="upper left", bbox_to_anchor=(0, -.1), frameon=False)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, action="append")
    p.add_argument("--mode", action="append")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--figure", type=Path, help="Draw from --out alone or after generating it")
    args = p.parse_args()
    if bool(args.run) != bool(args.mode) or not (args.run or args.figure):
        p.error("Supply both --run and --mode to generate data; --figure can render existing CSV")
    if args.run:
        generate(args.run, set(args.mode), args.out)
    if args.figure:
        render(args.out, args.figure)
