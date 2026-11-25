"""Export and plot the frozen30 Gemma study without publishing model outputs."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ["native256", "flow_w1", "flow_w2", "flow_w3", "flow_w4", "b3"]
DEST = ROOT/"experiments/diffusiongemma-wave"


def records(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def save(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(native, wave):
    samples = json.loads((ROOT/"benchmarks/frozen30/manifest.json").read_text())["samples"]
    rows = []
    for policy in POLICIES:
        source = native if policy == "native256" else wave
        scores = [r for r in records(source/"scores.csv") if r["policy"] == policy]
        assert len(scores) == len(samples) == len({r["index"] for r in scores}) == 30
        for task, score in zip(samples, sorted(scores, key=lambda r: int(r["index"]))):
            idx = task["index"]
            r = json.loads((source/f"{idx:02d}-{policy}.json").read_text())
            assert all(r[k] == task[k] for k in ("index", "seed", "dataset", "task_id"))
            assert r["policy"] == policy and score["task_id"] == r["task_id"]
            assert score["status"] == r["status"] == "success" and not r["truncated"]
            assert score["correct"] in {"True", "False"}
            assert r["events"][0][0] == "encoder"
            events = r["events"][1:]
            decoder = [q for phase, q in events if phase == "decoder"]
            assert len(events) == r["generation_forwards"]
            assert sum(q for _, q in events) == r["generation_query_rows"]
            if policy == "flow_w1":
                ref = json.loads((native/f"{idx:02d}-native256.json").read_text())
                assert r["output_ids"] == ref["output_ids"] and r["events"] == ref["events"]
            active = len(decoder) if policy == "native256" else sum(t["active"] for t in r["trace"])
            if policy != "native256":
                assert decoder == [t["q"] for t in r["trace"]]
            rows.append({k: r[k] for k in ("policy", "index", "dataset", "task_id", "seed",
                "output_tokens", "generation_forwards", "generation_query_rows",
                "generation_physical_padded_rows")} | dict(correct=int(score["correct"] == "True"),
                    decoder_forwards=len(decoder), active_slot_sum=active))
    save(DEST/"requests.csv", rows)


def render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = records(DEST/"requests.csv")
    summary = []
    for policy in POLICIES:
        p = [r for r in rows if r["policy"] == policy]
        assert len(p) == len({r["task_id"] for r in p}) == 30
        for dataset in ["all", "gsm8k", "math500", "humaneval"]:
            s = p if dataset == "all" else [r for r in p if r["dataset"] == dataset]
            assert len(s) == (30 if dataset == "all" else 10)
            def total(k):
                return sum(int(r[k]) for r in s)
            n, f, q = map(total, ("output_tokens", "generation_forwards", "generation_query_rows"))
            summary.append(dict(policy=policy, dataset=dataset, requests=len(s), correct=total("correct"),
                output_tokens=n, generation_forwards=f, generation_query_rows=q, mean_tpf=mean(
                    int(r["output_tokens"])/int(r["generation_forwards"]) for r in s),
                token_efficiency=n/q, q_per_forward=q/f,
                mean_active_slots=total("active_slot_sum")/total("decoder_forwards")))
    save(DEST/"summary.csv", summary)
    allrows = [r for r in summary if r["dataset"] == "all"]
    labels = ["Native", "Flow W1", "Flow W2", "Flow W3", "Flow W4", "B3"]
    colors = ["#52616b"] + ["#287c80"]*4 + ["#a14b26"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.1), layout="constrained")
    for ax, metric, title, factor, fmt in zip(axes, ["mean_tpf","token_efficiency","correct"],
            ["Delivered tokens / forward", "Token efficiency (%)", "Correct out of 30"],
            [1, 100, 1], [".2f",".2f",".0f"]):
        vals = [r[metric]*factor for r in allrows]
        bars = ax.barh(labels, vals, color=colors)
        ax.invert_yaxis()
        ax.bar_label(bars, labels=[format(v, fmt) for v in vals], padding=4)
        ax.set_xlim(0, max(vals)*1.22)
        ax.set_title(title, loc="left", fontsize=11)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle("DiffusionGemma 26B-A4B  |  Frozen 30 requests, B=256", fontsize=14)
    (DEST/"figures").mkdir(exist_ok=True)
    fig.savefig(DEST/"figures/comparison.png", dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--wave", type=Path)
    args = parser.parse_args()
    if bool(args.native) != bool(args.wave):
        parser.error("Pass both private run directories, or neither to plot the public CSV")
    if args.native:
        export(args.native, args.wave)
    render()
