"""Reconcile measured fixed-K0 request work, separating protocol calls from sparse choices."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.evaluation.frozen import sha


HERE = Path(__file__).resolve().parent
FOLDERS = {
    "llada21": {4: "llada21-4-matched", 8: "llada-q8", 16: "llada21-16-matched", 32: "llada21-32-matched"},
    "gemma": {32: "gemma-q32", 64: "gemma-grid-k1", 128: "gemma-grid-k1", 256: "gemma-grid-k1"},
}
PHASES = ("bootstrap", "selected", "verify", "commit", "reencode")
COHORTS = ((0, 30, "original-0-29"), (30, 90, "added-30-89"), (0, 90, "all-0-89"))


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def selected_types(event, rows, q, offer):
    plan = event["prefix_plan"]
    groups = [plan[k] for k in ("fresh_rows", "zero_mask_rows", "zero_backfill_rows")]
    union = sorted(x for group in groups for x in group)
    require(plan["quota"] == 0 and not plan["added_rows"] and not plan["evicted_rows"]
            and union == sorted(set(union)) == plan["selected_rows"] == rows
            and event["mandatory_fresh"] == len(groups[0]) and q == len(union)
            and plan["budget"] == offer and max(0, q-offer) == plan["overflow"]
            and (q <= offer or q == len(groups[0])), "Invalid K0 row partition")
    return tuple(map(len, groups))


def events(row, model):
    if model == "llada21":
        prompt = 0
        for i, e in enumerate(row["events"]):
            q = e["physical_q"]
            require(e["layer_physical_q"] == [q]*20 and q == len(set(e["absolute_positions"])) == e["actual_q"],
                    "LLaDA projection mismatch")
            if e["phase"] == "prefill":
                require(i == 0 and e["stage"] == "prefill", "Noninitial free prefill")
                prompt = q
                continue
            yield e["stage"], q, e, [p - 32*e["block"] for p in e["absolute_positions"]]
        require(prompt == row["prefill_rows"], "Incorrect prefill")
    else:
        it = iter(row["events"])
        phys = row["physical_events"]
        require(phys and phys[0] == ["encoder", row["prefill_rows"]], "Incorrect prompt encoder")
        for kind, q in phys[1:]:
            if kind == "encoder":
                require(q == 256, "Noncanvas encoder")
                yield "reencode", q, None, None
            else:
                e = next(it, None)
                require(kind == "decoder" and e and e["q"] == q == len(set(e["positions"])),
                        "Gemma decoder mismatch")
                yield e["stage"], q, e, e["positions"]
        require(next(it, None) is None, "Extra decoder events")


def count(row, model, offer):
    c = Counter(N=row["output_tokens"], requests=1)
    widths = Counter()
    for stage, q, e, rows in events(row, model):
        require(stage in PHASES and type(q) == int and q > 0, "Invalid charged forward")
        widths[q] += 1
        over = max(0, q-offer)
        c.update(dict(F=1, Q=q, over_F=int(q>offer), over_Q=over, under_Q=max(0, offer-q)))
        for k, v in (("F", 1), ("Q", q), ("over_Q", over), ("over_F", int(q>offer))):
            c[f"{stage}_{k}"] += v
        if stage == "selected":
            fresh, mask, backfill = selected_types(e, rows, q, offer)
            c.update(dict(selected_fresh_Q=fresh, selected_mask_Q=mask, selected_backfill_Q=backfill))
    require(c["F"] == row["generation_forwards"] and c["Q"] == row["generation_physical_padded_rows"]
            and c["Q"] == offer*c["F"] + c["over_Q"] - c["under_Q"]
            and c["selected_Q"] == sum(c[k] for k in ("selected_fresh_Q", "selected_mask_Q", "selected_backfill_Q"))
            and all(c[f"{s}_F"] == row["forwards_by_phase"].get(s, 0)
                    and c[f"{s}_Q"] == row["q_by_phase"].get(s, 0) for s in PHASES),
            f"Inconsistent charged work for case {row['index']}")
    c["requests_over"] = int(c["over_F"] > 0)
    return c, widths


def percentile(hist, num, den):
    n = sum(hist.values())
    target = (n*num + den-1)//den
    observed = 0
    for width, samples in sorted(hist.items()):
        observed += samples
        if observed >= target:
            return width
    raise ValueError("Empty histogram")


def report(data, fixed):
    samples = data / "samples.json"
    tasks, digest = load_tasks(samples), sha(samples)
    with fixed.open() as handle:
        ref = {(r["model"], int(r["slot"]), r["cohort"]): r for r in csv.DictReader(handle)}
    out = []
    for model, slots in FOLDERS.items():
        for offer, name in slots.items():
            folder = data / name
            policy = f"t2t_q{offer}_f0_m{offer}_hard_prefix_zero"
            meta = read(folder/"run.json")
            require(meta["model"] == model and meta["suite"] == "frozen90" and
                    meta["samples_sha256"] == digest and policy in meta["policies"] and
                    read(folder/"qualified.json") == dict(model=model, full_width=32 if model=="llada21" else 256,
                                                          matched=90) and
                    read(folder/f"complete-{policy}.json") == dict(policy=policy, requests=90),
                    "Unqualified run")
            rows = []
            for task in tasks:
                row = read(folder/f"{task['index']:02d}-{policy}.json")
                require(row["status"] == "success" and row["policy"] == policy and
                        all(row[k] == task[k] for k in ("index", "task_id", "seed", "dataset")), "Wrong request")
                rows.append(count(row, model, offer))
            for a,b,cohort in COHORTS:
                c, hist = Counter(), Counter()
                for stats, widths in rows[a:b]:
                    c.update(stats)
                    hist.update(widths)
                rr = ref[model, offer, cohort]
                require(all(c[k] == int(rr[rk]) for k,rk in (("N","N"), ("F","F"),
                                                               ("Q","physical_Q"), ("requests","requests"))),
                        "Mismatch vs published reference")
                out.append(dict(model=model, offer=offer, cohort=cohort, source=name, policy=policy,
                    **{k:c[k] for k in ("requests","requests_over","N","F","Q","over_F","over_Q","under_Q",
                        "selected_fresh_Q","selected_mask_Q","selected_backfill_Q")},
                    **{f"{s}_{k}": c[f"{s}_{k}"] for s in PHASES for k in ("F","Q","over_F","over_Q")},
                    mean_q=c["Q"]/c["F"], p50_q=percentile(hist,1,2), p95_q=percentile(hist,95,100),
                    max_q=max(hist), over_fraction=c["over_F"]/c["F"],
                    protocol_over_fraction=1-c["selected_over_Q"]/c["over_Q"] if c["over_Q"] else None))
    return out


def plot(rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    colors = ("#376995", "#c65d36", "#559567", "#836fac", "#bd9133")
    for ax, (model, slots) in zip(axes, FOLDERS.items(), strict=True):
        chosen = {int(r["offer"]):r for r in rows if r["model"] == model and r["cohort"] == "all-0-89"}
        require(set(chosen) == set(slots), "Missing official fixed single-block slot")
        labels = [f"Q{q}" for q in slots]
        left = [0.] * len(labels)
        for stage, color in zip(PHASES, colors, strict=True):
            values = [int(chosen[q][f"{stage}_over_Q"])/1000 for q in slots]
            ax.barh(labels, values, left=left, color=color, label=stage)
            left = [x+y for x,y in zip(left, values)]
        for i, q in enumerate(slots):
            r = chosen[q]
            ax.text(left[i]+1.3, i, f"{int(r['over_Q']):,}\n{int(r['over_F'])/int(r['F']):.1%} of calls",
                    va="center", fontsize=8.5)
        ax.set_xlim(0, max(left)*1.36)
        ax.set_xlabel("Physical rows above offer (thousands)")
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("LLaDA 2.1 / B32" if model == "llada21" else "DiffusionGemma / B256")
    fig.suptitle("Fixed sparse K0 | Excess rows and share of calls exceeding offer | 90 requests")
    axes[1].legend(loc="lower right", frameon=False, fontsize=8)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path)
    source.add_argument("--from-csv", type=Path, help="Replot without private request traces")
    p.add_argument("--fixed", type=Path, default=HERE/"fixed-k0.csv")
    p.add_argument("--out", type=Path)
    p.add_argument("--plot", type=Path)
    args = p.parse_args()
    if args.data:
        if not args.out:
            p.error("--data requires --out")
        rows = report(args.data, args.fixed)
        with args.out.open("w") as out:
            w = csv.DictWriter(out, fieldnames=rows[0], lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    else:
        if args.out or not args.plot:
            p.error("--from-csv takes --plot, not --out")
        with args.from_csv.open() as source:
            rows = list(csv.DictReader(source))
    if args.plot:
        plot(rows, args.plot)
