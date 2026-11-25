"""Render one-seat interventions with controls from the published frozen30 tables."""
import argparse
import csv
from pathlib import Path
from xml.sax.saxutils import escape

from tools.render_hard_study import ROOT, all_rows, point, read_baselines


def single(path, policy):
    row, = [x for x in all_rows(path) if x["policy"] == policy]
    return point(row, "token_efficiency")


def render(out, root=ROOT):
    parent = root / "experiments/t2t"
    baselines = read_baselines(root / "experiments/baselines/summary.csv")
    items = []
    W, H = 1060, 640
    def text(x, y, s, size=15, weight="normal", color="#202b2a", align="start"):
        items.append(f'<text x="{x}" y="{y}" text-anchor="{align}" font-size="{size}" '
                     f'font-weight="{weight}" fill="{color}">{escape(str(s))}</text>')
    def line(y):
        items.append(f'<path d="M26,{y}H1030" stroke="#c4cdce"/>')
    text(26, 35, "What changes when exactly one Q row changes?", 25, "bold")
    text(26, 62, "Frozen 30 per model · one matched-state substitution per eligible request · no substitution on the others")
    for j, (model, q, full_dir, full_policy, name) in enumerate((
        ("llada21", 4, "placement/geometry-online", "t2t_q4_f0_m3_hard_slack_geometry", "LLaDA 2.1"),
        ("gemma", 32, "adaptive", "t2t_q32_f0_m28_hard_trigger", "DiffusionGemma"),
    )):
        base = 104 + 252*j
        text(26, base, f"{name}   |   nominal Q = {q}", 19, "bold")
        for x, t in zip((520, 665, 824, 991), ("Correct / 30", "TPF", "Token eff.", "Actual Q/F")):
            text(x, base, t, 14, "bold", align="end")
        line(base+10)
        rows = [(f"Optimus {mode.upper()}{q}", baselines[model,q,mode]) for mode in ("ibs", "obs")]
        rows += [(label, single(out/f"{model}-online-summary.csv", f"t2t_q{q}_f0_m{q}_hard_once_{s}"))
            for s, label in (("base","No predicted seat (neutral replay)"), ("random","One swap: uniform old row"),
                             ("rank","One swap: predicted old row"))]
        rows.append(("Repeated swaps: original online policy", single(parent/full_dir/
            f"{model}-online-summary.csv", full_policy)))
        for k, (label, p) in enumerate(rows):
            y = base+38+31*k
            ink = "#006b5f" if k == 4 else "#263036"
            text(26, y, label, 16, "bold" if k == 4 else "normal", ink)
            for x, val in zip((520,665,824,991), (f"{p.correct}", f"{p.tpf:.3f}",
                    f"{100*p.efficiency:.2f}%", f"{p.q_per_f:.2f}")):
                text(x, y, val, 16, "bold" if k == 4 else "normal", ink, "end")
        line(base+206)
    text(26, 598, "TPF = mean(request tokens / forwards); efficiency = pooled tokens / physical Q. All generation closure work is charged.", 13)
    text(26, 620, "The final row repeatedly applies the selector; it is not a single intervention. TPF is not wall-clock throughput.", 13)
    path = out / "one-seat.svg"
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Single-seat paired results" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/>'
        '<g font-family="Arial, sans-serif">' + "".join(items) + '</g></svg>', encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "experiments/t2t/placement/one-seat")
    render(p.parse_args().out)
