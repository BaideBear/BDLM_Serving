"""Render one-seat interventions with controls from the published frozen30 tables."""
import argparse
from pathlib import Path
from xml.sax.saxutils import escape

from tools.render_hard_study import ROOT, all_rows, point, read_baselines


def single(path, policy):
    row, = [x for x in all_rows(path) if x["policy"] == policy]
    return point(row, "token_efficiency")


class SeatFigure:
    def __init__(self):
        self.items = []

    def text(self, x, y, s, size=15, weight="normal", color="#202b2a", align="start"):
        self.items.append(f'<text x="{x}" y="{y}" text-anchor="{align}" font-size="{size}" '
                          f'font-weight="{weight}" fill="{color}">{escape(str(s))}</text>')

    def line(self, y):
        self.items.append(f'<path d="M26,{y}H1030" stroke="#c4cdce"/>')

    def save(self, path, height, label):
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{label}" '
            f'width="1060" height="{height}" viewBox="0 0 1060 {height}"><rect width="100%" height="100%" fill="white"/>'
            '<g font-family="Arial, sans-serif">' + ''.join(self.items) + '</g></svg>', encoding="utf-8")


def render(out, root=ROOT):
    parent = root / "experiments/t2t"
    baselines = read_baselines(root / "experiments/baselines/summary.csv")
    fig = SeatFigure()
    text, line = fig.text, fig.line
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
    fig.save(out / "one-seat.svg", 640, "Single-seat paired results")


def render_factorial(out, root=ROOT):
    from tools.render_hard_study import render as render_points
    baselines = read_baselines(root / "experiments/baselines/summary.csv")
    model, q = "gemma", 32
    rows = [(f"Optimus {mode.upper()}{q}", mode, baselines[model, q, mode]) for mode in ("ibs", "obs")]
    def get(folder, arm):
        return single(folder/"gemma-online-summary.csv", "t2t_q32_f0_m32_hard_once_" + arm)
    rows.append(("No reserved seat", "hard", get(out.parent, "base")))
    rows.append(("Uniform single swap", "placebo", get(out.parent, "random")))
    for code, desc in (("kk", "Keep KV, use score"), ("dk", "Restore KV, use score"),
                       ("kd", "Keep KV, freeze score"), ("dd", "Restore KV, freeze score")):
        rows.append((desc, "geometry" if code[0] == "k" else "adaptive", get(out, code)))
    render_points(out/"factorial.svg", model, 256, [(q, rows)], adaptive=True,
                  study_title="One predicted seat: KV persistence vs sampling")


ARMS = ("base", "rank", "random")
KINDS = ("common_mask", "mask_victim")
STRATA = {"all": "All requests", "slack": "Old refill (slack)",
          "mask": "MASK displaced", "none": "No anchor"}


def beneficiary_cells(out, model):
    cells = {}
    for r in all_rows(out / f"{model}-beneficiary-summary.csv"):
        if r["measure"] != "arm":  # Pair rows contain signed deltas, not waiting times.
            continue
        key = r["stratum"], r["row_kind"], r["comparison"]
        requests, n, total = (int(r[k]) for k in ("requests", "eligible", "stable_wait_sum"))
        if (r["model"] != model or key in cells or not 0 <= requests <= 30 or
                not 0 <= total <= 4*n or (not requests and n)):
            raise ValueError(f"{model} {key}: invalid or duplicate beneficiary cell")
        cells[key] = requests, n, total
    if set(cells) != {(s, k, a) for s in STRATA for k in KINDS for a in ARMS}:
        raise ValueError(f"{model}: expected all beneficiary arm cells, including empty strata")
    for s in STRATA:
        if len({cells[s, k, a][0] for k in KINDS for a in ARMS}) != 1:
            raise ValueError(f"{model} {s}: unmatched request counts")
        for k in KINDS:
            if len({cells[s, k, a][1] for a in ARMS}) != 1:
                raise ValueError(f"{model} {s} {k}: unmatched eligible rows")
    if cells["all", KINDS[0], ARMS[0]][0] != 30:
        raise ValueError(f"{model}: expected the full 30-request cohort")
    return cells


def render_beneficiary(out):
    fig = SeatFigure()
    text = fig.text
    text(26, 35, "How long until MASK rows match the base endpoint?", 25, "bold")
    text(26, 62, "Mean stable wait over H = 4 observations (lower is earlier). n = eligible rows shared by all three arms.")
    for j, (model, title) in enumerate((("llada21", "LLaDA 2.1   |   Q4"), ("gemma", "DiffusionGemma   |   Q32"))):
        cells = beneficiary_cells(out, model)
        y = 108 + 222*j
        text(26, y, title, 19, "bold")
        for x, title in ((460, "Common MASKs"), (862, "Displaced MASK victim")):
            text(x, y, title, 17, "bold", align="middle")
        text(26, y+26, "Anchor stratum", 14)
        text(239, y+26, "Requests", 14, align="end")
        cols = ((320, 410, 505, 595), (725, 815, 910, 1000))
        for group in cols:
            for x, label in zip(group, ("n", "Base", "Targeted", "Random")):
                text(x, y+26, label, 14, "bold", align="end")
        fig.line(y+36)
        for i, (s, label) in enumerate(STRATA.items()):
            ry = y+62+31*i
            fig.items.append(f'<g id="{model}-{s}">')
            text(26, ry, label, 16, "bold" if s == "all" else "normal")
            text(239, ry, cells[s, KINDS[0], ARMS[0]][0], 16, align="end")
            for kind, group in zip(KINDS, cols):
                n = cells[s, kind, "base"][1]
                text(group[0], ry, n, 15, color="#536361", align="end")
                for x, a in zip(group[1:], ARMS):
                    value = f"{cells[s, kind, a][2] / n:.2f}" if n else "N/A"
                    text(x, ry, value, 16, "bold" if a == "rank" and n else "normal",
                         "#006b5f" if a == "rank" and n else "#263036", "end")
            fig.items.append('</g>')
        fig.line(y+169)
    text(26, 531, "Anchor = 0. Wait = 4 if no match holds through observation 3. After block exit, the last state is carried without extra forwards.", 13)
    text(26, 553, "Reference = token delivered by the base arm at the same output offset, not ground truth. N/A = no eligible rows.", 13)
    fig.save(out / "beneficiary.svg", 578, "H4 stable waiting time and eligible MASK rows by arm")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--factorial", action="store_true")
    modes.add_argument("--beneficiary", action="store_true",
                       help="read both public beneficiary summaries in --out and write beneficiary.svg there")
    args = p.parse_args(argv)
    folder = ROOT / "experiments/t2t/placement/one-seat"
    if args.beneficiary:
        render_beneficiary(args.out or folder/"beneficiary")
    elif args.factorial:
        render_factorial(args.out or folder/"kv-factorial")
    else:
        render(args.out or folder)


if __name__ == "__main__":
    main()
