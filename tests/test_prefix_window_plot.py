"""Regression checks for the published, aggregate-only prefix-window figure."""

import csv
import io
from pathlib import Path
import runpy
import xml.etree.ElementTree as ET

import pytest


FOLDER = Path(__file__).resolve().parents[1] / "experiments/t2t/prefix_window"


def read_rows(filename):
    with (FOLDER / filename).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_gemma_q32_numeric_labels_and_layout():
    plot = runpy.run_path(str(FOLDER / "plot.py"))
    groups = plot["load_groups"](FOLDER, model="gemma", q=32)
    expected = {}

    def record(window, policy, requests, correct, tpf, efficiency):
        values = (f"{correct}/{requests}", f"{float(tpf):.3f}", f"{float(efficiency):.2%}")
        for metric, val in enumerate(values):
            expected[f"value-{window}-{policy}-{metric}"] = val

    for k in (1, 4):
        for mode in ("right", "random"):
            for row in read_rows(f"gemma-q32-k{k}-{mode}.csv"):
                if row["scope"] == "all":
                    for side in ("baseline", "arm"):
                        record(row["window"], row[f"{side}_policy"], row["requests"],
                               row[f"{side}_correct"], row[f"{side}_macro_tpf"],
                               row[f"{side}_pooled_token_efficiency"])
    for row in read_rows("native90.csv"):
        if row["model"] == "gemma" and row["dataset"] == "all":
            record(row["requests"], "native256", row["requests"], row["correct"],
                   row["macro_tpf"], row["token_efficiency"])

    plt = plot["plt"]
    with plt.rc_context({"svg.fonttype": "none", "font.family": "DejaVu Sans"}):
        figure = plot["make_figure"]("gemma", 32, groups["gemma", 32])
        try:
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            content = [*figure.texts, *(text for ax in figure.axes for text in ax.texts)]
            bounds = [text.get_window_extent(renderer) for text in content]
            assert all(figure.bbox.contains(*b.min) and figure.bbox.contains(*b.max) for b in bounds)
            assert not any(a.overlaps(b) for i, a in enumerate(bounds) for b in bounds[i + 1:])
            buffer = io.BytesIO()
            figure.savefig(buffer, format="svg")
        finally:
            plt.close(figure)

    tree = ET.fromstring(buffer.getvalue())
    labels = {g.get("id"): "".join(g.itertext()).strip() for g in tree.iter()
              if g.get("id", "").startswith("value-")}
    assert len(expected) == 54
    assert {key: labels[key] for key in expected} == expected


@pytest.mark.parametrize("scenario", ["missing", "failed", "mismatched_policy"])
def test_plot_rejects_incomplete_or_mismatched_comparisons(scenario):
    plot = runpy.run_path(str(FOLDER / "plot.py"))
    rows = [r for r in read_rows("gemma-q32-k1-right.csv") if r["scope"] == "all"]
    if scenario == "missing":
        rows.pop()
    elif scenario == "failed":
        rows[0]["failed"] = "1"
    else:
        rows[0]["baseline_policy"] = rows[0]["arm_policy"]
    with pytest.raises(ValueError):
        plot["read_comparison"](rows, "fixture")
