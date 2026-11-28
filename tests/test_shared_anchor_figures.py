import csv
from pathlib import Path

import pytest

from tools.shared_anchor_figures import draw, make_rows


BASELINES = Path(__file__).resolve().parents[1]/"experiments/t2t/serving_slots/controls/fixed-request-comparison/metrics.csv"


def test_refuses_partial_figures():
    with pytest.raises(ValueError, match="sealed"):
        make_rows(dict(seen=899, deferred_incomplete_lines=[], points=[]), BASELINES)


def test_separate_candidate_and_actual_budget_baselines(tmp_path):
    # Synthetic candidate for a renderer test, not a GPU or research result.
    points = {}
    with BASELINES.open() as f:
        for r in csv.DictReader(f):
            if r["method"] != "Live profile":
                continue
            key = (r["model"], int(r["budget"]))
            p = points.setdefault(key, dict(model=key[0], offer=key[1], cohorts={}))
            group = {"frozen90":"all90"}.get(r["cohort"],r["cohort"])
            p["cohorts"][group] = dict(requests=int(r["n"]), correct=0, macro_TPF=0., pooled_TE=0.,
                                       N=0,F=900,Q=3000)
    rows = make_rows(dict(version="synthetic", seen=900, deferred_incomplete_lines=[], points=list(points.values())),
                     BASELINES)
    assert len([r for r in rows if r["method"]=="Shared planner"]) == 30
    assert not [r for r in rows if r["method"]=="Native" and r["model"]=="gemma" and r["budget"]!=256]
    assert all(r["token_efficiency"] is None for r in rows if r["method"]=="Flow W2")
    output = tmp_path/"synthetic.png"
    draw(rows, shared_output=output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
