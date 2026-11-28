import json
from pathlib import Path
import shutil

import pytest

from tools.plot_paid_budget_points import point


Q8 = (Path(__file__).resolve().parents[1] / "experiments/t2t/serving_slots/controls"
      / "paid-agreement/llada-q8")


def test_sealed_fourth_point_uses_per_request_mean_and_pooled_q():
    doc = point(Q8 / "all90-summary.json")
    a = doc["cohorts"]["frozen90"]["apply"]
    assert (doc["model"], doc["fixed_q"], a["correct"]) == ("llada", 8, 71)
    assert a["tpf"] == pytest.approx(2.5553602115)
    assert a["te"] == pytest.approx(44609/148924)


@pytest.mark.parametrize("bad", ["overlap", "ratio", "device", "source"])
def test_cross_run_mixups_rejected(tmp_path, bad):
    for p in Q8.glob("*summary.json"):
        shutil.copyfile(p, tmp_path/p.name)
    p = tmp_path/("all90-summary.json" if bad == "ratio" else "remaining-summary.json")
    d = json.loads(p.read_bytes())
    if bad == "ratio":
        d["all90"]["totals"]["apply"]["TPF"] = 44609/20278
    elif bad == "device":
        d["registration"]["gpu"] = "another gpu"
    elif bad == "source":
        d["raw_sha256"] = "unrelated run"
    else:
        d["cases"][0]["index"] = 7
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        point(tmp_path/"all90-summary.json")
