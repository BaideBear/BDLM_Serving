import csv
from xml.etree import ElementTree as ET

import pytest

from tools import render_seat_study as study


def summary(model):
    # Both model panels have mask anchors; only Gemma has old-refill anchors.
    slack = 0 if model == "llada21" else 4
    requests = {"all": 30, "mask": 5, "slack": slack, "none": 25 - slack}
    rows = []
    for kind in ("common_mask", "mask_victim"):
        n_mask = 20 if kind == "common_mask" else 5
        n_slack = slack if kind == "common_mask" else 0
        for s, n in (("all", n_mask+n_slack), ("slack", n_slack), ("mask", n_mask), ("none", 0)):
            for arm, total in (("base", 4*n), ("rank", 0), ("random", 2*n + int(n > 0 and s != "slack"))):
                rows.append(dict(model=model, dataset="all", stratum=s, row_kind=kind, measure="arm",
                                 comparison=arm, requests=requests[s], eligible=n, stable_wait_sum=total))
    # Neither a dataset subtotal nor a paired delta belongs in an arm estimate.
    rows += [dict(rows[0], dataset="gsm8k", requests=10, eligible=1, stable_wait_sum=0),
             dict(rows[0], measure="pair", comparison="rank_vs_base", stable_wait_sum=-80)]
    return rows


def write_summary(out, model, rows):
    with (out / f"{model}-beneficiary-summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_beneficiary_cli(tmp_path):
    for model in ("llada21", "gemma"):
        write_summary(tmp_path, model, summary(model))
    study.main(["--beneficiary", "--out", str(tmp_path)])
    svg = tmp_path / "beneficiary.svg"
    first = svg.read_bytes()
    root = ET.fromstring(first)
    ns = {"s": "http://www.w3.org/2000/svg"}

    def cells(row):
        return [t.text for t in root.findall(f".//s:g[@id='{row}']/s:text", ns)]

    assert cells("llada21-all") == ["All requests", "30", "20", "4.00", "0.00", "2.05",
                                   "5", "4.00", "0.00", "2.20"]
    assert cells("gemma-all")[2:6] == ["24", "4.00", "0.00", "2.04"]
    empty = ["0", "N/A", "N/A", "N/A"]
    assert cells("llada21-slack")[1:] == ["0", *empty, *empty]
    assert cells("gemma-slack")[1:] == ["4", "4", "4.00", "0.00", "2.00", *empty]
    assert cells("gemma-none")[1:] == ["21", *empty, *empty]
    assert all(label in first.decode() for label in ("Q4", "Q32", "Common MASKs", "Displaced MASK", "H = 4"))
    for model in ("llada21", "gemma"):
        write_summary(tmp_path, model, summary(model)[::-1])
    study.main(["--beneficiary", "--out", str(tmp_path)])
    assert svg.read_bytes() == first
    assert list(tmp_path.glob("*.svg")) == [svg]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "bad_wait", "unmatched", "incomplete"])
def test_inconsistent_input_is_not_plotted(tmp_path, fault):
    write_summary(tmp_path, "llada21", summary("llada21"))
    rows = summary("gemma")
    if fault == "missing":
        rows.pop(0)
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "bad_wait":
        rows[0]["stable_wait_sum"] += 1  # exceeds H times the eligible count
    elif fault == "unmatched":
        rows[0]["eligible"] += 1
    else:
        for r in rows:
            if r["stratum"] == "all":
                r["requests"] = 29
    write_summary(tmp_path, "gemma", rows)
    with pytest.raises(ValueError):
        study.render_beneficiary(tmp_path)
    assert not (tmp_path / "beneficiary.svg").exists()
