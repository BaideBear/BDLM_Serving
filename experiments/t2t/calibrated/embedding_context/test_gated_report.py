import csv
import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

from experiments.t2t.calibrated.embedding_context import report as study
from tools.t2t_results import summarize, table


PREFIX = "t2t_q8_f0_m7_hard_context_"
ZERO = "t2t_q8_f0_m8_hard_noedit"
EXP = study.ROOT / "experiments/t2t"
FIXED = EXP / "calibrated/embedding_context/rollout"


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def measured(out, tie_quality=False):
    _, source = study.scored_rows(EXP / "hard", ZERO)
    incorrect = [r["index"] for r in source if r["correct"] == "0" and r["failed"] == "0"]
    requests, summaries = [], []
    for mode in ("slack", "slack_random", "gate", "gate_random"):
        meta = dict(model="llada21", policy=PREFIX+mode, nominal_q=8, fresh=0, mask=7, edit=1)
        arm = []
        for r in source:
            row = {**r, **meta, **{k: int(r[k]) for k in
                   ("N", "F", "Q", "physical_Q", "correct", "failed", "truncated", "max_q")}}
            flips = {"slack": 0 if tie_quality else 1, "slack_random": 2, "gate": 1, "gate_random": 0}
            if r["index"] in incorrect[:flips[mode]]:
                row["correct"] = 1
            if mode == "gate" and r["index"] == "0":
                # A real loss in mean request N/F hidden by three-place display rounding.
                for k in ("N", "F", "Q", "physical_Q"):
                    row[k] *= 1000
                row["F"] += 1
            if mode == "slack":
                row["physical_Q"] += row["F"]  # distinguish physical from logical efficiency
            arm.append(row)
        requests.extend(reversed(arm))
        summaries.extend({**meta, "dataset": d, **summarize([r for r in arm if d in ("all", r["dataset"])])}
                         for d in ("all", "gsm8k", "math500", "humaneval"))
    table(out / "llada21-online-requests.csv", requests)
    table(out / "llada21-online-summary.csv", summaries)


def test_gated_cli_and_paired_bootstrap(tmp_path, capsys):
    measured(tmp_path)
    study.main(["--gated", "--out", str(tmp_path)])
    verdicts = capsys.readouterr().out
    assert "slack feasibility vs zero reserve: PASS" in verdicts
    assert "gate feasibility vs zero reserve: FAIL" in verdicts
    work = read(tmp_path / "gated-arms.csv")
    arms = {r["policy"]: r for r in work}
    assert len(work) == len(arms) == 6
    assert all(r["requests"] == "30" for r in work)
    arm = arms[PREFIX + "slack"]
    assert int(arm["physical_Q"]) > int(arm["Q"])
    assert float(arm["physical_efficiency"]) == int(arm["N"]) / int(arm["physical_Q"])
    assert float(arm["physical_q_per_f"]) == int(arm["physical_Q"]) / int(arm["F"])
    pairs = read(tmp_path / "gated-matched.csv")
    by = {(p["policy"], p["reference"]): p for p in pairs}
    expected = {(PREFIX+s, r) for s in ("slack", "gate") for r in (PREFIX+s+"_random", ZERO, PREFIX+"fixed")}
    assert len(pairs) == 7 and set(by) == expected | {(PREFIX+"gate", PREFIX+"slack")}
    assert by[PREFIX+"slack", ZERO]["primary_feasible"] == "True"
    gate = by[PREFIX+"gate", ZERO]
    assert gate["primary_feasible"] == "False" and float(gate["delta_TPF"]) < 0
    assert round(float(arms[PREFIX+"gate"]["macro_tpf"]), 3) == round(float(arms[ZERO]["macro_tpf"]), 3)
    placement = by[PREFIX+"slack", PREFIX+"slack_random"]
    assert placement["purpose"] == "same_gate_placement" and int(placement["delta_correct"]) == -1
    assert all(not r["primary_feasible"] for r in pairs if r["reference"] != ZERO)

    manifest = study.ROOT / "benchmarks/frozen30/manifest.json"
    keys = sorted(study.request_key(r) for r in json.loads(manifest.read_text())["samples"])
    draws = study.stratified_draws(keys)
    matrices = {}
    for policy in arms:
        folder = EXP / "hard" if policy == ZERO else FIXED if policy == PREFIX+"fixed" else tmp_path
        _, matrices[policy] = study.bundle(folder, policy, keys)
    for (policy, ref), r in by.items():
        for k, v in study.paired_uncertainty(matrices[policy], matrices[ref], draws).items():
            assert float(r[k]) == pytest.approx(v)
        assert float(r["delta_eff_pp"]) == pytest.approx(100 * (
            float(arms[policy]["physical_efficiency"]) - float(arms[ref]["physical_efficiency"])))

    svg = tmp_path / "gated.svg"
    labels = [node.text for node in ElementTree.parse(svg).iter("{http://www.w3.org/2000/svg}text")]
    baselines = study.read_baselines(study.ROOT / "experiments/baselines/summary.csv")
    for mode in ("ibs", "obs"):
        p = baselines["llada21", 8, mode]
        assert {f"Optimus {mode.upper()}8 (historical)", f"{p.correct}/30", f"{p.tpf:.3f}",
                f"{p.efficiency:.2%}", f"{p.q_per_f:.2f}"} <= set(labels)
    first = {p.name: p.read_bytes() for p in tmp_path.glob("gated*")}
    study.main(["--out", str(tmp_path), "--gated"])
    assert first == {p.name: p.read_bytes() for p in tmp_path.glob("gated*")}
    assert not (tmp_path / "matched.csv").exists()
    details = read(tmp_path / "gated-request-deltas.csv")
    assert len(details) == 7 * 30
    for (policy, ref), r in by.items():
        group = [d for d in details if (d["policy"], d["reference"]) == (policy, ref)]
        assert len({d["index"] for d in group}) == 30
        for d in group:
            length, forwards = float(d["length_component"]), float(d["forwards_component"])
            assert length + forwards == pytest.approx(float(d["delta_TPF"]), abs=1e-10)
            if d["delta_N"] == "0":
                assert length == 0
            if d["delta_F"] == "0":
                assert forwards == 0
        for part in ("length_component", "forwards_component"):
            assert sum(float(d[part]) for d in group) / 30 == pytest.approx(float(r[part]))


def test_equal_quality_does_not_pass(tmp_path):
    measured(tmp_path, tie_quality=True)
    assert not study.gated_report(tmp_path)[PREFIX+"slack"]


@pytest.mark.parametrize("fault", ["absent", "wrong_task", "wrong_mask", "duplicate_all", "wrong_work"])
def test_reject_partial_or_mismatched_inputs(tmp_path, fault):
    measured(tmp_path)
    kind = "summary" if fault in ("duplicate_all", "wrong_work") else "requests"
    path = tmp_path / f"llada21-online-{kind}.csv"
    rows = read(path)
    if fault == "absent":
        rows = [r for r in rows if r["policy"] != PREFIX+"gate_random"]
    elif fault == "wrong_task":
        rows[0]["task_id"] = "unmatched"
    elif fault == "wrong_mask":
        rows[0]["mask"] = "8"
    elif fault == "duplicate_all":
        rows.append(rows[0])
    else:
        rows[0]["physical_Q"] = "1"
    table(path, rows)
    with pytest.raises(ValueError):
        study.gated_report(tmp_path)
    assert not list(tmp_path.glob("gated*"))


def test_default_still_uses_fixed_rollout(monkeypatch):
    calls = []
    monkeypatch.setattr(study, "report", calls.append)
    study.main([])
    assert calls == [Path(study.__file__).parent / "rollout"]
