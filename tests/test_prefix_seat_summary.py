from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json

_path = Path(__file__).resolve().parents[1] / "experiments/t2t/serving_slots/controls/prefix-seat-q64-a0/summarize.py"
_spec = spec_from_file_location("prefix_seat_summary", _path)
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)
summarize = _module.summarize


SIX = (7, 13, 25, 33, 55, 71)


def metric(n=10, correct=True):
    return dict(N=n, correct=correct, scoring="pass", rng_after="rng",
                output_sha256="sha",
                bill=dict(F=2, Q=64, encoder_Q=0, peak_decoder_Q=64,
                          admissions=n, paid_rows=n))


def arm(n=10, correct=True, exchanges=(), tick=None):
    seat = {"exchanges": list(exchanges), "blocks_used": [0] if exchanges else []}
    first_fork = None
    if exchanges:
        first = exchanges[0]
        first_fork = dict(kind="trace", tick=tick if tick is not None else first["tick"],
                          reference=first["before"], treatment=first["after"])
    return metric(n, correct), seat, first_fork


def case(index, *, error=None, observer_metric=None, seat_metric=None,
         exchanges=(), tick=None):
    stock, _, _ = arm(10)
    observer = observer_metric or stock
    seat_v, seat_record, first_fork = arm(seat_metric if seat_metric is not None else index,
                                          exchanges=exchanges, tick=tick)
    if not exchanges:
        seat_v, seat_record, first_fork = stock, {"exchanges": [], "blocks_used": []}, None
    return dict(index=index, task_id=f"task/{index}", dataset="gsm8k", seed=17000 + index,
                error=error, stock_observer_identical=error is None,
                first_fork=None if error else first_fork,
                arms=dict(stock=stock, observer=observer,
                          seat=dict(seat_v, seat=seat_record)))


def write(root, case_factory=case, gate="passed"):
    cases = [case_factory(i) for i in SIX]
    (root / "summary.json").write_text(json.dumps(dict(
        cases=list(SIX), arms=["stock", "observer", "seat"], gate=gate,
        failed=[] if gate == "passed" else list(SIX))))
    for c in cases:
        (root / f"case-{c['index']:02}.json").write_text(json.dumps(c))
    (root / "raw.log").write_text("RAW\n")


def test_valid_receipt_compacts_without_output_ids(tmp_path):
    root = tmp_path / "valid"
    root.mkdir()
    exchange = [dict(tick=2, before=[1, 2, 3, 4], after=[1, 2, 3, 5])]
    write(root, case_factory=lambda i: case(i, seat_metric=11, exchanges=exchange, tick=2))
    report = summarize(root)
    assert report["classification"] == "valid"
    assert report["gate_launch_ok"] is True
    assert report["valid_case_count"] == 6
    assert all(c["seat_delta"] == dict(N=1, F=0, Q=0) for c in report["valid_cases"])
    assert all("ids" not in key.lower() for key in report["valid_cases"][0])
    assert report["valid_cases"][0]["first_fork"]["tick"] == 2


def test_instrumentation_failure_is_invalid_not_scored(tmp_path):
    root = tmp_path / "invalid"
    root.mkdir()
    write(root, case_factory=lambda i: case(i, error="TypeError('postprocessing')"), gate="failed")
    report = summarize(root)
    assert report["classification"] == "invalid_instrumentation"
    assert report["gate_launch_ok"] is False
    assert report["valid_case_count"] == 0
    assert all("postprocessing" in c["reasons"][0] for c in report["invalid_cases"][:6])


def test_stock_observer_metric_mismatch_fails_gate(tmp_path):
    root = tmp_path / "mismatch"
    root.mkdir()
    write(root, case_factory=lambda i: case(i, observer_metric=metric(99)))
    report = summarize(root)
    assert report["classification"] == "invalid_instrumentation"
    assert report["valid_case_count"] == 0
    assert all("stock/observer reported metrics differ" in c["reasons"]
               for c in report["invalid_cases"][:6])
