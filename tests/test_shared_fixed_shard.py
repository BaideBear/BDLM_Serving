from collections import Counter
import json
from types import SimpleNamespace

import pytest
from experiments.t2t.serving_slots import shared_fixed_shard as shard


def test_screen_ids_are_balanced_and_not_a_formal_cell():
    samples = json.loads(shard.MANIFEST.read_bytes())["samples"]
    select = lambda stage: [r for board in shard.BOARDS for r in shard.shard_cases(samples, board, stage)]
    screen = select("screen18")
    assert len({r["index"] for r in screen}) == 18
    assert list(Counter(r["dataset"] for r in screen).values()) == [6, 6, 6]
    assert len({r["index"] for r in select("formal90")}) == 90


def test_registration_digest_covers_both_model_selections_and_stage():
    p = shard.param_values(["1/4", "1/2"])
    g = shard.param_values(["1/8", "1/4"])
    r = dict(stage="screen18", params=dict(gemma=g, llada21=p))
    assert shard.identity(r) != shard.identity(dict(r, params=dict(llada21=g, gemma=p)))
    assert shard.identity(r) != shard.identity(dict(r, stage="formal90"))
    assert shard.param_values(["2/8", "0.5"]) == p
    with pytest.raises(ValueError):
        shard.param_values(["2", "1/4"])


def test_single_load_for_all_fresh_requests(monkeypatch, tmp_path, capsys):
    # Fake adapter; no inference. Check registration routing and the emitted compact ledger.
    calls = []
    params = shard.param_values(["1/4", "1/2"])
    reg = dict(format=shard.FORMAT, cap=2048, blobs={}, stage="screen18",
               params=dict(llada21=params, gemma=shard.param_values(["0", "1"])))
    file = tmp_path/"registration.json"
    file.write_text(json.dumps(reg))
    args = SimpleNamespace(registration=file, version=shard.identity(reg), samples=None,
                           model="llada21", board=shard.BOARDS[0], checkpoint="fake",
                           offers=(4, 64), preflight=False)
    samples = json.loads(shard.MANIFEST.read_bytes())["samples"]
    def evaluate(handle, task, offer, actual):
        assert actual.prefix_share == shard.Fraction(1,4)
        calls.append(offer)
        return dict(status="complete", correct=True, N=1, Q=2, F=1, decoder_F=1, encoder_F=0,
                    output_ids=[1])
    fake = SimpleNamespace(load=lambda path: calls.append("load"), evaluate=evaluate)
    monkeypatch.setattr(shard, "check_board", lambda board: None)
    monkeypatch.setattr(shard, "load_tasks", lambda path: samples)
    monkeypatch.setattr(shard, "import_module", lambda module, package: fake)
    shard.run(args)
    receipts = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert calls == ["load", 4, 64, 64, 4, 4, 64]
    assert len(receipts) == 6 and all("output_ids" not in r for r in receipts)
    assert all(r["version"] == args.version and r["stage"] == "screen18" for r in receipts)
