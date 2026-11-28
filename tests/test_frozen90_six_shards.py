import json

import pytest

from tools.frozen90_six_shards import MANIFEST, OFFERS, split, verify_complete, verify_matrix


def test_real_manifest_rejects_missing_duplicate_different_seed_board_or_version():
    samples = json.loads(MANIFEST.read_text())["samples"]
    cfg = dict(model="gemma", offer=128, version="fixture")
    rows = [dict(r, **cfg, board=dev, status="complete")
            for dev, shard in split(samples).items() for r in shard]
    rows[-1]["status"] = "valid_algorithm_failure"
    assert verify_complete(samples, rows, **cfg) == 90
    for wrong in (rows[:-1], rows+rows[:1], [dict(rows[0], version="old")]+rows[1:],
                  [dict(rows[0], status="infrastructure_error")]+rows[1:],
                  [dict(rows[0], board="A800:7")]+rows[1:],
                  [dict(rows[0], seed=-1)]+rows[1:]):
        with pytest.raises(ValueError):
            verify_complete(samples, wrong, **cfg)


def test_version_is_not_complete_until_all_ten_separate_offers_seal():
    samples = json.loads(MANIFEST.read_text())["samples"]
    rows = [dict(r, board=board, model=model, offer=q, version="two-model-registration", status="complete")
            for model, offers in OFFERS.items() for q in offers
            for board, shard in split(samples).items() for r in shard]
    verify = lambda rows: verify_matrix(samples, rows, version="two-model-registration")
    assert list(verify(rows).values()) == [90] * 10
    with pytest.raises(ValueError, match="gemma/Q512: Missing"):
        verify(rows[:-90])
    with pytest.raises(ValueError, match="duplicate"):
        verify(rows[:-1] + rows[-2:-1])
    with pytest.raises(ValueError, match="Unregistered"):
        verify(rows + [dict(rows[0], offer=12)])
