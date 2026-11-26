from experiments.t2t.prefix_window.grid import MODELS, arms


def test_gemma_dose_no_reruns_or_missing_cells():
    for q in MODELS["gemma"][1]:
        old, extra = (set(arms("gemma", q, mode)) for mode in ("slack", "dose"))
        assert not old & extra
        assert old | extra == {f"{q}:0:{q-k}:hard_prefix_slack_{selector}"
                               for k in (1, 2, 4) for selector in ("right", "random")}
