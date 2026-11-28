from experiments.t2t.serving_slots.gemma_full_admission import admission_certificate


def certificate(**changes):
    base = admission_certificate(
        set(), set(), set(), ordinal=1, nominal_q=64,
        parent_rows=list(range(64)))
    return {key: changes.get(key, value) for key, value in base.items()}


def test_first_new_mask_block_over_parent_is_certified():
    item = certificate()
    assert item["physical_q"] == 256
    assert item["over_parent_rows"] == list(range(64, 256))
    assert item["explainable"]


def test_dirty_or_accepted_rows_are_not_eligible():
    assert not certificate(dirty={0})["explainable"]
    assert not certificate(accepted={0})["explainable"]


def test_first_canvas_or_parent_full_selection_is_not_admission():
    assert not certificate(ordinal=0)["explainable"]
    assert not certificate(parent_rows=list(range(256)))["explainable"]
