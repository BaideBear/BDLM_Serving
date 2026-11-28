from experiments.t2t.serving_slots.gemma_full_admission import admission_certificate


def certificate(**changes):
    base = admission_certificate(
        rows=tuple(range(64)), initialized=set(), dirty=set(), accepted=set(),
        ordinal=1, nominal_q=64, parent_rows=list(range(64)))
    return {key: changes.get(key, value) for key, value in base.items()}


def test_full_new_mask_block_over_parent_is_certified():
    item = certificate()
    assert item['physical_q'] == 256 and item['over_parent_rows'] == list(range(64))
    assert item['eligible'] and item['explainable']


def test_dirty_or_accepted_rows_are_not_eligible():
    assert not certificate(dirty={0})['eligible']
    assert not certificate(accepted={0})['eligible']


def test_parent_full_selection_is_not_an_admission():
    assert not certificate(parent_rows=list(range(256)))['over_parent']
    assert not certificate(parent_rows=list(range(256)))['explainable']
