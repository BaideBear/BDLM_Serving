import pytest

from tools.physical_split import request_split


def test_partition_selected_rows_and_count_full_checks_separately():
    e = dict(stage="selected", positions=[0, 1, 2, 3, 4], q=5, mandatory_fresh=2,
             selected_fresh=2, selected_mask=1, reserved_extra_edit_positions=[4], overflow=0)
    row = dict(model="gemma", events=[e], output_tokens=7, generation_forwards=2,
               generation_query_rows=261, generation_physical_padded_rows=261,
               tpf=3.5, token_efficiency=7/261, q_by_phase=dict(selected=5, verify=256))
    c = request_split(row)
    assert [c[k] for k in ("mandatory_Q", "mask_Q", "reserved_old_Q", "backfill_old_Q",
                            "selected_Q", "other_Q")] == [2, 1, 1, 1, 5, 256]
    e["selected_fresh"] = 1
    with pytest.raises(ValueError):
        request_split(row)
