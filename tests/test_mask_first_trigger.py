from copy import deepcopy

import pytest

from experiments.t2t.serving_slots.mask_pair_report import check_llada_prefix, efficiency_interval, first_eligible


def events():
    def step(pos, end):
        q = len(pos)
        return dict(phase="denoise", stage="selected", absolute_positions=pos,
                    visible_kv=end, physical_q=q, actual_q=q, layer_physical_q=[q], edited=0)
    return [dict(phase="prefill"), step([60, 64], 96), step([61, 75], 96)]


def test_trigger_uses_absolute_query_rows_and_excludes_prompt():
    seq = events()
    assert first_eligible(seq, 61) == 2
    assert first_eligible(seq, 62) is None
    on = dict(events=seq)
    off = dict(events=deepcopy(seq), mask_pair=dict(opened_forward_count=1,
                                                    first_trigger=dict(event_index=2)))
    off["events"][2]["edited"] = 1
    assert check_llada_prefix(on, off, 61)
    off["events"][1]["edited"] = 1
    with pytest.raises(ValueError, match="before first"):
        check_llada_prefix(on, off, 61)


def test_zero_edges_and_first_forward_work_require_exact_parity():
    seq = events()
    on = dict(events=seq)
    off = dict(events=deepcopy(seq), mask_pair=dict(opened_forward_count=0))
    assert not check_llada_prefix(on, off, 62)
    with pytest.raises(ValueError, match="disagrees"):
        check_llada_prefix(on, off, 61)
    off["mask_pair"].update(opened_forward_count=1, first_trigger=dict(event_index=2))
    off["events"][2]["layer_physical_q"] = [9]
    with pytest.raises(ValueError, match="physical inputs"):
        check_llada_prefix(on, off, 61)


def test_bootstrap_reuses_paired_ids_dataset_sizes_and_physical_rows():
    cases=[dict(index=0,dataset="a"),dict(index=1,dataset="b")]
    old={0:dict(N=5,physical_Q=10,Q=9),1:dict(N=20,physical_Q=200,Q=80)}
    new={0:dict(N=8,physical_Q=8,Q=6),1:dict(N=12,physical_Q=24,Q=15)}
    delta=20/32-25/210
    assert efficiency_interval(cases,old,new,40)==(delta,delta)
    cases[1]["dataset"]="a"
    assert efficiency_interval(cases,old,old,200)==(0.,0.)
