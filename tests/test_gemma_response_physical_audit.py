from experiments.t2t.serving_slots.strict_handle_qualify import physical_projection_audit


def event(rows=64, certificate=None, *, tick=3, ordinal=1, layers=None):
    return dict(tick=tick, ordinal=ordinal, positions=list(range(rows)),
                q=rows, decoder_Q=rows, history_Q=rows,
                response_full_mask_certificate=certificate or [],
                parts=[dict(ordinal=ordinal, rows=list(range(rows)), sampled=list(range(rows)))]), [[rows] * 3 for _ in range(90)]


def test_exact_projection_is_fully_measured():
    e, proj = event()
    checks = physical_projection_audit(e, list(range(64)), 64, proj, 1, 3)
    assert checks['tick'] and checks['head_ordinal'] and checks['positions_match_rows']
    assert checks['width'] and checks['offer_cap'] and checks['unique_rows']
    assert checks['all_layers_measured']
    assert not checks['response_overage_explainable']


def test_only_identity_and_complete_new_block_explains_overage():
    cert = [dict(ordinal=2, rows=list(range(256)), sampled=list(range(256)),
                 complete_new_block=True, sampled_rows_match=True)]
    e, proj = event(rows=320, certificate=cert)
    checks = physical_projection_audit(e, list(range(320)), 64, proj, 1, 3)
    assert not checks['offer_cap']
    assert checks['unique_rows'] and checks['all_layers_measured']
    assert checks['response_full_mask_identity_and_complete_rows']
    assert checks['response_overage_explainable']


def test_partial_or_unsampled_new_block_is_rejected():
    for rows, complete, match in ((128, True, True), (256, False, True), (256, True, False)):
        cert=[dict(ordinal=2, rows=list(range(rows)), sampled=list(range(rows)) if match else [],
                   complete_new_block=complete, sampled_rows_match=match)]
        e,proj=event(rows=64+rows,certificate=cert)
        checks=physical_projection_audit(e,list(range(64+rows)),64,proj,1,3)
        assert not checks['response_overage_explainable']
