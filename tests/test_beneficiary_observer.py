from bdlm_serving.methods.t2t.seat_audit import SeatAudit


def test_private_post_forward_observer_lifetime_and_offsets():
    audit = SeatAudit("hard_once_rank")
    event = {}
    audit.observe_masks(event, 2, -3, [41] * 8, [True] * 8)
    assert not event
    audit.current = {"base": [4, 7, 6], "masks": [7, 4, 5], "victim": 7}
    audit.observe_masks(event, 2, -3, list(range(8)), [True] * 7 + [False])
    assert event["beneficiary"] == {"rows": [4, 7], "ids": [4, None], "output_offsets": [1, 4]}
    audit.current = None
    event = {}
    audit.observe_masks(event, 2, -3, list(range(8)), [True] * 8)
    assert event["beneficiary"]["ids"] == [4, 7]
    event = {}
    audit.observe_masks(event, 3, 29, list(range(8)), [True] * 8)
    assert not event
