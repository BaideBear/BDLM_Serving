from copy import deepcopy

import pytest

from bdlm_serving.methods.t2t.physical_prefix import plan_prefix
from experiments.t2t.prefix_window.action_lock import AUDIT_KEYS, audit_one, drop


def pair(strict=False, revoke=False):
    plan = plan_prefix([True]*254 + [False]*2, range(256), {253}, 32, 8,
                       "hard_prefix_slack_right", seed=1, block_start=0, clock=3)
    added = list(plan.added_rows)
    s, r = (added[:1] if strict else []), (added[1:2] if revoke else [])
    trace = dict(block=0, clock=3, stage="selected", q=len(plan.positions),
                 positions=list(plan.positions), prefix_plan=plan.audit(),
                 prefix_actions=dict(locked=False, proposed_strict=s, proposed_revoked=r),
                 prefix_canvas_sha256="a"*64, prefix_draft_sha256="b"*64,
                 edited=len(s), hard_strict_edits=len(s), revoked=len(r),
                 resolved=254-len(r), reserved_strict_edit_positions=s,
                 reserved_extra_edit_positions=added)
    p = "t2t_q32_f0_m24_hard_prefix_slack_right"
    normal = dict(policy=p, events=[trace], output_ids=[1])
    old = dict(normal, events=[drop(trace, AUDIT_KEYS)])
    lock = deepcopy(normal)
    lock["policy"] += "_action_lock"
    e = lock["events"][0]
    e["prefix_actions"]["locked"] = True
    e.update(edited=0, hard_strict_edits=0, revoked=0, resolved=254,
             reserved_strict_edit_positions=[])
    if s or r:
        e["prefix_canvas_sha256"] = "c"*64
    if r:
        e["prefix_draft_sha256"] = "d"*64
    return old, normal, lock


@pytest.mark.parametrize("strict,revoke", [(False, False), (True, False), (False, True), (True, True)])
def test_local_first_suppression(strict, revoke):
    counts = audit_one(*pair(strict, revoke), 0)
    assert counts["requests_directly_suppressed"] == int(strict or revoke)
    assert (counts["normal_strict"], counts["lock_revoked"]) == (int(strict), int(revoke))


def test_no_suppression_must_keep_trajectory_work_and_output():
    old, normal, lock = pair()
    lock["output_ids"] = [2]
    with pytest.raises(ValueError, match="Unexposed"):
        audit_one(old, normal, lock, 0)
    lock["events"][0]["prefix_draft_sha256"] = "f"*64
    with pytest.raises(ValueError, match="Pre-suppression"):
        audit_one(old, normal, lock, 0)


def test_repeat_and_proposals_are_not_assumed_equal():
    old, normal, lock = pair(True, True)
    lock["events"][0]["prefix_actions"]["proposed_revoked"] = []
    with pytest.raises(ValueError, match="Proposal mismatch"):
        audit_one(old, normal, lock, 0)
    old["output_ids"] = [3]
    with pytest.raises(ValueError, match="Historical normal"):
        audit_one(old, normal, lock, 0)
