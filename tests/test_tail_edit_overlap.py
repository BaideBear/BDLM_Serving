import unittest

from experiments.t2t.serving_slots.gemma_top_roles import tail_edit_overlap


def event(head, *, edit=False, write=True, discard=False):
    parts = [
        dict(ordinal=head, edited=[7] if edit else []),
        dict(ordinal=head+1, rows=range(256), sampled=[0], admitted=[0] if write else [],
             edited=[], input_changed=[], revoked=[]),
    ]
    return dict(ordinal=head, parts=parts, discarded=discard)


class TailExposure(unittest.TestCase):
    def test_first_projection_and_epoch_reset_are_not_warm(self):
        result = tail_edit_overlap([
            event(0, edit=True), event(0, edit=True), event(0, edit=True, write=False),
            event(1, edit=True, discard=True), event(1, edit=True),
        ])
        self.assertEqual(result["head_edit_plus_tail_sample_calls"], 5)
        self.assertEqual(result["head_edit_plus_warm_tail_sample_calls"], 2)
        self.assertEqual(result["head_edit_plus_warm_tail_write_calls"], 1)
