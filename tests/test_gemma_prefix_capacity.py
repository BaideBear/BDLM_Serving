"""Synthetic CPU ledgers; no checkpoint, frozen outputs or scorer required."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.t2t.serving_slots.gemma_prefix_capacity import (
    ownership, replay, seats, verified_rows,
)


def span(a, b):
    return list(range(a, b))


def shifted_obs():
    # At the slide: 64 previously flushed tail rows become current-block rows.
    history = [
        (0, 0, span(0, 256), span(0, 128), 0),
        (0, 1, span(0, 192) + span(256, 320), span(128, 192) + span(256, 320), 128),
        (0, 2, span(128, 320), span(192, 256), 192),
        (0, 3, span(192, 256), [], 320),
        (1, 0, span(320, 512), span(320, 512), 64),
        (1, 1, span(320, 512), [], 256),
    ]
    events, trace = [['encoder', 17]], []
    for tick, (block, step, pos, admitted, ready) in enumerate(history):
        if block and step == 0:
            events.append(['encoder', 256])
        events.append(['decoder', len(pos)])
        trace.append(dict(block=block, step=step, q=len(pos), accepted=len(admitted), ready=ready,
                          cross=sum(p >= (block + 1) * 256 for p in pos), visible_end=max(pos) + 1,
                          tick=tick, offer=256, proposed_q=len(pos), positions=pos, admitted=admitted))
    q = sum(q for _, q in events[1:])
    return dict(policy='obs_cap_256', events=events, trace=trace, output_ids=[0]*257, output_tokens=250,
                generation_forwards=7, generation_query_rows=q, generation_physical_padded_rows=q,
                max_q=256, max_decoder_q=256, tpf=250/7, token_efficiency=250/q)


class GemmaCapacity(unittest.TestCase):
    def test_current_block_only_and_distinct_free_seats(self):
        seen = set(range(280))
        c, _ = seats(256, 3, {259, 260}, seen, set(), seen, 6)
        self.assertEqual((c['prefix_pool_rows'], c['free_slots'], c['free_k1_ticks']), (3, 3, 1))
        c, _ = seats(256, 3, {257, 260}, seen, set(), seen, 6)
        self.assertEqual(c['free_slots'], 2)

    def test_guard_dirty_and_unseen(self):
        for dirty, seen in [({0}, {0, 1}), (set(), {1})]:
            with self.assertRaisesRegex(ValueError, 'unclean'):
                seats(0, 1, {1}, {0}, dirty, seen, 4)

    def test_full_one_seat_protects_progress_dirty_edge_and_holes(self):
        # Dirty position 2 cannot be recycled, 3 is first MASK, 4 is an unseen
        # interior MASK, 5 is a seen MASK, 6 the possibly new visible endpoint.
        c, donor = seats(0, 2, {2, 3, 4, 5, 6}, {0, 1, 2}, {2}, {0, 1, 2, 5}, 5)
        self.assertEqual((c['free_slots'], c['clean_swap_ticks'], c['mask_swap_full_ticks'], donor),
                         (0, 0, 1, 5))
        c, donor = seats(0, 2, {2, 3, 4, 6}, {0, 1, 2}, {2}, {0, 1, 2, 3}, 4)
        self.assertEqual((c['k1_any_ticks'], c['mask_shrink_only_full_ticks'], donor), (0, 1, None))

    def test_clean_and_mask_swap_do_not_add_to_two(self):
        c, donor = seats(0, 1, {1, 2, 3, 4}, {0, 1}, set(), set(range(5)), 4)
        self.assertEqual((c['clean_swap_ticks'], c['mask_swap_ticks'], c['k1_any_ticks'], donor),
                         (1, 1, 1, 3))
        self.assertEqual(c['k1_mask_only_full_ticks'], 0)

    def test_obs_shift_pre_call_timing_and_encoder_billing(self):
        stats = replay(shifted_obs(), 256)
        a = stats['all']
        self.assertEqual((a['decoder_F'], a['encoder_gen_F'], a['generation_F']), (6, 1, 7))
        self.assertEqual((a['encoder_gen_Q'], a['encoder_prefill_Q'], a['inherited_prefix_first_ticks']),
                         (256, 17, 1))
        self.assertEqual(a['decoder_Q'] + 256, a['generation_Q'])
        self.assertEqual(a['prefix_pool_rows'], 0 + 0 + 128 + 192 + 64 + 64)
        self.assertEqual(a['free_slots'], 64 + 192 + 64 + 64)
        self.assertEqual(stats['mask_work']['free_slots'], 128)

    def test_order_not_merely_totals(self):
        row = shifted_obs()
        row['events'][4], row['events'][5] = row['events'][5], row['events'][4]
        with self.assertRaisesRegex(ValueError, 'order'):
            replay(row, 256)

    def test_post_ready_cannot_be_used_early(self):
        row = shifted_obs()
        row['trace'][1]['ready'] -= 1
        with self.assertRaisesRegex(ValueError, 'ready'):
            replay(row, 256)

    def test_reject_partial_and_wrong_frozen_identity(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            policy = 'ibs_cap_32'
            marker = folder / f'complete-{policy}.json'
            tasks = {0: dict(index=0, dataset='a', seed=1, task_id='a/b')}
            for name in ('run.json', 'qualified-logits.json'):
                (folder / name).touch()
            for n, message in ((89, 'marker'), (90, 'inventory')):
                marker.write_text(json.dumps(dict(policy=policy, requests=n)))
                with self.assertRaisesRegex(ValueError, message):
                    list(verified_rows(folder, policy, tasks, ownership(folder)))
            (folder / f'00-{policy}.json').write_text(json.dumps(dict(tasks[0], seed=2, model='gemma',
                                                                      policy=policy, status='success')))
            with self.assertRaisesRegex(ValueError, 'frozen request'):
                list(verified_rows(folder, policy, tasks, ownership(folder)))
