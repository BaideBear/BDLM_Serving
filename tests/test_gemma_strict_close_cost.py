import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.t2t.serving_slots import gemma_strict_close_cost as cost


def block(drains=(), prefix=()):
    trace = []

    def event(phase, rows, dirty=()):
        trace.append(dict(phase=phase, tick=len(trace), ordinal=0, rows=list(rows), q=len(rows),
                          dirty_before=list(dirty), sampled=list(rows) if phase == 'sample' else [],
                          encoded=False, stop_reason='resolved' if phase == 'verify' else None))

    for phase, rows in prefix:
        event(phase, rows)
    event('sample', [3])
    for i, rows in enumerate(drains):
        event('drain', rows, [r for later in drains[i:] for r in later])
    event('verify', range(128))
    event('verify', range(128, 256))
    return trace


class CloseGeometry(unittest.TestCase):
    def test_paid_drain_labels_survive_partial_writes(self):
        def step(phase,dirty,rows,a=(),t=(),r=()):
            return dict(phase=phase,dirty_before=dirty,rows=rows,admitted=a,edited=t,revoked=r)
        tr = [step('sample',[],[0,1,2],[0]), step('drain',[0,1,2],[1]),
              step('sample',[0,2],[0,2,3],[],[0],[2]), step('drain',[0,2,3],[0,2,3])]
        self.assertEqual(cost.drain_composition(tr),dict(drain_write_or_flip_Q=2,
                         drain_unaccepted_noise_Q=2,pure_drains=2))
        tr[-1]['dirty_before'] = [0,3]
        with self.assertRaisesRegex(ValueError,'unpaid'):
            cost.drain_composition(tr)

    def test_inside_first_partition(self):
        b = cost.close_geometry(block([[0, 127]]))
        self.assertEqual((b['original_order_save_F'], b['original_order_save_Q']), (1, 2))
        self.assertEqual(b['first_verify_plus_all_dirty_Q'], 128)

    def test_tail_only_or_one_tail_row_is_not_overwritten(self):
        for rows, overlap in [([128], [0, 1]), ([0, 127, 128], [2, 1])]:
            b = cost.close_geometry(block([rows]))
            self.assertEqual(b['dirty_by_verify_chunk'], overlap)
            self.assertEqual((b['original_order_save_F'], b['original_order_save_Q']), (0, 0))
            self.assertEqual((b['repacked_ceiling_F'], b['repacked_ceiling_Q']), (1, len(rows)))
            self.assertEqual(b['first_verify_plus_all_dirty_Q'], 129)

    def test_nonterminal_barrier_not_saved(self):
        b = cost.close_geometry(block(prefix=[('sample', [0]), ('drain', [0])]))
        self.assertEqual((b['other_drain_F'], b['other_drain_Q']), (1, 1))
        self.assertEqual((b['repacked_ceiling_F'], b['repacked_ceiling_Q']), (0, 0))

    def test_multiple_drains_with_intervening_read(self):
        b = cost.close_geometry(block([range(128), [255]]))
        self.assertEqual((b['original_order_save_F'], b['repacked_ceiling_F'], b['repacked_ceiling_Q']),
                         (0, 2, 129))

    def test_reject_bad_sweep_and_bad_debt(self):
        tape = block([[127]])
        tape[-1]['rows'][0] = 127
        with self.assertRaisesRegex(ValueError, 'sweep'):
            cost.close_geometry(tape)
        tape = block([[127]])
        tape[1]['dirty_before'].append(128)
        with self.assertRaisesRegex(ValueError, 'drain'):
            cost.close_geometry(tape)

    def test_lone_arm_does_not_enter_totals(self):
        samples = [dict(index=i) for i in range(90)]
        with patch.object(cost, 'read', side_effect=[({'samples': samples}, ''), ({'added_tokens': []}, '')]), \
                patch.object(Path, 'is_file', lambda p: 'k0-' in p.name), patch.object(cost, 'case_arm') as arm:
            result = cost.analyze()
        arm.assert_not_called()
        self.assertEqual(result['paired_indices'], [])
        self.assertEqual([x['missing_compact_k'] for x in result['excluded']], [[32]]*3)

    @unittest.skipUnless(cost.ROOT.exists() and cost.TOKENIZER.exists(), 'requires sealed n164 archive')
    def test_frozen_receipts_and_privacy(self):
        report = cost.analyze()
        self.assertEqual(report['paired_indices'], [0, 10, 20])
        self.assertEqual([(t['N'], t['F'], t['Q'], *(t[k] for k in cost.BOUNDS)) for t in report['totals']],
                         [(911, 76, 8511, 2, 18, 5, 65), (913, 79, 8988, 2, 3, 5, 22)])
        text = json.dumps(report)
        for key in ('output_text', 'output_ids', 'tokens', 'prompt', 'encoder_inputs', 'extracted', 'reference'):
            self.assertNotIn('"' + key + '"' + ':', text)

    @unittest.skipUnless(cost.ROOT.exists(), 'requires n164 projection tape')
    def test_projection_corruption_fails(self):
        tag = 'cpu-q128-k0-00'
        ev, _ = cost.read(cost.ROOT/'raw'/tag/'evidence.json')
        compact, _ = cost.read(cost.ROOT/'compact'/(tag+'.json'))
        ev['projection_widths']['0.q_proj'][0] -= 1
        with self.assertRaisesRegex(ValueError, 'projection tape'):
            cost.audit_tape(ev, compact, set())
