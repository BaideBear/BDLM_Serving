"""Real common planner through the Gemma adapter. No weights, tensors or GPU."""
from fractions import Fraction
from types import SimpleNamespace as NS
import unittest

from bdlm_serving.methods.t2t import unified_rows as shared
from experiments.t2t.serving_slots import unified_gemma as g


def canvas(ordinal, accepted=(), initialized=(), dirty=(), sampling=True):
    return NS(ordinal=ordinal, rows=NS(accepted=frozenset(accepted), initialized=frozenset(initialized),
                                      dirty=frozenset(dirty)), can_sample=sampling,
              sampler=NS(sc=dict.fromkeys(initialized)))


class Dispatch(g._Unified):
    def __init__(self, h=None, tail=None):
        self.head = h or canvas(0)
        self.ordinal, self.origin = self.head.ordinal, 7
        self.spawned = self.cursor = 0
        self.max_new_tokens = 3*g.B
        self.rebuild, self.tail, self.params = False, tail, shared.Params()
        self.bank = NS(live={self.origin + c.ordinal*g.B + i for c in (self.head, tail)
                             if c is not None for i in c.rows.initialized})

    def _head_canvas(self):
        return self.head

    def _spawn(self):
        self.spawned += 1
        return canvas(self.ordinal+1)

    def _execute(self, states, q, route, pos, *, decision=None, opened):
        return route, pos, decision, opened


class TestUnifiedGemma(unittest.TestCase):
    def test_all_offers_exact_including_cold_and_admission(self):
        for q in g.OFFERS:
            with self.subTest(q=q):
                r = Dispatch()
                a = shared.Block(7, 256, range(7,263), (), ())
                b = shared.Block(263, 256, range(263,519), (), ())
                expected = shared.plan_rows(q, 256, a, b)
                route, pos, out, opened = r._advance(q)
                self.assertFalse(opened)
                self.assertEqual((route, pos, out), ("ordinary", expected.rows, expected))
                self.assertEqual(r.spawned, int(expected.open_next))
                self.assertEqual(out.cold, out.mask)

    def test_sticky_tail_drained_on_downshift_without_sampling_or_spawning(self):
        r = Dispatch(canvas(0, initialized=g.FULL, dirty={1, 2}),
                     canvas(1, initialized=g.FULL, dirty=range(64)))
        _, pos, out, opened = r._advance(32)
        self.assertTrue(opened and out.open_next and out.drain_only)
        self.assertEqual(out.sampled, ())
        self.assertEqual(pos, (8, 9, *range(263, 293)))
        self.assertEqual(r.spawned, 0)

    def test_prefix_mask_and_debt_not_reclassified_in_mixed_measurement(self):
        valid = g.FULL - {255}
        h = canvas(0, accepted=range(145), initialized=valid, dirty={141, 146})
        r = Dispatch(h)
        r.params = shared.Params(prefix_share=Fraction(1, 4))
        _, pos, out, _ = r._advance(32)
        expected = shared.plan_rows(32,256,
            shared.Block(7,256,range(152,263),{148,153},range(7,262)),
            shared.Block(263,256,range(263,519),(),()), params=r.params)
        self.assertEqual(out, expected)
        self.assertEqual(out.cold, (262,))
        self.assertTrue(out.prefix)
        self.assertTrue(set(out.sampled) == set(out.mask) | set(out.prefix))
        self.assertFalse(set(out.sampled) & set(out.dirty))
        self.assertEqual(pos, out.rows)

    def test_last_valid_head_cannot_spawn(self):
        r = Dispatch(canvas(2))
        self.assertEqual(r._advance(512)[1], tuple(range(519,775)))
        self.assertIsNone(r.tail)

    def test_terminal_pays_tail_then_ordered_verify_without_reopening(self):
        r = Dispatch(canvas(0, g.FULL, g.FULL, sampling=False),
                     canvas(1, initialized={0, 1}, dirty={0}))
        _, pos, out, _ = r._advance(512)
        self.assertEqual(pos, (263,))
        self.assertTrue(out.drain_only and out.open_next)
        r.tail.rows.dirty = frozenset()
        r.cursor = 224
        route, pos, out, _ = r._advance(128)
        self.assertEqual((route, pos, out), ("verify", tuple(range(231,263)), None))
        r.head.sampler.sc.pop(0)
        with self.assertRaisesRegex(RuntimeError, "coverage"):
            r._advance(128)

    def test_rebuild_charges_only_missing_epoch_rows(self):
        r = Dispatch(canvas(1, initialized={1,8,9}))
        r.bank.live.remove(264)
        with self.assertRaisesRegex(RuntimeError, "rebuild"):
            r._advance(32)
        r.rebuild = True
        self.assertEqual(r._advance(32), ("rebuild", (264,), None, False))

    def test_request_denominator_and_encoder_separate_from_te(self):
        class Driver:
            def __init__(self, model, data, *, q, **kw):
                self.status, self.fixed_q, self.origin = "ready", q, 7
            def step(self, q):
                if q == 64:
                    raise RuntimeError("partial write")
                self.status, self.tokens = "done", [3,4,5]
                return {}
        class Req(g._Unified, Driver):
            pass
        m = g.RequestMeter()
        a, b = [Req(None, None, q=32, seed=1, meter=m) for _ in range(2)]
        with self.assertRaisesRegex(RuntimeError, "partial"):
            b.step(64)
        with self.assertRaisesRegex(RuntimeError, "partial"):
            b.completed_bill(set())
        a.step()
        a.decoder_F, a.decoder_Q = 2, 64
        a.encoder_receipts = [dict(prefill=True,Q=7),dict(prefill=False,Q=256)]
        bill = a.completed_bill({5})
        self.assertEqual((m.started,m.completed,m.failed,m.in_flight), (2,1,1,0))
        self.assertEqual((bill["requests"],bill["TPF"],bill["TE"]), (1,2/3,2/64))
        with self.assertRaises(RuntimeError):
            a.step()
        self.assertEqual(m.completed,1)
