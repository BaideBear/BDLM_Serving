"""No model: synthetic score tapes, native criteria/denoising step, real fixed planner."""
from types import SimpleNamespace as NS
from unittest import TestCase, skipUnless
from unittest.mock import Mock

import torch

from experiments.t2t.serving_slots.gemma_fullstop import B, FULL, FullHeadStop, HeadSample, HeadTick

try:
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as native
except ModuleNotFoundError:
    native = None


def tape(margin=9.):
    scores = torch.zeros(1, B, 3)
    scores[:, :, 1] = margin
    return scores


def publish(obs, tick, scores, *, ordinal=0, epoch=13, rows=FULL):
    stamp = HeadTick(tick, ordinal, epoch)
    return obs.observe(stamp, phase="sample", sample=HeadSample(stamp, rows, scores))


class TestGuards(TestCase):
    def test_registered_pairs_and_qualification(self):
        from experiments.t2t.serving_slots.cross_race import fullstop_cases
        all_cases = list(fullstop_cases())
        self.assertEqual(len(set(all_cases)), 2*(2*90+3))
        lanes = [set(fullstop_cases(lane=k)) for k in (0,1)]
        self.assertFalse(lanes[0] & lanes[1])
        self.assertEqual(set.union(*lanes), set(all_cases))
        for lane in lanes:
            for case in lane:
                self.assertIn((*case[:-1],not case[-1]), lane)
                self.assertEqual(case[2], 'half' if case[1]==512 else 'off')
        self.assertEqual({x[0] for x in fullstop_cases(budget=128)}, {0,10,20})
        self.assertEqual(len(list(fullstop_cases(qualify=True, arm='on'))), 9)

    def test_only_current_ordered_head_sample_reaches_criterion(self):
        criterion = Mock(return_value=torch.tensor([False]))
        obs = FullHeadStop(criterion)
        current = HeadTick(3, 0, 13)
        scores = tape()
        obs.observe(current, phase="sample", sample=HeadSample(current, FULL, scores))
        draft, actual = criterion.call_args.args
        self.assertIs(actual, scores)
        self.assertTrue(torch.equal(draft, scores.argmax(-1)))
        for row_ids in (tuple(reversed(FULL)), (0,)*B, tuple(range(B, 2*B))):
            with self.assertRaises(ValueError):
                publish(obs, obs.last.tick + 1, scores, rows=row_ids)
        for bad in (scores.bfloat16(), scores[:, :-1], scores.repeat(1, 2, 1)):
            with self.assertRaises(ValueError):
                publish(obs, obs.last.tick + 1, bad)
        for dt, ordinal, epoch in ((-1, 0, 13), (0, 1, 13), (0, 0, 12)):
            current = HeadTick(obs.last.tick + 1, 0, 13)
            old = HeadTick(current.tick + dt, ordinal, epoch)
            with self.assertRaisesRegex(ValueError, "fresh"):
                obs.observe(current, phase="sample", sample=HeadSample(old, FULL, scores))
        with self.assertRaisesRegex(ValueError, "Replayed"):
            obs.observe(current, phase="drain")
        self.assertEqual(criterion.call_count, 1)
        self.assertFalse(publish(FullHeadStop(None), 0, scores))


@skipUnless(native, "requires pinned DiffusionGemma generation code")
class TestPinned(TestCase):
    def observer(self):
        cfg = NS(stability_threshold=1, confidence_threshold=.005)
        # The factory uses only generation_config; no nn.Module or checkpoint.
        return FullHeadStop.from_model(native.DiffusionGemmaGenerationMixin(), cfg)

    def test_reset_and_full_entropy(self):
        scores = tape()
        for phase in ("sample", "absent", "verify", "drain", "rebuild"):
            obs = self.observer()
            self.assertFalse(publish(obs, 0, scores))
            if phase == "sample":
                # Two complementary half-calls must never be stitched into B.
                self.assertFalse(publish(obs, 1, scores[:, :128], rows=FULL[:128]))
                self.assertFalse(publish(obs, 2, scores[:, 128:], rows=FULL[128:]))
            else:
                self.assertFalse(obs.observe(HeadTick(1, 0, 13), phase=phase))
            next_tick = obs.last.tick + 1
            self.assertFalse(publish(obs, next_tick, scores))
            self.assertTrue(publish(obs, next_tick+1, scores))
        obs = self.observer()
        self.assertFalse(publish(obs, 0, scores))
        self.assertFalse(publish(obs, 2, scores))  # Missed decoder tick.
        self.assertFalse(publish(obs, 3, scores, epoch=14))
        self.assertFalse(publish(obs, 4, scores, epoch=14, ordinal=1))
        last_row_edit = scores.clone()
        last_row_edit[0, -1] = scores[0, -1].roll(1)
        self.assertFalse(publish(obs, 5, last_row_edit, epoch=14, ordinal=1))
        self.assertTrue(publish(obs, 6, last_row_edit, epoch=14, ordinal=1))
        self.assertFalse(publish(obs, 7, last_row_edit * .1, epoch=14, ordinal=1))

    def test_distinct_full_offer_windows(self):
        scores, obs = tape(), self.observer()
        hits = []
        for tick, q in enumerate((512,256,512,512,256,256)):
            stamp = HeadTick(tick,0,13,q)
            hits.append(obs.observe(stamp,phase="sample",sample=HeadSample(stamp,FULL,scores)))
        self.assertEqual(hits,[False,False,False,True,False,True])

    def test_all_accepted_is_not_a_stop_and_confidence_bound_is_strict(self):
        scores = tape(30.)
        ids = torch.zeros(1, B, dtype=torch.long)
        sampler = native.EntropyBoundSampler(native.EntropyBoundSamplerConfig(.1), B, 3, 48)
        sampler.accept_canvas(ids, scores.argmax(-1), scores, 48)
        self.assertTrue(sampler.accepted_token_mask.all())
        self.assertFalse(publish(self.observer(), 0, scores))  # No prior draft.
        threshold = torch.distributions.Categorical(logits=scores).entropy().mean().item()
        obs = FullHeadStop(native.StableAndConfidentStoppingCriteria(1, threshold))
        self.assertFalse(publish(obs, 0, scores))
        self.assertFalse(publish(obs, 1, scores))  # Stable, but equal is not below.

    def test_last_iteration_processed_args_and_budget_is_not_a_hit(self):
        from transformers.generation.logits_process import LogitsProcessorList
        temp = native.LinearTemperatureScheduleLogitsProcessor(.4, .8, 48)
        ref = native.StableAndConfidentStoppingCriteria(1, .005)
        # Raw margin 4 is not confident; processed margin ~9.8 at t=1 is.
        for margin, expected in ((4., True), (.1, False)):
            ref.reset()
            raw = tape(margin)
            ids = torch.zeros(1, B, dtype=torch.long)
            sampler = native.EntropyBoundSampler(native.EntropyBoundSamplerConfig(.1), B, 3, 48)
            shell = NS(config=NS(text_config=NS(vocab_size=3)),
                       model=NS(decoder=NS(embed_tokens=NS(weight=torch.empty(0, dtype=torch.bfloat16)))))
            obs = self.observer()
            for tick, remaining in enumerate((2, 1)):
                def check(draft, processed):
                    self.assertIsNotNone(sampler.accepted_token_mask)
                    self.assertTrue(torch.equal(processed, temp(ids, raw, cur_step=torch.tensor(remaining))))
                    self.assertTrue(torch.equal(draft, processed.argmax(-1)))
                    before = torch.get_rng_state()
                    hit = publish(obs, tick, processed)
                    self.assertTrue(torch.equal(before, torch.get_rng_state()))
                    decision = ref(draft, processed)
                    self.assertEqual(hit, decision.item())
                    return decision
                with torch.random.fork_rng(devices=[]):
                    _, draft, sc, finished = native.DiffusionGemmaGenerationMixin._denoising_step(
                        shell, decoder_forward=lambda **_: NS(logits=raw), current_canvas=ids,
                        argmax_canvas=ids, input_ids=ids[:, :1], decoder_position_ids=None,
                        self_conditioning_logits=None, mask_mapping={}, past_key_values=None,
                        finished_denoising=torch.tensor([False]), cur_step=remaining, sampler=sampler,
                        logits_processor=LogitsProcessorList([temp]), diffusion_stopping_criteria=check)
                self.assertTrue(torch.equal(draft, raw.argmax(-1)))
                self.assertEqual(sc.dtype, torch.bfloat16)
                self.assertEqual(bool(finished), tick == 1 and expected)

    def test_fixed_joint_hit_pays_head_and_tail_debt_then_verifies(self):
        from experiments.t2t.serving_slots.gemma_fixed_dual import (
            AblationRequest, SelectedRowSampler, StrictRows,
        )
        from experiments.t2t.serving_slots.gemma_fixed_dual_eval import fullstop_receipt
        from bdlm_serving.methods.t2t.budget_family import Law

        for enabled in (False, True):
            with self.assertRaisesRegex(ValueError, "unfused"):
                AblationRequest(None, None, q=512, seed=23, fullstop=enabled, fuse_first_verify=True)

        def run(enabled, q=512, dense_stop=False, switched=False):
            r = object.__new__(AblationRequest)
            r.origin, r.ordinal, r.cursor, r.tick, r.seed = 13, 0, 0, 0, 23
            r.status, r.tokens, r.eos, r.max_new_tokens = "ready", [], {2}, B*2
            r.ids = torch.zeros(1, r.origin, dtype=torch.long)
            r.ablation, r.law, r.rebuild, r.tail, r.fixed_q = "00", Law(), False, None, q
            r.rows = StrictRows(48)
            s = r.sampler = SelectedRowSampler(cap=B, vocab_size=3, seed=r.seed)
            r.stop_observer, r.fullstop = self.observer(), enabled
            if dense_stop:
                r.fuse_first_verify = r.full_head_at_b = True
                r.top_gate = 'off'
            if switched:
                r.stop_at_single_b_only=True
                r.max_new_tokens=B
            r.cache = NS(get_seq_length=lambda: r.origin)
            events = []
            def project(states, pos):
                raw = torch.zeros(1, len(pos), 3)
                if not states[0].can_sample and (dense_stop or not any(c.rows.dirty for c in states)):
                    raw[:, :, 2] = 20.  # EOS only in the real verify.
                else:
                    raw[:, :, 1] = 9*(.4 + .4*(48-s.age)/48)
                return raw
            r._project = project
            while r.status == "ready" and len(events) < 200:
                if switched:
                    sequence=(512,256,512,256,256)
                    events.append(r.step(sequence[len(events)] if len(events)<len(sequence) else 32))
                else:
                    events.append(r.step())
            self.assertEqual(r.tokens, [2])  # Not the observer's argmax token 1.
            self.assertEqual(r.status, "done")
            return events, None if switched else fullstop_receipt(events, r.origin, enabled, dense_stop=dense_stop)

        control, shadow = run(False)
        trial, active = run(True)
        self.assertEqual([e['parts'] for e in control[:2]], [e['parts'] for e in trial[:2]])
        self.assertEqual((shadow['applied_hits'], shadow['first_hit']['tick']), (0,1))
        self.assertEqual((active['positive_ticks'], active['actionable_hits'], active['applied_hits']), (1,1,1))
        self.assertEqual(active['head_tick_tape'], 'FFDV')
        self.assertEqual(active['first_hit']['tick'], 1)
        self.assertEqual([e["route"] for e in trial], ["dense_joint"]*2 + ["drain", "verify"])
        due = trial[1]["dirty_after_absolute"]
        self.assertTrue(any(p < 13+B for p in due) and any(p >= 13+B for p in due))
        self.assertEqual(trial[2]["positions"], due)
        self.assertFalse(trial[2]["sampled_absolute"])
        self.assertEqual(trial[3]["positions"], list(range(13, 13+B)))
        self.assertEqual(trial[3]["dirty_before_absolute"], [])
        self.assertEqual((len(control), len(trial)), (50, 4))
        short0, off = run(False,128)
        short1, on = run(True,128)
        self.assertEqual(short0, short1)
        self.assertEqual((off['qualified_full_ticks'],on['positive_ticks']), (0,0))
        self.assertGreater(on['resets']['partial'],0)
        dense, obs = run(True, B, True)
        self.assertEqual(obs['head_tick_tape'], 'FFV')
        self.assertEqual([e['route'] for e in dense], ['dense_single']*2+['verify_paid'])
        verify = dense[-1]
        self.assertTrue(verify['dirty_before_absolute'])
        self.assertFalse(verify['sampled_absolute'] or verify['dirty_after_absolute'])
        self.assertEqual(verify['parts'][0]['age_before'],verify['parts'][0]['age_after'])
        self.assertEqual(verify['positions'],list(range(13,13+B)))
        control, obs = run(False, B, True)
        self.assertEqual(obs['applied_hits'],0)
        self.assertGreater(len(control),len(dense))
        steps,_=run(True,B,True,switched=True)
        self.assertEqual([e['fullstop']['latched'] for e in steps[:5]],[False]*4+[True])
        self.assertEqual([e['fullstop']['in_scope'] for e in steps[:5]],[False,True,False,True,True])
        p=steps[4]['parts'][0]
        self.assertGreater(steps[4]['head_mask_before']-len(p['admitted'])+len(p['revoked']),0)
        self.assertLess(p['age_after'],48)
        self.assertTrue(steps[4]['dirty_after_absolute'])
        self.assertTrue(all(e['offer']==32 and not e['sampled_absolute'] and e['q']<=32 for e in steps[5:]))
        self.assertEqual(sum(e['q'] for e in steps[5:] if e['route'].startswith('verify')),B)
