"""Synthetic tensors only; no model weights."""
from unittest import TestCase
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from experiments.t2t.serving_slots.gemma_rejected_noise import RejectNoise, noise_receipt
from experiments.t2t.serving_slots.gemma_rejected_noise_eval import QUAL, TIERS, path, qualify
from experiments.t2t.serving_slots.layout_sweep import write
from experiments.t2t.serving_slots.gemma_strict_sampler import SelectedRowSampler


class TestRejectNoise(TestCase):
    def test_only_warm_still_rejected(self):
        rows=(0,2,5,6,9,11)
        ids=torch.tensor([[3,3,3,3,3,3]])
        old=torch.tensor([[False,False,True,True,False,False]])
        after=torch.tensor([[False,False,False,True,True,False]])
        baseline=torch.tensor([[8,7,9,6,4,3]])
        seen={2,5,6,9,11}
        for on in (False,True):
            p={}
            actual=RejectNoise(on,p)(rows,seen,ids,old,after,baseline)
            self.assertEqual(actual.tolist(),[[8,3 if on else 7,9,6,4,3]])
            self.assertEqual(p['renoise'],{'eligible':[2,11],'baseline_changed':[0,2,5,6,9]})

    def test_rng_accept_sc_unchanged_through_first_exposure(self):
        arm=[SelectedRowSampler(cap=32,vocab_size=47,seed=21) for _ in range(3)]
        rows=tuple(range(32))
        logits=torch.zeros(1,32,47)
        for tick in range(2):
            marks=[{},{}]
            plain,off,on=[s.consume(rows,logits,**({'after_renoise':RejectNoise(bool(j-1),marks[j-1])}
                          if j else {})) for j,s in enumerate(arm)]
            self.assertEqual(plain,off)
            self.assertEqual(plain,on)
            self.assertTrue(torch.equal(arm[0].ids,arm[1].ids))
            self.assertEqual(marks[0],marks[1])
            for s in arm[1:]:
                self.assertTrue(torch.equal(arm[0].rng.get_state(),s.rng.get_state()))
                self.assertTrue(torch.equal(arm[0].accepted,s.accepted))
                self.assertTrue(all(torch.equal(arm[0].sc[r],s.sc[r]) for r in rows))
            if tick==0:
                self.assertFalse(marks[1]['renoise']['eligible'])
                self.assertTrue(torch.equal(arm[0].ids,arm[2].ids))
            else:
                self.assertFalse(torch.equal(arm[0].ids,arm[2].ids))

    def test_replay_rejects_cold_as_warm_or_unpaid_flip(self):
        p=dict(ordinal=0,phase='sample',rows=[1],sampled=[1],admitted=[1],revoked=[],
               input_changed=[],dirty_before=[],dirty_after=[1],
               renoise=dict(eligible=[],baseline_changed=[]))
        e=dict(ordinal=0,route='sparse',tick=0,parts=[p])
        for bad,msg in (({'dirty_after':[]},'Dirty ledger'),
                        ({'renoise':dict(eligible=[1],baseline_changed=[])},'eligibility')):
            with self.assertRaisesRegex(ValueError,msg):
                noise_receipt([{**e,'parts':[{**p,**bad}]}],True)


class TestQualificationGate(TestCase):
    def cases(self,root,*,mismatch=False,unexposed=False):
        hist=root/'hist'
        hist.mkdir()
        for c in TIERS:
            for i in QUAL:
                common=dict(index=i,budget=c,seed=17000+i,dataset='test',task_id=str(i),
                            input_sha256='prompt',samples_sha256='samples',scorer_sha256='scorer',
                            F=1,N=1,output_ids=[42],correct=True,decoder_F=1,encoder_F=0,
                            decoder_Q=c,encoder_Q=0,route_F={'sparse':1},route_Q={'sparse':c},
                            max_decoder_Q=c)
                history=dict(**common,schema='paid-close-v1',fuse=False,gate='off',prefix=[1,8],slack=False)
                if mismatch and (c,i)==(64,10):
                    history['output_ids']=[43]
                write(hist/f'C{c}-fuse0-{i:02d}.json',history)
                for on in (False,True):
                    actual=int(not unexposed or c!=128)
                    avoided=actual*on
                    noise=dict(eligible=1,warm_rows=1,sampled_rows=c,would_change=actual,avoided=avoided,
                               token_changes=actual-avoided,baseline_token_changes=actual,
                               drain_F=0,drain_Q=0,first_exposure={'tick':1} if actual else None)
                    write(path(root,c,i,on),dict(**common,schema='reject-noise-v1',keep=on,
                          source_sha256={'sampler':'sha'},gpu_visible='GPU-test',policy={},
                          checkpoint='pinned',noise=noise))
        return hist

    def test_history_token_drift_refuses_promotion(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            hist=self.cases(root,mismatch=True)
            with self.assertRaisesRegex(ValueError,'C64 #10.*output_ids'):
                qualify(root,hist)

    def test_eligible_but_identical_redraw_is_not_exposure(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            hist=self.cases(root,unexposed=True)
            with self.assertRaisesRegex(ValueError,'C128: no effective exposure'):
                qualify(root,hist)
