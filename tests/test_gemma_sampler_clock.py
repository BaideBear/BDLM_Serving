"""CPU diagnostic: isolate shared sampler RNG when a resident tail is parked."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
import torch


def replay(mode, seed):
    native = pytest.importorskip("transformers.models.diffusion_gemma.generation_diffusion_gemma")
    from transformers.generation.logits_process import LogitsProcessorList

    class Observed(native.EntropyBoundSampler):
        def accept_canvas(self, current_canvas, denoiser_canvas, logits, cur_step):
            self.proposal = denoiser_canvas.clone()
            return super().accept_canvas(current_canvas, denoiser_canvas, logits, cur_step)

    if mode not in ("active","park","park_shadow"):
        raise ValueError(mode)
    b, v = 256, 23
    # Different row entropies, with both accepted and renoised sites.
    logits = torch.zeros((1,b,v),dtype=torch.float32)
    logits[0,:,0] = torch.linspace(0,4,b)
    def canvas():
        sampler = Observed(native.EntropyBoundSamplerConfig(16.),b,v,8)
        ids = sampler.initialize_canvas(1,torch.device("cpu"))
        return NS(s=sampler,ids=ids,arg=ids,sc=None,age=0)
    def step(x):
        x.ids,x.arg,x.sc,_ = native.DiffusionGemmaGenerationMixin._denoising_step(
            NS(config=NS(text_config=NS(vocab_size=v)),model=NS(decoder=NS(embed_tokens=NS(weight=logits)))),
            decoder_forward=lambda **_:NS(logits=logits),current_canvas=x.ids,
            argmax_canvas=x.arg,input_ids=torch.tensor([[1]]),decoder_position_ids=None,
            self_conditioning_logits=x.sc,mask_mapping={},past_key_values=None,
            finished_denoising=torch.tensor([False]),cur_step=8-x.age,sampler=x.s,
            logits_processor=LogitsProcessorList(),diffusion_stopping_criteria=None)
        x.age += 1
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        head,tail=canvas(),canvas()
        step(head)
        step(tail)
        paused_ids=tail.ids.clone()
        paused_mask=tail.s.accepted_token_mask.clone()
        path=[]
        for _ in range(3):
            step(head)
            path.append((head.s.proposal.clone(),head.ids.clone(),torch.get_rng_state().clone(),
                         head.s.accepted_token_mask.clone()))
            if mode=="active":
                step(tail)
            elif mode=="park_shadow":
                # Diagnostic only: identical native draw shape, discarded state, no model forward.
                step(deepcopy(tail))
        if mode!="active":
            assert tail.age==1 and torch.equal(tail.ids,paused_ids)
            assert torch.equal(tail.s.accepted_token_mask,paused_mask)
        return path


@pytest.mark.parametrize("seed",[0,7,19])
def test_tail_parking_can_change_head_with_fixed_logits(seed):
    active,park,shadow=[replay(mode,seed) for mode in ("active","park","park_shadow")]
    for j,(a,p,s) in enumerate(zip(active,park,shadow,strict=True)):
        proposals,ids,rng,mask=a
        assert mask.any() and not mask.all()
        assert all(torch.equal(x,y) for x,y in zip(a,s,strict=True))
        if not j:
            assert all(torch.equal(x,y) for x,y in zip(a,p,strict=True))
        else:
            assert not torch.equal(rng,p[2]) and not torch.equal(ids,p[1])
            assert not torch.equal(proposals[:,mask[0]],p[0][:,mask[0]])
