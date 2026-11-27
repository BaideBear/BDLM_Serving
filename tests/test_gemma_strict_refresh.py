"""CPU tests of the fixed-cap Optimus hook; no scorer or checkpoint loads."""
from collections import defaultdict
from contextlib import ExitStack
from hashlib import sha256

import numpy as np
import pytest
import torch

from experiments.t2t.serving_slots import gemma_strict_refresh as g


def state(mode, *, full=True):
    base = np.arange(4, 36 if full else 34)
    prior = np.zeros(48, dtype=np.uint8)
    prior[:4] = 1
    prior[[5, 18, 28, 33, 35]] = 1
    seen = np.ones(48, dtype=bool)
    dirty = np.zeros_like(seen)
    dirty[33] = True
    return base, dict(mode=mode, cap=32, policy=g.ARM[mode], ready=4,
                      prior=prior, dirty=dirty, seen=seen)


@pytest.mark.parametrize('mode', ['ibs', 'obs'])
def test_roles_and_identity_on_ineligible_ticks(mode):
    b, kw = state(mode, full=mode == 'obs')
    rows, edit = g.choose(b, **kw)
    assert edit == g.Edit(3, 28 if mode == 'obs' else None)
    assert len(rows) == len(b) + (mode == 'ibs') and rows[-1] == b[-1]
    mask = kw['prior'] == 0
    assert np.array_equal(rows[mask[rows]], b[mask[b]]) and 33 in rows
    for empty in (dict(ready=0), dict(prior=np.ones_like(kw['prior'])), dict(policy='sham')):
        same, nop = g.choose(b, **(kw | empty))
        assert same is b and nop == g.Edit()
    b, kw = state(mode, full=mode == 'ibs')
    assert g.choose(b, **kw)[0] is b


def test_fail_closed_and_never_donate_edge_dirty_or_first_visit():
    b, kw = state('obs')
    kw['seen'][[5, 18, 28]] = False
    assert g.choose(b, **kw)[0] is b
    kw['dirty'][1] = True
    with pytest.raises(ValueError, match='stable'):
        g.choose(b, **kw)
    with pytest.raises(ValueError, match='unsupported'):
        g.choose(b, **(kw | dict(policy='slack_k1')))


def test_resident_writeback_overwrites_prefix_not_donor_across_layers():
    base, kw = state('obs')
    rows, edit = g.choose(base, **kw)
    bank = g.opt.Resident()
    start = 256
    for positions, offset in [(np.arange(36), 0), (rows, 1000)]:
        bank.select(start, start + positions, 'cpu')
        for layer in (0, 1):
            q = torch.zeros(1, 1, len(positions), 1)
            k = torch.tensor(positions + offset + layer).reshape_as(q)
            live_k, live_v = bank.extend(layer, q, k, -k)
            assert torch.equal(live_v, -live_k) and live_k.shape[-2] == 36
        assert bank.visited == {0, 1}
    for layer, (k, v) in bank.buffers.items():
        for pos, delta in [(edit.add, 1000), (edit.drop, 0)]:
            assert k[0, 0, start + pos, 0] == pos + delta + layer
            assert v[0, 0, start + pos, 0] == -(pos + delta + layer)


class Probe:
    """Independent ID/SC/logit oracle addressed by absolute generated position."""
    def __init__(self, unmask, prompt_len, vocab):
        self.unmask, self.plen = unmask, prompt_len
        self.sc = torch.zeros(g.opt.CAP, vocab)
        self.seen = torch.zeros(g.opt.CAP, dtype=torch.bool)
        self.tokens = torch.full((g.opt.CAP,), -1)
        self.widths = defaultdict(list)
        self.digests = []
        self.positions = []

    def before(self, _, args, kw):
        self.pos = kw['decoder_position_ids'][0].long() - self.plen
        self.positions.append(self.pos.tolist())
        old = self.tokens[self.pos]
        assert torch.equal(kw['decoder_input_ids'][0][old >= 0], old[old >= 0])
        sc = kw['self_conditioning_logits']
        if self.seen[self.pos].any():
            assert torch.equal(sc[0][self.seen[self.pos]], self.sc[self.pos][self.seen[self.pos]])
        else:
            assert sc is None

    def after(self, _, args, result):
        self.logits = result.logits[0].detach()
        assert torch.isfinite(self.logits).all()
        self.sc[self.pos] = self.logits
        self.seen[self.pos] = True
        self.digests.append(sha256(self.logits.contiguous().numpy().tobytes()).digest())

    def low_confidence_dynamic_sparse(self, logits, proposals, prior, offsets, row_width):
        mask = self.tokens[self.pos] < 0
        pos = self.pos[mask]
        # Tiny test deliberately ends after the first B256 retirement.
        assert torch.equal(pos, offsets) and torch.equal(self.logits[mask], logits)
        assert torch.equal(proposals, logits.argmax(-1))
        out = self.unmask.low_confidence_dynamic_sparse(logits, proposals, prior, offsets, row_width)
        took = out[offsets].bool()
        self.tokens[pos[took]] = proposals[took]
        return out

    def width(self, name):
        def hook(_, args):
            self.widths[name].append(args[0].shape[1])
        return hook


@pytest.fixture(scope='module')
def tiny():
    pytest.importorskip('transformers.models.diffusion_gemma.modeling_diffusion_gemma')
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield g.tiny_inputs()
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize('mode', ['ibs', 'obs'])
def test_real_b256_exact_sham_and_active_projection_sc_rng(tiny, mode):
    model, data, controls, unmask = tiny
    outputs = []
    for policy in ('baseline', 'sham', g.ARM[mode]):
        probe = Probe(unmask, data['input_ids'].shape[1], model.config.text_config.vocab_size)
        with ExitStack() as stack, torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(17)
            stack.callback(model.register_forward_pre_hook(probe.before, with_kwargs=True).remove)
            stack.callback(model.register_forward_hook(probe.after).remove)
            modules = {'head': model.lm_head}
            for i, layer in enumerate(model.model.decoder.layers):
                for attr in ('q_proj', 'k_proj', 'v_proj'):
                    mod = getattr(layer.self_attn, attr)
                    if mod is not None:
                        modules[i, attr] = mod
            for name, mod in modules.items():
                stack.callback(mod.register_forward_pre_hook(probe.width(name)).remove)
            r = g.generate_request(model, data, controls, probe, mode=mode, cap=256, policy=policy)
            outputs.append((r, probe.digests, torch.get_rng_state()))
        widths = [s['q'] for s in r['trace']]
        assert probe.positions == [s['positions'] for s in r['trace']]
        assert all(probe.widths[k] == widths for k in modules)
        assert r['decoder_Q'] == sum(widths) and r['decoder_F'] == len(widths)
        assert (r['prefill_Q'], r['encoder_gen_F'], r['encoder_gen_Q']) == (3, 0, 0)
    (base, base_logits, rng), (sham, sham_logits, sham_rng), (active, active_logits, _) = outputs
    assert (base['trace'], base['output_ids'], base['events']) == (sham['trace'], sham['output_ids'], sham['events'])
    assert base_logits == sham_logits and torch.equal(rng, sham_rng)
    edits = [(t, e) for t, e in enumerate(active['edits']) if e['refresh'] is not None]
    assert edits
    first = edits[0][0]
    assert base['trace'][:first] == active['trace'][:first] and base_logits[:first] == active_logits[:first]
    assert active['trace'][first]['visible_end'] == base['trace'][first]['visible_end']
    for i, e in edits:
        s = active['trace'][i]
        assert e['refresh'] in s['positions'] and e['refresh'] not in s['admitted']
        assert e['donor'] not in s['positions'] and (e['donor'] is None) == (mode == 'ibs')
        assert s['q'] == e['base_q'] + (mode == 'ibs') <= 256
    assert active['policy'] != base['policy']
