"""Opt-in, fixed physical decoder cap, K=1 refresh on the pinned Optimus loop.

    python -B -m experiments.t2t.serving_slots.gemma_strict_refresh --mode obs --cap 256

The CLI runs a random-weight, real B256 DiffusionGemma on CPU, not a benchmark.
generate_request accepts an eval model and the caller's ordinary Optimus seed
state. The prototype is single-request, like pinned resident_attention. The
default calls native generate directly; no global functions are monkeypatched.

Optimus admits MASK -> token and re-noises still-unresolved rows. Resolved IDs
are never revised/revoked. Re-projecting one such ID refreshes resident KV and
its SC logits, NOT the token; this is not the native Gemma T2T sampler.
"""
import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np
import torch

from bdlm_serving.runtime import diffusiongemma_optimus as opt
from bdlm_serving.runtime.optimus_limits import validate_trace

PIN = 'a09d04477361cb2a260dac85ce2195c0bfee9218306b7cfb96d7ddf4f1dba9af'
FIXED = (32, 64, 128, 256)
ARM = {'ibs': 'slack_k1', 'obs': 'donor_k1'}


def check_arm(mode, cap, policy):
    if mode not in ARM or cap not in FIXED or policy not in ('baseline', 'sham', ARM[mode]):
        raise ValueError(f'unsupported fixed control: {mode}/{cap}/{policy}')


@dataclass(frozen=True)
class Edit:
    add: int | None = None
    drop: int | None = None


def choose(base, *, mode, cap, policy, ready, prior, dirty, seen):
    """Block-local inputs BEFORE projection. Return the original array on no-op."""
    check_arm(mode, cap, policy)
    if (len(base) == 0 or len(base) > cap or np.any(np.diff(base) <= 0) or base[0] < 0
            or base[-1] >= len(prior) or not 0 <= ready < opt.B
            or seen.shape != prior.shape or dirty.shape != prior.shape):
        raise ValueError('invalid base plan or pre-call state')
    if (policy in ('baseline', 'sham') or not ready or not np.any(prior[base] == 0)
            or (mode == 'ibs' and len(base) == cap) or (mode == 'obs' and len(base) != cap)):
        return base, Edit()
    if not (seen[:ready].all() and prior[:ready].all() and not dirty[:ready].any()):
        raise ValueError('prefix frontier is not stable and initialized')
    if np.any(~np.isin(np.flatnonzero(dirty), base)):
        raise ValueError('MASK tick bypasses mandatory dirty replay')
    prefix = np.setdiff1d(np.arange(min(ready, int(base[-1]))), base, assume_unique=True)
    if not prefix.size:
        return base, Edit()
    add, drop = int(prefix[-1]), None
    keep = base
    if mode == 'obs':
        # Rightmost clean resolved suffix row except the visible endpoint.
        eligible = base[(base >= ready) & (base != base[-1]) & (prior[base] != 0) & seen[base] & ~dirty[base]]
        if not eligible.size:
            return base, Edit()
        drop = int(eligible[-1])
        keep = base[base != drop]
    actual = np.sort(np.append(keep, add))
    if actual[-1] != base[-1] or not np.array_equal(actual[prior[actual] == 0], base[prior[base] == 0]):
        raise RuntimeError('visible span or ordered MASK-to-logit mapping changed')
    return actual, Edit(add, drop)


class RequestHook:
    def __init__(self, bank, mode, cap, policy):
        self.bank, self.mode, self.cap, self.policy = bank, mode, cap, policy
        self.log = []

    def controller(self, controls, mode, c, start, previous):
        self.start = start
        self.ctl, width = opt.controller(controls, mode, c, start, previous)
        return self.ctl, width

    def limit(self, candidate, prior, dirty, offer):
        if offer != self.cap:
            raise ValueError('expected a fixed decoder cap')
        base = opt.limited_rows(candidate, prior, dirty, offer)
        ready = int(self.ctl.cached_prefix_len)
        rows, edit = choose(base, mode=self.mode, cap=offer, policy=self.policy, ready=ready,
                            prior=prior, dirty=dirty, seen=self.bank.initialized[self.start:self.start+len(prior)])
        self.log.append(dict(ready_before=ready, base_q=len(base), upstream_proposed_q=len(candidate),
                             refresh=None if edit.add is None else self.start + edit.add,
                             donor=None if edit.drop is None else self.start + edit.drop))
        return rows

    def bind(self):
        # Only two global lookups in the private call are rebound. Reuse the
        # pinned code object itself; no source rewriting, frame inspection, or
        # process-global replacement. Refuse to adapt an unreviewed version.
        native = opt._generate
        if sha256(Path(opt.__file__).read_bytes()).hexdigest() != PIN:
            raise RuntimeError('pinned Optimus source changed; review hook before reuse')
        scope = dict(native.__globals__, controller=self.controller, limited_rows=self.limit)
        return FunctionType(native.__code__, scope, native.__name__, native.__defaults__, native.__closure__)


@torch.inference_mode()
def generate_request(model, data, controls, unmask, *, mode, cap, policy='baseline'):
    """Return a separately labelled request, physical ledger and per-tick edits.

    Never seeds the caller's RNG. Equal seeds give equal initialization; the
    intervention uses no randomness. Later admissions/RNG consumption can diverge.
    """
    check_arm(mode, cap, policy)
    if model.training or model.config.canvas_length != opt.B:
        raise ValueError('requires eval DiffusionGemma B256')
    log = []
    with opt.observe(model) as events:
        if policy == 'baseline':
            ids, trace = opt.generate(model, data, controls, unmask, mode, cap, caps=(cap,))
        else:
            bank = opt.Resident()
            bank.bypass = False
            hook = RequestHook(bank, mode, cap, policy)
            generate = hook.bind()
            with opt.resident_attention(model, bank):
                ids, trace = generate(bank, model, data, controls, unmask, mode, cap, False, (cap,))
            log = hook.log
            for s, e in zip(trace, log, strict=True):
                added, dropped = e['refresh'], e['donor']
                extra = int(added is not None and dropped is None)
                if (s['q'] != e['base_q'] + extra or s['proposed_q'] != e['upstream_proposed_q'] or
                        (added is not None and (added not in s['positions'] or added in s['admitted'])) or
                        dropped in s['positions']):
                    raise RuntimeError('edited physical trace mismatch')
                # Preserve original upstream proposal in log; for the physical
                # plan, an IBS addition is one extra proposed AND projected row.
                s['proposed_q'] += extra
    enc, dec = ([q for t, q in events[1:] if t == tag] for tag in ('encoder', 'decoder'))
    if events[0] != ('encoder', data['input_ids'].shape[1]) or dec != [s['q'] for s in trace]:
        raise RuntimeError('observed prefill or physical decoder ledger mismatch')
    ledger = dict(generation_forwards=len(enc)+len(dec), generation_query_rows=sum(enc)+sum(dec),
                  generation_physical_padded_rows=sum(enc)+sum(dec), max_q=max(dec+enc), max_decoder_q=max(dec))
    # The canonical validator expects a base cap label. Never export an edited
    # run under that label or write to the archive.
    validate_trace(dict(policy=f'{mode}_cap_{cap}', events=events, trace=trace, **ledger))
    return dict(policy=f'{mode}_cap_{cap}' + ('' if policy == 'baseline' else '_' + policy),
                output_ids=ids, trace=trace, edits=log, events=events, decoder_F=len(dec), decoder_Q=sum(dec),
                encoder_gen_F=len(enc), encoder_gen_Q=sum(enc), prefill_Q=events[0][1], **ledger)


def tiny_inputs():
    """Real transformer, real controller/unmasker, random weights. CPU only."""
    from transformers import DiffusionGemmaConfig, DiffusionGemmaForBlockDiffusion

    text = dict(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                num_experts=4, top_k_experts=2, moe_intermediate_size=8, num_attention_heads=2,
                num_key_value_heads=1, num_global_key_value_heads=1, head_dim=8, global_head_dim=8,
                sliding_window=12, layer_types=['sliding_attention', 'full_attention'])
    cfg = DiffusionGemmaConfig(canvas_length=opt.B, text_config=text, vision_config=dict(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, head_dim=8))
    cfg._attn_implementation = 'sdpa'
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.default_generator.manual_seed(31)
        model = DiffusionGemmaForBlockDiffusion(cfg).eval()
    # Retire a full B256 canvas, then stop. This artificial EOS set is only
    # for bounding the CPU mechanism test; delivered length is not a metric.
    model.generation_config.eos_token_id = list(range(text['vocab_size']))
    controls, strategy, _ = opt.load_controls()
    unmask = controls.UnmaskingProcessor(SimpleNamespace(confidence_threshold=.9, block_length=opt.B,
        denoising_steps=opt.B, unmasking_strategy=strategy.LOW_CONFIDENCE_DYNAMIC))
    prompt = torch.tensor([[2, 8, 9]], device='cpu')
    return model, dict(input_ids=prompt, attention_mask=torch.ones_like(prompt)), controls, unmask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=ARM, required=True)
    parser.add_argument('--cap', choices=FIXED, type=int, default=256)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    torch.set_num_threads(1)
    inputs = tiny_inputs()
    for policy in ('baseline', 'sham', ARM[args.mode]):
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(args.seed)
            row = generate_request(*inputs, mode=args.mode, cap=args.cap, policy=policy)
        print(json.dumps(dict(evidence='tiny_cpu_mechanism_only', seed=args.seed,
                              refresh_ticks=sum(e['refresh'] is not None for e in row['edits']),
                              **{k: row[k] for k in ('policy','decoder_Q','decoder_F','encoder_gen_Q','prefill_Q')})))


if __name__ == '__main__':
    main()
