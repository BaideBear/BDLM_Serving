"""Fixed-budget extension of FamilyRequest. No pretrained loader or scheduler."""
from contextlib import ExitStack
from dataclasses import dataclass

import torch

from bdlm_serving.methods.t2t.budget_family import Law, choose
from experiments.t2t.serving_slots.gemma_budget_family import FamilyRequest
from experiments.t2t.serving_slots.gemma_strict_rows import B, OFFERS, RowPlan, StrictRows
from experiments.t2t.serving_slots.gemma_strict_sampler import SelectedRowSampler
from experiments.t2t.serving_slots.gemma_wave_kv import WaveKV, wave_attention


FULL = frozenset(range(B))
TIERS = (*OFFERS, 2 * B)


def measured_rows(logits, offsets):
    # B x vocabulary can be large. Ordinary contiguous block parts are views.
    if offsets[-1] - offsets[0] + 1 == len(offsets):
        return logits[:, offsets[0]:offsets[-1]+1]
    return logits[:, offsets]


@dataclass
class Canvas:
    ordinal: int
    rows: StrictRows
    sampler: SelectedRowSampler

    @property
    def can_sample(self):
        # Independent of mandatory writes, which are always paid first.
        return self.rows.age < self.rows.max_steps and self.rows.accepted != FULL


class FixedFamilyRequest(FamilyRequest):
    """Fixed offer for the lifetime of the request. step() performs exactly one D.

    Small tiers use FamilyRequest's exact trace and implementation. The Q512
    trace instead has a single physical receipt with absolute positions, and
    block-local actions under `parts`. Only a verified head may commit.
    """
    def __init__(self, model, data, *, q, seed, max_new_tokens=2048, law=Law()):
        if type(q) is not int or q not in TIERS or not isinstance(law, Law):
            raise ValueError('Invalid fixed tier or law')
        self.fixed_q, self.tail, self.rebuild = q, None, False
        super().__init__(model, data, seed=seed, max_new_tokens=max_new_tokens, law=law)

    def _start_block(self):
        if self.tail is None:
            super()._start_block()
        else:
            # StrictRequest._commit already encoded the sealed head. Keep only
            # tentative sampler state. No old generated K/V crosses this epoch.
            if self.tail.ordinal != self.ordinal:
                raise RuntimeError('Nonconsecutive promotion')
            self.rows, self.sampler = self.tail.rows, self.tail.sampler
            self.tail, self.rebuild, self.cursor = None, True, 0
        if self.fixed_q > B:
            self.bank = WaveKV(B)
            self.lane = None

    def _spawn(self):
        ordinal = self.ordinal + 1
        s = SelectedRowSampler(cap=B, vocab_size=self.sampler.vocab,
                seed=(self.seed + ordinal) % (1 << 64), device=self.ids.device,
                embedding_dtype=self.sampler.dtype)
        return Canvas(ordinal, StrictRows(s.steps, seed=self.seed, ordinal=ordinal), s)

    @torch.inference_mode()
    def step(self, q=None):
        if q is not None and (type(q) is not int or q != self.fixed_q):
            raise ValueError('Changing a fixed offer is not allowed')
        if self.fixed_q <= B:
            return super().step(self.fixed_q)
        if self.status != 'ready':
            raise RuntimeError(f'Request is {self.status}')
        try:
            return self._dual_step()
        except BaseException:
            self.status = 'failed'  # No retry of a partially written layer/canvas.
            raise

    def _project(self, states, positions):
        if self.cache.get_seq_length() != self.origin + B * self.ordinal:
            raise RuntimeError('Wrong encoder boundary')
        ids, histories = [], []
        for c in states:
            base = self.origin + B * c.ordinal
            idx = [p - base for p in positions if base <= p < base + B]
            if idx:
                ids.append(c.sampler.ids[:, idx])
                histories.extend(c.sampler.sc.get(i) for i in idx)
        valid = [x is not None for x in histories]
        sc = None
        with ExitStack() as stack:
            stack.enter_context(wave_attention(self.model, self.bank))
            self.bank.begin(positions, self.origin, self.ids.device, causal=True)
            if any(valid):
                zero = None if all(valid) else torch.zeros_like(next(x for x in histories if x is not None))
                sc = torch.stack([x if x is not None else zero for x in histories])[None]
                if not all(valid):
                    mask = torch.tensor(valid, device=self.ids.device)[None, :, None]
                    stack.enter_context(self.model.model.decoder.self_conditioning.register_forward_pre_hook(
                        lambda _, a: (a[0], a[1].masked_fill(~mask, 0))))
            logits = self.model(decoder_input_ids=torch.cat(ids, dim=1),
                decoder_position_ids=torch.tensor([positions], dtype=torch.int32, device=self.ids.device),
                self_conditioning_logits=sc, past_key_values=self.cache,
                decoder_attention_mask=dict.fromkeys(self.model.config.text_config.layer_types)).logits
        if (logits.shape != (1, len(positions), self.sampler.vocab) or
                len(self.bank.visited) != len(self.model.model.decoder.layers)):
            raise RuntimeError('Incomplete physical projection')
        return logits

    def _dual_step(self):
        head = Canvas(self.ordinal, self.rows, self.sampler)
        if self.rows.status == 'incomplete':
            raise RuntimeError('Incomplete head')
        rebuild = self.rebuild
        verify = not rebuild and self.rows.status != 'running'
        if (head.can_sample and not rebuild and self.tail is None and
                self.ordinal + 1 < self.max_new_tokens // B):
            self.tail = self._spawn()
        states = [head] + ([self.tail] if self.tail is not None else [])
        mandatory, masks, cold, prefix = set(), set(), set(), []
        for c in states:
            base, rows = self.origin + c.ordinal * B, c.rows
            due = set(rows.dirty)
            if c is head and verify:
                due.update(FULL)
            if rebuild:
                due.update(rows.initialized)
            mandatory.update(base + i for i in due)
            if c.can_sample and not (verify or rebuild):
                unresolved = FULL - rows.accepted
                masks.update(base + i for i in unresolved)
                cold.update(base + i for i in FULL - rows.initialized)
                end = min(unresolved, default=0)
                # Head first, then tail; within each use the registered right edge.
                prefix.extend(base + i for i in reversed(range(end)) if i in rows.initialized and i not in due)
        limit = min(self.fixed_q, B * len(states))
        plan = choose(limit, B, mandatory, masks, prefix, cold=cold, law=self.law)
        pos = plan.rows
        if not 0 < len(pos) <= limit or not mandatory.issubset(pos):
            raise RuntimeError('Empty/over-cap decoder call or unpaid mandatory debt')
        sample = set() if plan.drain_only or verify or rebuild else set(plan.cold or pos)
        logits = self._project(states, pos)
        event = dict(tick=self.tick, ordinal=self.ordinal, offer=self.fixed_q, q=len(pos),
                     positions=list(pos), parts=[], encoded=False, encoder_Q=0, decoder_F=1,
                     decoder_Q=len(pos),encoder_F=0,discarded=[],promoted=None)
        for c in states:
            base, rows, s = self.origin + c.ordinal * B, c.rows, c.sampler
            offsets = [j for j,p in enumerate(pos) if base <= p < base + B]
            if not offsets:
                continue
            local = tuple(pos[j] - base for j in offsets)
            active = [j for j in offsets if pos[j] in sample and c.can_sample]
            sampled = tuple(pos[j] - base for j in active)
            extra = tuple(i for i in local if base+i in plan.prefix)
            phase = ('verify' if verify and c is head else 'rebuild' if rebuild else
                     'sample' if sampled else 'drain')
            part = dict(ordinal=c.ordinal, phase=phase, rows=list(local), sampled=list(sampled),
                        prefix=list(extra), dirty_before=sorted(rows.dirty),age_before=s.age,
                        admitted=[],edited=[],revoked=[])
            if phase == 'verify':
                if rows.initialized != FULL or rows.dirty or set(local) != FULL:
                    raise RuntimeError('Unready head or incomplete terminal sweep')
                s.begin_close()
                s.close_chunk(local, lambda *_: measured_rows(logits, offsets))
            else:
                rp = RowPlan(self.fixed_q,local,sampled,tuple(i for i in local if base+i in plan.cold),extra)
                rows._pending = rp  # StrictRows.finish validates this exact transaction.
                if sampled:
                    idx = list(sampled)
                    before = s.ids[0,idx]
                    changes = s.consume(idx, measured_rows(logits, active))
                    rows.finish(rp, accepted_after=s.accepted[0,idx].tolist(),
                                input_changed=(before != s.ids[0,idx]).tolist())
                    part.update(admitted=list(changes.admitted),edited=list(changes.edited),
                                revoked=list(changes.revoked))
                else:
                    rows.finish(rp)
            if s.age != rows.age:
                raise RuntimeError('Block sampler clock mismatch')
            part['age_after'] = s.age
            event['parts'].append(part)
        self.rebuild = False
        if verify:
            tail = self.tail
            if self._commit():
                event.update(encoded=True, encoder_F=1, encoder_Q=B,
                             promoted=tail.ordinal if tail is not None else None)
            else:
                event['discarded'] = [tail.ordinal] if tail is not None else []
                self.tail = self.bank = None
        self.tick += 1
        return event
