"""Fixed-budget extension of FamilyRequest. No pretrained loader or scheduler."""
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import blake2b

import torch

from bdlm_serving.methods.t2t.budget_family import Law, choose
from experiments.t2t.serving_slots.gemma_budget_family import FamilyRequest
from experiments.t2t.serving_slots.gemma_strict_rows import B, OFFERS, RowPlan, StrictRows
from experiments.t2t.serving_slots.gemma_strict_sampler import SelectedRowSampler
from experiments.t2t.serving_slots.gemma_wave_kv import WaveKV, wave_attention


FULL = frozenset(range(B))
TIERS = (*OFFERS, 2 * B)
ABLATIONS = ('00', '10', '11')


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
        return not self.rows._native_stop and self.rows.age < self.rows.max_steps and self.rows.accepted != FULL


class FixedFamilyRequest(FamilyRequest):
    """Fixed offer for the lifetime of the request. step() performs exactly one D.

    Small tiers use FamilyRequest's exact trace and implementation. The Q512
    trace instead has a single physical receipt with absolute positions, and
    block-local actions under `parts`. Only a verified head may commit.
    """
    stop_observer = None
    keep_rejected_noise = None

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
        return self._apply(states, pos, sample, plan.prefix, plan.cold, self.fixed_q,
                           'verify' if verify else 'rebuild' if rebuild else 'sparse')

    def _apply(self, states, pos, sample, prefix, cold, offer, route):
        head = states[0]
        joint_verify = route == 'verify_joint_paid'
        verify, rebuild = route in ('verify', 'verify_paid') or joint_verify, route == 'rebuild'
        if joint_verify:
            from .gemma_joint_verify import validate_plan
            validate_plan(self, states, pos, sample, offer)
        obs = self.stop_observer
        scoped = getattr(self, 'stop_at_single_b_only', False)
        single_top = getattr(self, 'stop_at_single_2b', False)
        eligible_offer = offer==B or (single_top and offer==2*B)
        in_scope = not scoped or (eligible_offer and len(states)==1 and route=='dense_single')
        if obs is not None:
            from .gemma_fullstop import HeadSample, HeadTick
            window_offer = offer if single_top else None
            source = HeadTick(self.tick, head.ordinal, int(self.cache.get_seq_length()), window_offer)
            current = HeadTick(self.tick, self.ordinal, self.origin + B * self.ordinal, window_offer)
            if source != current:
                raise RuntimeError('Wrong fullstop source head or epoch')
            hit = False

            def observe_head(rows, scores):
                nonlocal hit
                hit = obs.observe(current, phase='sample', sample=HeadSample(source, rows, scores))

        logits = self._project(states, pos)
        event = dict(tick=self.tick, ordinal=self.ordinal, offer=offer, q=len(pos), route=route,
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
            extra = tuple(i for i in local if base+i in prefix)
            phase = ('verify' if verify and c is head else 'rebuild' if rebuild else
                     'sample' if sampled else 'drain')
            part = dict(ordinal=c.ordinal, phase=phase, rows=list(local), sampled=list(sampled),
                        prefix=list(extra), dirty_before=sorted(rows.dirty),age_before=s.age,
                        admitted=[],edited=[],revoked=[], input_changed=[])
            if phase == 'verify':
                if route == 'verify_paid' or (joint_verify and rows.dirty):
                    if ((not joint_verify and len(states) != 1) or self.cursor or not rows.dirty or
                            not rows.dirty.issubset(local) or c.can_sample or s.phase != 'sample'):
                        raise RuntimeError('Cannot combine this writeback with the first terminal chunk')
                    # _project has already written these exact inputs at every layer.
                    paid = RowPlan(offer, local, (), (), ())
                    rows._pending = paid
                    rows.finish(paid)
                if (rows.initialized != FULL or rows.dirty or
                        local != tuple(range(self.cursor, self.cursor + len(local)))):
                    raise RuntimeError('Unready head or incomplete terminal sweep')
                if self.cursor == 0:
                    s.begin_close()
                s.close_chunk(local, lambda *_: measured_rows(logits, offsets))
                self.cursor += len(local)
            else:
                rp = RowPlan(offer,local,sampled,tuple(i for i in local if base+i in cold),extra)
                rows._pending = rp  # StrictRows.finish validates this exact transaction.
                if sampled:
                    idx = list(sampled)
                    before = s.ids[0,idx]
                    tap = obs is not None and c is head and in_scope
                    hooks = {'on_scores': observe_head} if tap else {}
                    if self.keep_rejected_noise is not None:
                        from .gemma_rejected_noise import RejectNoise
                        hooks['after_renoise'] = RejectNoise(self.keep_rejected_noise, part)
                    changes = s.consume(idx, measured_rows(logits, active), **hooks)
                    changed = (before != s.ids[0,idx]).tolist()
                    rows.finish(rp, accepted_after=s.accepted[0,idx].tolist(), input_changed=changed,
                                native_stop=bool(tap and hit and self.fullstop))
                    part.update(admitted=list(changes.admitted),edited=list(changes.edited),
                                revoked=list(changes.revoked),
                                input_changed=[i for i,v in zip(idx,changed,strict=True) if v])
                    if c is not head and getattr(self, 'receipt_tail_ids', False):
                        touched = sorted(set(changes.admitted) | set(changes.edited))
                        part['tail_updates'] = list(zip(touched, s.ids[0,touched].tolist()))
                else:
                    rows.finish(rp)
            if s.age != rows.age:
                raise RuntimeError('Block sampler clock mismatch')
            part['age_after'] = s.age
            part['dirty_after'] = sorted(rows.dirty)
            event['parts'].append(part)
        if obs is not None:
            hp = next((p for p in event['parts'] if p['ordinal'] == current.ordinal), None)
            phase = hp['phase'] if hp else 'absent'
            if not in_scope or phase != 'sample':
                obs.observe(current, phase=phase if in_scope else 'absent')
            event['fullstop'] = dict(epoch=current.epoch, hit=hit, latched=bool(hit and self.fullstop))
            if scoped:
                event['fullstop'].update(in_scope=in_scope, resident=len(states))
        if rebuild:
            base = self.origin + B * head.ordinal
            self.rebuild = any(base+i not in self.bank.live for i in head.rows.initialized)
        if verify and self.cursor == B:
            tail = self.tail
            if self._commit():
                event.update(encoded=True, encoder_F=1, encoder_Q=B,
                             promoted=tail.ordinal if tail is not None else None)
            else:
                event['discarded'] = [tail.ordinal] if tail is not None else []
                self.tail = self.bank = None
        self.tick += 1
        return event


class AblationRequest(FixedFamilyRequest):
    """00/10/11 with a common top-tier route. Only encoder epochs replace the bank."""

    top_gate = 'eager'
    full_head_at_b = False
    head_first = False
    late_den = 4
    receipt_tail_ids = False
    prefix_order = 'right'
    reuse_paid_masks = False
    fuse_first_verify = False

    def __init__(self, model, data, *, q, seed, ablation='00', max_new_tokens=2048, law=Law(),
                 top_gate='eager', head_first=False, late_den=4, receipt_tail_ids=False,
                 prefix_order='right', reuse_paid_masks=False, fuse_first_verify=False, fullstop=None,
                 keep_rejected_noise=None, dense_stop=False, stop_at_single_b_only=False,
                 sample_full_rebuild=False):
        if ablation not in ABLATIONS:
            raise ValueError('Unknown Gemma ablation')
        if top_gate not in ('off','eager','half','quarter') or late_den not in (2,4,8):
            raise ValueError('Unknown opening gate')
        if head_first and ablation != '10':
            raise ValueError('Slack-only selection requires a cross-block arm without dense build')
        if (prefix_order not in ('right', 'oldest', 'uniform') or type(reuse_paid_masks) is not bool
                or type(fuse_first_verify) is not bool):
            raise ValueError('Unknown prefix ranking or refresh policy')
        self.top_gate, self.head_first = top_gate, head_first
        self.late_den, self.receipt_tail_ids = late_den, receipt_tail_ids
        self.prefix_order, self.reuse_paid_masks = prefix_order, reuse_paid_masks
        self.fuse_first_verify = fuse_first_verify
        if keep_rejected_noise is not None and (
                type(keep_rejected_noise) is not bool or q not in (32,64,128) or ablation!='00'
                or top_gate!='off' or head_first or prefix_order!='right' or law!=Law(1,8)
                or reuse_paid_masks or fuse_first_verify or fullstop is not None):
            raise ValueError('Rejected-noise experiment requires the fixed head-only 1/8 control')
        self.keep_rejected_noise = keep_rejected_noise
        fixed_promotion = q==2*B and fullstop is None and law==Law()
        switching_promotion = stop_at_single_b_only is True and fullstop is True
        if type(sample_full_rebuild) is not bool or sample_full_rebuild and (
                not (fixed_promotion or switching_promotion) or ablation!='00' or top_gate!='half'
                or head_first or not fuse_first_verify or reuse_paid_masks or prefix_order!='right'):
            raise ValueError('Sampled promotion requires a fused half-gate profile')
        self.sample_full_rebuild = sample_full_rebuild
        self.fullstop = fullstop  # None: legacy; False: shadow; True: apply.
        self.stop_at_single_b_only = stop_at_single_b_only
        if type(stop_at_single_b_only) is not bool or stop_at_single_b_only and (
                dense_stop or fullstop is not True or q not in TIERS or ablation!='00' or top_gate!='half'
                or head_first or reuse_paid_masks or prefix_order!='right' or not fuse_first_verify):
            raise ValueError('The switchable stop requires the single-head profile and half gate at 2B')
        if type(dense_stop) is not bool or dense_stop and (
                q != B or ablation != '00' or top_gate != 'off' or head_first or reuse_paid_masks
                or prefix_order != 'right' or law != Law(1,8) or type(fullstop) is not bool
                or not fuse_first_verify):
            raise ValueError('Dense stop requires the registered fused, head-only B control')
        if dense_stop or stop_at_single_b_only:
            self.full_head_at_b = True
        if fullstop is not None:
            from .gemma_fullstop import FullHeadStop
            if type(fullstop) is not bool or fuse_first_verify and not (dense_stop or stop_at_single_b_only):
                raise ValueError('fullstop requires a boolean arm and separate unfused verify')
            cfg, unused = model._prepare_generation_config(None, max_new_tokens=max_new_tokens)
            if unused or (cfg.stability_threshold, cfg.confidence_threshold) != (1, .005):
                raise ValueError('Unregistered Gemma stopping criterion')
            self.stop_observer = FullHeadStop.from_model(model, cfg)
            if self.stop_observer.criterion is None:
                raise ValueError('Missing native criterion')
        self.ablation = ablation
        self.next_side = 0
        super().__init__(model, data, q=q, seed=seed, max_new_tokens=max_new_tokens, law=law)

    def _start_block(self):
        super()._start_block()
        if self.lane is not None:
            self.bank = self.lane.bank
        elif self.fixed_q <= B:  # Tail promotion in the low-tier legacy base.
            self.bank = WaveKV(B)
        self.lane = None
        self.rebuild = self.rebuild and bool(self.rows.initialized)

    def _no_accepted_eos(self, head):
        if not self.eos or not head.rows.accepted:
            return True
        # Unaccepted rows are re-noised random vocabulary IDs, not EOS decisions.
        ids = head.sampler.ids[0, sorted(head.rows.accepted)].tolist()
        return self.eos.isdisjoint(ids)

    def _clean_prefix(self, c, unresolved, start):
        rows = c.rows
        pool = [i for i in reversed(range(min(unresolved, default=B)))
                if i in rows.initialized and i not in rows.dirty]
        if self.prefix_order == 'oldest':
            pool.sort(key=lambda i: (rows._stamps[i], -i))
        elif self.prefix_order == 'uniform':
            # A reproducible permutation, separate from the model's sampling RNG.
            salt = f'{rows.seed}:{c.ordinal}:{rows.tick}:'.encode('ascii')
            pool.sort(key=lambda i: blake2b(salt + bytes([i]), digest_size=8).digest())
        return [start+i for i in pool]

    @torch.inference_mode()
    def step(self, q=None):
        q = self.fixed_q if q is None else q
        if self.status != 'ready' or type(q) is not int or q not in TIERS:
            raise ValueError('Expected a ready request and a supported offer')
        try:
            return self._advance(q)
        except BaseException:
            self.status = 'failed'
            raise

    def _advance(self, q):
        head = Canvas(self.ordinal, self.rows, self.sampler)
        base = self.origin + B * self.ordinal
        tail_base = base+B
        if head.rows.status == 'incomplete':
            raise RuntimeError('Denoising clock exhausted before covering the head canvas')
        states = [head] + ([self.tail] if self.tail is not None else [])
        debt = {self.origin+B*c.ordinal+i for c in states for i in c.rows.dirty}
        head_masks = B-len(head.rows.accepted)
        head_initialized = len(head.rows.initialized)
        room_for_tail = self.ordinal+1 < self.max_new_tokens//B
        opened = None

        def run(route, positions, sampled=(), prefix=(), cold=()):
            pos = tuple(positions)
            limit = B if route == 'build_sample' else q
            if (not 0 < len(pos) <= limit or pos != tuple(sorted(set(pos))) or
                    not set(sorted(debt)[:q]).issubset(pos) or not set(sampled).issubset(pos)):
                raise RuntimeError('Unpaid writes, invalid positions or physical width')
            e = self._apply(states, pos, set(sampled), prefix, cold, q, route)
            written = {self.origin+B*p['ordinal']+i for p in e['parts']
                       for key in ('input_changed','admitted','revoked') for i in p[key]}
            e.update(opened=opened, head_mask_before=head_masks, head_initialized_before=head_initialized,
                     dirty_before_absolute=sorted(debt),
                     dirty_after_absolute=sorted(self.origin+B*c.ordinal+i for c in states for i in c.rows.dirty),
                     written_absolute=sorted(written),
                     sampled_absolute=[self.origin+B*p['ordinal']+i for p in e['parts'] for i in p['sampled']])
            return e

        if self.rebuild:
            # Promoted sampler and clock survive; decoder KV from the old encoder epoch does not.
            missing = sorted((base+i for i in head.rows.initialized if base+i not in self.bank.live))
            if (getattr(self,'sample_full_rebuild',False) and q==2*B and self.ordinal>0 and
                    self.tail is None and not debt and head.can_sample and head.sampler.phase=='sample'
                    and len(missing)==B and
                    not (room_for_tail and head_masks<=B//2 and self._no_accepted_eos(head))):
                event=run('rebuild_sample',missing,missing)
                if not set(missing).issubset(self.bank.live):
                    raise RuntimeError('Incomplete new-epoch KV at promoted full-head sample')
                self.rebuild=False
                return event
            return run('rebuild', missing[:q])

        def spawn():
            nonlocal opened
            self.tail = self._spawn()
            states.append(self.tail)
            opened = self.tail.ordinal

        denoising = head.sampler.phase == 'sample' and head.can_sample
        # The single-block endpoint may consume a complete head only if ALL
        # outstanding writes fit, including a parked successor after a downshift.
        single_full = self.full_head_at_b and q == B and not any(p >= tail_base for p in debt)
        if denoising and (q == 2*B or single_full):
            open_gate = (self.top_gate == 'eager' or
                         (self.top_gate != 'off' and head_initialized == B and
                          head_masks <= B//(2 if self.top_gate == 'half' else 4)))
            allowed = q == 2*B and room_for_tail and self._no_accepted_eos(head) and (self.tail is not None or open_gate)
            if self.tail is None and allowed:
                spawn()
            # Dense even when the debt fills the offer: the same forward encodes
            # every old write before consuming either canvas's fresh logits.
            live = states if allowed else [head]
            pos = tuple(sorted(set(range(base, base+B*len(live))) | debt))
            sampled = [self.origin+B*c.ordinal+i for c in live if c.can_sample for i in range(B)]
            route = ('dense_joint' if len(live)==2 else 'dense_head_debt' if len(pos)>B else 'dense_single')
            return run(route, pos, sampled)

        if not denoising:
            # Terminal sweep and encoder epoch change are head-only barriers.
            if debt:
                first = range(base, base+min(B,q))
                if (getattr(self,'joint_verify',False) and q==2*B and self.tail is not None
                        and not self.cursor and head_initialized==B and head.sampler.phase=='sample'):
                    return run('verify_joint_paid', sorted(set(first) | debt))
                if (self.fuse_first_verify and self.tail is None and self.cursor == 0 and
                        head_initialized == B and head.sampler.phase == 'sample' and
                        not head.can_sample and debt.issubset(first)):
                    return run('verify_paid', first)
                return run('drain', sorted(debt)[:q])
            return run('verify', range(base+self.cursor, base+min(B,self.cursor+q)))

        late_open = (self.ablation != '00' and self.tail is None and room_for_tail
                     and head_initialized == B and head_masks <= B//self.late_den and self._no_accepted_eos(head))
        if self.head_first:
            # Schedule global debt plus the unchanged head plan, then lend only unused seats.
            # A parked tail never loses its debt; a drain does not advance a sampler clock.
            unresolved = FULL-head.rows.accepted
            prefix = self._clean_prefix(head, unresolved, base)
            plan = choose(q, B, debt, {base+i for i in unresolved}, prefix, law=self.law,
                          cold={base+i for i in FULL-head.rows.initialized}-debt)
            pos = plan.rows
            sample = () if plan.drain_only else tuple(p for p in (plan.cold or pos) if p < tail_base)
            room = q-len(pos)
            if room and not plan.drain_only and (late_open or self.tail is not None and self._no_accepted_eos(head)):
                if self.tail is None:
                    spawn()
                t = self.tail
                if t.can_sample:
                    clean = {i for i in FULL-t.rows.accepted if tail_base+i not in debt}
                    extra = tuple(tail_base+i for i in sorted(clean, key=lambda i:(i in t.rows.initialized,i))[:room])
                    cold = tuple(p for p in extra if p-tail_base not in t.rows.initialized)
                    pos = tuple(sorted((*pos,*extra)))
                    sample = (*sample,*extra)
                    return run('sparse' if sample else 'drain', pos, sample, plan.prefix, (*plan.cold,*cold))
            return run('sparse' if sample else 'drain', pos, sample, plan.prefix, plan.cold)
        if late_open:
            if debt and (self.ablation == '11' or len(debt) >= q):
                return run('drain', sorted(debt)[:q])
            spawn()
            if self.ablation == '11':
                pos = tuple(range(tail_base, tail_base+B))
                return run('build_sample', pos, pos, cold=pos)

        # A tail opened at 512 stays resident after a downshift; 00 parks its
        # discretionary rows, never its outstanding writes.
        active = (states if len(states)==2 and self.ablation != '00' and q < 2*B
                  and self._no_accepted_eos(head) else [head])
        masks, cold, prefix = set(), set(), []
        for c in active:
            if not c.can_sample:
                continue
            start = self.origin+B*c.ordinal
            unresolved = FULL-c.rows.accepted
            masks.update(start+i for i in unresolved)
            cold.update(start+i for i in FULL-c.rows.initialized)
            prefix.extend(self._clean_prefix(c, unresolved, start))
        if len(active)==2:
            # Rank clean seats within each physical canvas, then alternate. Map
            # back to absolute addresses before projecting; debt always comes first.
            def alternate(a, b):
                sides = (b,a) if self.next_side else (a,b)
                return [s[i] for i in range(max(len(a),len(b))) for s in sides if i<len(s)]
            candidates = sorted(masks-debt, key=lambda p:(p not in cold,p))
            rank_masks = alternate([p for p in candidates if p<tail_base],
                                   [p for p in candidates if p>=tail_base])
            rank_prefix = alternate([p for p in prefix if p<tail_base], [p for p in prefix if p>=tail_base])
            addresses = sorted(debt)+rank_masks+rank_prefix
            d, m = len(debt), len(rank_masks)
            unresolved_ranks = [i for i,p in enumerate(addresses[:d+m]) if p in masks]
            plan = choose(q,B,range(d),unresolved_ranks,range(d+m,len(addresses)),law=self.law)
            pos = tuple(sorted(addresses[i] for i in plan.rows))
            pref = tuple(addresses[i] for i in plan.prefix)
            if plan.mask:
                self.next_side = int(addresses[plan.mask[-1]]<tail_base)
            # _project supplies each row's own SC history. Each eligible canvas
            # consumes its measured slice once, including its editable prefix.
            eligible = {c.ordinal for c in active if c.can_sample}
            sample = () if plan.drain_only else tuple(p for p in pos if (p-self.origin)//B in eligible)
            return run('sparse' if sample else 'drain', pos, sample, pref, cold.intersection(pos))
        plan = choose(q, B, debt, masks, prefix, law=self.law, cold=cold)
        eligible = {c.ordinal for c in active if c.can_sample}
        if (self.reuse_paid_masks and self.tail is None and plan.drain_only and
                head_initialized == B and debt and debt == set(plan.rows)):
            sample = tuple(p for p in plan.rows if p in masks)
            if sample:
                return run('resample_paid', plan.rows, sample)
        sample = () if plan.drain_only else tuple(p for p in (plan.cold or plan.rows)
                                                  if (p-self.origin)//B in eligible)
        return run('sparse' if sample else 'drain', plan.rows, sample, plan.prefix, plan.cold)
