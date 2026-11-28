"""Fixed-offer Gemma executor with one exact later-canvas admission event.

Parent semantics are `dense_stop_joint_verified_eos` / ablation 00.  Canvas 0 is
unchanged.  For each later ordinal, the first ordinary projection unions the full
valid MASK canvas into the parent plan and samples eligible rows in that forward.
The rolling offer stays fixed; this one event may exceed it.  All later ticks use
the unchanged parent planner.  Debt, retirement/promotion encoder, rebuild,
verify/EOS, sampler state and actual per-layer width accounting are preserved.
"""
import json

from bdlm_serving.methods.t2t.serving_profiles import StrictHandle
from .gemma_fixed_dual import AblationRequest, B, TIERS
from .gemma_strict_rows import RowPlan
from .gemma_verified_eos import SwitchableVerifiedEOSRequest, verified_end


def admission_certificate(rows, initialized, dirty, accepted, *, ordinal, nominal_q,
                          parent_rows):
    """Return a fail-closed certificate for the one allowed over-offer event."""
    full_rows = list(range(B))
    eligible = bool(rows) and not (set(full_rows) & set(dirty)) and not (set(full_rows) & set(accepted))
    over_parent = set(full_rows) - set(parent_rows)
    accepted_at_entry = sorted(set(full_rows) & set(accepted))
    explainable = bool(eligible and over_parent and not accepted_at_entry)
    return dict(
        ordinal=ordinal,
        nominal_q=nominal_q,
        physical_q=B,
        valid_rows=B,
        eligible_rows=len(rows),
        parent_rows=list(parent_rows),
        over_parent_rows=sorted(over_parent),
        accepted_at_entry=accepted_at_entry,
        explainable=explainable,
        rows=full_rows,
    )


class FullAdmissionRequest(SwitchableVerifiedEOSRequest):
    """Verified-EOS Live parent with one bounded, certified later admission."""

    fixed_fixture = True

    def __init__(self, model, data, *, seed, initial_offer, early=True,
                 max_new_tokens=2048):
        if type(initial_offer) is not int or initial_offer not in TIERS:
            raise ValueError("Requires a registered fixed Gemma offer")
        if type(early) is not bool:
            raise ValueError("early must be explicitly true or false")
        law = Law(1, 4) if initial_offer == 128 else Law()
        self.nominal_offer = initial_offer
        self.early = early
        self.admission_ordinals = set()
        self.events = []
        AblationRequest.__init__(
            self, model, data, q=initial_offer, seed=seed,
            max_new_tokens=max_new_tokens, ablation='00', law=law,
            top_gate='half', prefix_order='right', fuse_first_verify=True)

    def step(self, q=None):
        if q is not None and q != self.nominal_offer:
            raise ValueError("Full admission uses the registered fixed offer")
        return StrictRequest.step(self, self.nominal_offer)

    def _admission_event(self, nominal_q):
        """Build and commit one full-MASK first projection for a later canvas."""
        rows, sampler = self.rows, self.sampler
        if self.ordinal == 0 or rows.initialized or rows.dirty or self.cursor:
            raise RuntimeError("Admission is only for a fresh later canvas")
        if rows.status != 'running' or sampler.phase != 'sample' or not rows.can_sample:
            raise RuntimeError("New canvas is not model-samplable")
        parent = rows.plan(nominal_q, prefix_quota=min(nominal_q, self.quota),
                           prefix_mode=self.mode)
        if len(parent.rows) == B:
            # Parent already selected the complete canvas; finish that exact plan.
            logits = self._forward(parent.rows, nominal_q)
            idx = list(parent.sampled)
            old_ids = sampler.ids[0, idx]
            measured = (logits if parent.sampled == parent.rows
                        else logits[:, list(parent.sample_indices)])
            changes = sampler.consume(idx, measured)
            changed = (old_ids != sampler.ids[0, idx]).tolist()
            rows.finish(parent,
                        accepted_after=sampler.accepted[0, idx].tolist(),
                        input_changed=changed)
            return dict(tick=self.tick, ordinal=self.ordinal, offer=nominal_q,
                        nominal_q=nominal_q, physical_q=len(parent.rows),
                        rows=list(parent.rows), q=len(parent.rows), prefix=[],
                        sampled=list(parent.sampled), admitted=list(changes.admitted),
                        edited=list(changes.edited), revoked=list(changes.revoked),
                        input_changed=[i for i, flag in zip(idx, changed) if flag],
                        dirty_before=[], encoded=False,
                        full_admission=dict(explainable=False, parent_already_full=True,
                                            ordinal=self.ordinal, nominal_q=nominal_q,
                                            physical_q=len(parent.rows)))
        if nominal_q >= B:
            raise RuntimeError("Full canvas was not selected despite sufficient offer")
        certificate = admission_certificate(
            tuple(parent.rows), rows.initialized, rows.dirty, rows.accepted,
            ordinal=self.ordinal, nominal_q=nominal_q, parent_rows=parent.rows)
        if not certificate['explainable']:
            raise RuntimeError("Admission certificate is not explainable: " + json.dumps(
                certificate, separators=(',', ':'), sort_keys=True))
        plan = RowPlan(nominal_q, tuple(range(B)), tuple(range(B)),
                       tuple(range(B)), ())
        # FamilyRows.plan already staged the parent plan; replace it atomically
        # with the certified full-canvas admission plan before projection.
        rows._pending = plan
        record = dict(tick=self.tick, ordinal=self.ordinal, offer=nominal_q,
                      nominal_q=nominal_q, physical_q=B, prefix=[], sampled=[],
                      admitted=[], edited=[], revoked=[], dirty_before=[],
                      encoded=False, full_admission=certificate)
        logits = self._forward(plan.rows, nominal_q)
        idx = list(plan.sampled)
        old_ids = sampler.ids[0, idx]
        changes = sampler.consume(idx, logits)
        changed = (old_ids != sampler.ids[0, idx]).tolist()
        rows.finish(plan, accepted_after=sampler.accepted[0, idx].tolist(),
                    input_changed=changed)
        record.update(rows=list(plan.rows), q=len(plan.rows),
                      sampled=list(plan.sampled), cold=list(plan.cold),
                      admitted=list(changes.admitted), edited=list(changes.edited),
                      revoked=list(changes.revoked),
                      input_changed=[i for i, flag in zip(idx, changed) if flag])
        if rows.age != sampler.age:
            raise RuntimeError("Sampler and planner clocks diverged")
        self.admission_ordinals.add(self.ordinal)
        return record

    def _ordinary_event(self, nominal_q):
        """Reproduce parent StrictRequest running semantics without admission."""
        rows, sampler = self.rows, self.sampler
        dirty = sorted(rows.dirty)
        plan = rows.plan(nominal_q, prefix_quota=min(nominal_q, self.quota),
                         prefix_mode=self.mode)
        if not set(dirty[:nominal_q]).issubset(plan.rows):
            raise RuntimeError("Missing required KV refresh")
        logits = self._forward(plan.rows, nominal_q)
        record = dict(tick=self.tick, ordinal=self.ordinal, offer=nominal_q,
                      nominal_q=nominal_q, physical_q=len(plan.rows), prefix=[],
                      sampled=[], admitted=[], edited=[], revoked=[],
                      dirty_before=dirty, encoded=False)
        if plan.drain_only:
            rows.finish(plan)
            record.update(rows=list(plan.rows), q=len(plan.rows),
                          phase='drain', sampled=list(plan.sampled))
        else:
            idx = list(plan.sampled)
            old_ids = sampler.ids[0, idx]
            measured = (logits if plan.sampled == plan.rows
                        else logits[:, list(plan.sample_indices)])
            changes = sampler.consume(idx, measured)
            changed = (old_ids != sampler.ids[0, idx]).tolist()
            rows.finish(plan, accepted_after=sampler.accepted[0, idx].tolist(),
                        input_changed=changed)
            record.update(rows=list(plan.rows), q=len(plan.rows),
                          phase='sample', sampled=list(plan.sampled),
                          prefix=list(plan.prefix),
                          admitted=list(changes.admitted),
                          edited=list(changes.edited),
                          revoked=list(changes.revoked),
                          input_changed=[i for i, flag in zip(idx, changed) if flag])
        if rows.age != sampler.age:
            raise RuntimeError("Sampler and planner clocks diverged")
        return record

    def _closing_event(self, nominal_q):
        """Ordered verified chunk; early EOS replaces only the redundant tail."""
        rows, sampler = self.rows, self.sampler
        if sampler.phase == 'sample':
            sampler.begin_close()
        start = self.cursor
        take = tuple(range(start, min(B, start + nominal_q)))
        record = dict(tick=self.tick, ordinal=self.ordinal, offer=nominal_q,
                      nominal_q=nominal_q, physical_q=len(take), prefix=[],
                      sampled=[], admitted=[], edited=[], revoked=[],
                      dirty_before=sorted(rows.dirty), encoded=False)
        logits = self._forward(take, nominal_q)
        sampler.close_chunk(take, lambda r, ids, sc: logits)
        record.update(rows=list(take), q=len(take), phase='verify')
        self.cursor += len(take)
        seal = sampler._sealed
        stop = verified_end(seal, start, self.cursor, self.eos)
        if stop is not None:
            if any(seal[i] in self.eos for i in range(start)):
                raise RuntimeError("Earlier sealed EOS was missed")
            self.tokens.extend(seal[i] for i in range(stop))
            record['verified_eos_end'] = stop
            self.status = 'done'
            self.lane = self.rows = self.sampler = None
            return record
        if self.cursor == B:
            record['encoded'] = self._commit()
        return record

    def _step(self, nominal_q):
        rows = self.rows
        if rows.status == 'incomplete':
            raise RuntimeError("Denoising clock expired before covering full canvas")
        if rows.status != 'running':
            return self._closing_event(nominal_q)
        event = self._admission_event(nominal_q)
        if event is None:
            event = self._ordinary_event(nominal_q)
        self.events.append(event)
        self.tick += 1
        return event


class FullAdmissionHandle:
    """Serial strict handle that records nominal and actual physical widths."""

    def __init__(self, request, offers, laws, block_size):
        if len(offers) != len(laws) or len(set(offers)) != len(offers):
            raise ValueError("Incomplete fixed-offer profile")
        if not all(isinstance(law, Law) for law in laws):
            raise ValueError("Expected budget laws")
        self._request, self._offers, self._laws = request, tuple(offers), tuple(laws)
        self._block_size = block_size
        self.opened_prefill_q = request.origin
        self._status, self._result = 'ready', None

    @property
    def status(self):
        return self._request.status

    @property
    def offers(self):
        return self._offers

    def _lane(self):
        pass

    def result(self):
        if self._request.status != 'done':
            raise RuntimeError(f'Request is {self._request.status}')
        return tuple(self._request.tokens)

    def close(self):
        pass

    def step(self, q):
        self._lane()
        if self.status != 'ready':
            raise RuntimeError(f'Request is {self.status}')
        if type(q) is not int or q not in self._offers:
            raise ValueError(f'Unsupported offer {q}')
        request = self._request
        request.law = self._laws[self._offers.index(q)]
        self._status = 'advancing'
        try:
            event = request.step(q)
            width = event['physical_q']
            nominal = event['nominal_q']
            certificate = event.get('full_admission')
            over_offer = width > nominal
            explainable = bool(certificate and certificate.get('explainable') and
                               certificate.get('physical_q') == width)
            if (event['tick'] != request.tick - 1 or event['ordinal'] != request.ordinal
                    or event['nominal_q'] != q or nominal not in self._offers
                    or width <= 0 or event['decoder_Q'] != width
                    or event.get('history_Q', width) != width
                    or (over_offer and not explainable)
                    or len(event['rows']) != width
                    or len(set(event['rows'])) != width):
                raise RuntimeError('Physical admission ledger mismatch')
            encoder_f = int(event.get('encoded', False))
            encoder_q = self._block_size * encoder_f
            out = dict(event, decoder_F=1, history_Q=width,
                       encoder_F=encoder_f, encoder_Q=encoder_q,
                       generation_F=1 + encoder_f,
                       physical_generation_Q=width + encoder_q)
            if request.status == 'done':
                self._result = tuple(request.tokens)
                self._request = None
            elif request.status != 'ready':
                raise RuntimeError(f'Unexpected request status {request.status}')
            self._status = request.status
            return out
        except BaseException:
            self._status, self._request = 'failed', None
            raise


def open_full_admission(model, data, *, seed, initial_offer, max_new_tokens=2048):
    """Open the separately named fixed-offer full-admission executor."""
    if type(initial_offer) is not int or initial_offer not in TIERS:
        raise ValueError('Invalid fixed Gemma initial offer')
    laws = tuple(Law(1, 4) if offer == 128 else Law() for offer in TIERS)
    request = FullAdmissionRequest(model, data, seed=seed,
                                   initial_offer=initial_offer, early=True,
                                   max_new_tokens=max_new_tokens)
    return FullAdmissionHandle(request, TIERS, laws, B)
