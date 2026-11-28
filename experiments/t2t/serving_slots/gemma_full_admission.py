"""Fixed-offer Gemma executor with one exact later-canvas admission event.

Parent semantics are exactly ``dense_stop_joint_verified_eos`` / ablation 00.
Canvas 0 is unchanged.  For each later ordinal, the parent's first ordinary
projection unions the full valid MASK canvas into the parent plan and samples
the eligible rows in that same forward.  The rolling offer stays fixed; this
one event may exceed it.  All later ticks use the unchanged parent planner.
Debt, retirement/promotion encoder, rebuild, verify/EOS, sampler state and
actual per-layer width accounting are preserved by the parent implementation.
"""
import json

from bdlm_serving.methods.t2t.budget_family import Law
from bdlm_serving.methods.t2t.serving_profiles import StrictHandle
from .gemma_fixed_dual import AblationRequest, B, TIERS
from .gemma_verified_eos import SwitchableVerifiedEOSRequest


def admission_certificate(initialized, dirty, accepted, *, ordinal, nominal_q,
                          parent_rows, full_rows=None):
    """Return a fail-closed certificate for the one allowed over-offer event."""
    full = list(range(B)) if full_rows is None else list(full_rows)
    if full != list(range(len(full))):
        raise ValueError("Admission rows must be a contiguous canvas prefix")
    parent = tuple(parent_rows)
    over_parent = sorted(set(full) - set(parent))
    accepted_at_entry = sorted(set(full) & set(accepted))
    eligible = (not initialized and not dirty and not accepted and
                ordinal > 0 and bool(parent) and set(parent) < set(full))
    explainable = eligible and bool(over_parent)
    return dict(
        ordinal=ordinal,
        nominal_q=nominal_q,
        physical_q=len(full),
        valid_rows=len(full),
        eligible_rows=len(full),
        parent_rows=list(parent),
        over_parent_rows=over_parent,
        accepted_at_entry=accepted_at_entry,
        explainable=bool(explainable),
        rows=full,
    )


class FullAdmissionRequest(SwitchableVerifiedEOSRequest):
    """Verified-EOS Live parent with one bounded, certified later admission."""

    def __init__(self, model, data, *, seed, initial_offer, max_new_tokens=2048):
        if type(initial_offer) is not int or initial_offer not in TIERS:
            raise ValueError("Requires a registered fixed Gemma offer")
        # This is the exact constructor closure used by the registered parent
        # profile.  The admission is added only by the _apply hook below.
        AblationRequest.__init__(
            self, model, data, q=initial_offer, seed=seed,
            max_new_tokens=max_new_tokens, ablation="00",
            law=Law(1, 4) if initial_offer == B // 2 else Law(),
            top_gate="half", prefix_order="right", fuse_first_verify=True,
            fullstop=True, stop_at_single_b_only=True, sample_full_rebuild=True)
        self.joint_verify = True
        self.stop_at_single_2b = True
        self.full_head_at_b = True
        self.nominal_offer = initial_offer
        self.admission_ordinals = set()

    def _first_projection(self, states, pos, sample, cold, route):
        if route != "sparse" or len(states) != 1 or self.tail is not None:
            return None
        canvas = states[0]
        base = self.origin + B * canvas.ordinal
        rows = canvas.rows
        parent = tuple(pos)
        if (self.ordinal == 0 or self.cursor or rows._pending is not None or
                rows.initialized or rows.dirty or rows.accepted or
                canvas.sampler.age or not canvas.can_sample or
                canvas.sampler.phase != "sample" or not parent or
                not set(parent).issubset(range(base, base + B)) or
                set(parent) == set(range(base, base + B)) or not sample):
            return None
        full = tuple(range(base, base + B))
        certificate = admission_certificate(
            rows.initialized, rows.dirty, rows.accepted,
            ordinal=canvas.ordinal, nominal_q=self.nominal_offer,
            parent_rows=[p - base for p in parent], full_rows=full)
        if not certificate["explainable"]:
            raise RuntimeError("Invalid full-admission geometry: " + json.dumps(
                certificate, separators=(",", ":"), sort_keys=True))
        return certificate, full

    def _apply(self, states, pos, sample, prefix, cold, offer, route):
        admission = self._first_projection(states, pos, sample, cold, route)
        if admission is not None:
            certificate, full = admission
            canvas, = states
            if canvas.ordinal in self.admission_ordinals:
                raise RuntimeError("Repeated full admission for one canvas")
            pos = full
            sample = full
            cold = full
            e = super()._apply(states, pos, sample, prefix, cold, offer, route)
            e["full_admission"] = certificate
            self.admission_ordinals.add(canvas.ordinal)
            return e
        return super()._apply(states, pos, sample, prefix, cold, offer, route)


class FullAdmissionHandle(StrictHandle):
    """Registered strict lane, with an explicit fail-closed admission exception."""

    def step(self, q):
        self._lane()
        if self.status != "ready":
            raise RuntimeError(f"Request is {self.status}")
        if type(q) is not int or q not in self.offers:
            raise ValueError(f"Unsupported offer {q}")
        request = self._request
        request.law = self._laws[self.offers.index(q)]
        self._status = "advancing"
        try:
            event = request.step(q)
            width = event["q"]
            certificate = event.get("full_admission")
            over_offer = width > q
            explainable = bool(certificate and certificate.get("explainable") and
                               certificate.get("nominal_q") == q and
                               certificate.get("physical_q") == width)
            if (event["offer"] != q or event["ordinal"] != request.ordinal or
                    not 0 < width <= 2 * B or
                    width != event.get("decoder_Q", width) or
                    event.get("decoder_F", 1) != 1 or
                    event.get("encoder_F", 0) not in (0, 1) or
                    event.get("encoder_Q", 0) != B * event.get("encoder_F", 0) or
                    (over_offer and not explainable)):
                raise RuntimeError("Paid work does not match strict offer")
            out = dict(event, prefix_share=(request.law.num, request.law.den),
                       decoder_F=1, decoder_Q=width, history_Q=width,
                       encoder_F=event.get("encoder_F", 0),
                       encoder_Q=event.get("encoder_Q", 0),
                       generation_F=1 + event.get("encoder_F", 0),
                       physical_generation_Q=width + event.get("encoder_Q", 0))
            if request.status == "done":
                self._result = tuple(request.tokens)
                self._request = None
            elif request.status != "ready":
                raise RuntimeError(f"Unexpected request status: {request.status}")
            self._status = request.status
            return out
        except BaseException:
            self._status, self._request = "failed", None
            raise


def open_full_admission(model, data, *, seed, initial_offer, max_new_tokens=2048):
    """Open the separately named fixed-offer full-admission executor."""
    if type(initial_offer) is not int or initial_offer not in TIERS:
        raise ValueError("Invalid fixed Gemma initial offer")
    laws = tuple(Law(1, 4) if offer == B // 2 else Law() for offer in TIERS)
    request = FullAdmissionRequest(model, data, seed=seed,
                                   initial_offer=initial_offer,
                                   max_new_tokens=max_new_tokens)
    return FullAdmissionHandle(request, TIERS, laws, B)
