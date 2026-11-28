"""Fixed-Q256 serial research endpoint: publish on a real native stop."""
from threading import current_thread

import torch

from . import gemma_strict_sampler
from .gemma_continuous_rng import sampler_class
from .gemma_fixed_dual import AblationRequest, B, Canvas, FULL, Law, StrictRows, WaveKV
from .gemma_native_visibility import representation
from .gemma_predicate_only import native_canvas


NativeCanvas = native_canvas(Canvas)


class _CaptureStop:
    """Capture the exact argmax tensor passed to the actual native criterion."""

    def __init__(self, criterion):
        self.criterion = criterion
        self.draft = None

    def reset(self):
        self.draft = None
        self.criterion.reset()

    def __call__(self, draft, scores):
        hit = self.criterion(draft, scores)
        self.draft = tuple(draft[0].tolist()) if bool(hit.item()) else None
        return hit


def _digest_tensor(tensor):
    from hashlib import sha256
    return sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


class NativeArgmaxRequest(AblationRequest):
    """Batch one. One full decoder forward per step; a hit may also encode.

    ``paid_verify=True`` is the paired control, not the native endpoint.
    """

    def __init__(self, model, data, *, seed, max_new_tokens=2048, paid_verify=False):
        if type(paid_verify) is not bool:
            raise TypeError("Expected an explicit paid-verify flag")
        self.paid_verify = paid_verify
        self._owner = current_thread()
        self.canvas_inits = []
        self._sampler_type = sampler_class(gemma_strict_sampler, seed, True,
                                           self.canvas_inits, _digest_tensor)
        super().__init__(model, data, q=B, seed=seed, max_new_tokens=max_new_tokens,
                         law=Law(1, 8), top_gate="off", fullstop=True,
                         dense_stop=True, fuse_first_verify=True)
        self.capture = _CaptureStop(self.stop_observer.criterion)
        self.stop_observer.criterion = self.capture

    def _start_block(self):
        weight = self.model.model.decoder.embed_tokens.weight
        self.rows = StrictRows(self._sampler_type.steps, seed=self.seed, ordinal=self.ordinal)
        self.sampler = self._sampler_type(cap=B, vocab_size=weight.shape[0],
                    seed=(self.seed + self.ordinal) % (1 << 64), device=self.ids.device,
                    embedding_dtype=weight.dtype)
        self.bank, self.lane, self.cursor = WaveKV(B), None, 0
        if self.cache.is_compileable:
            raise ValueError("Native visibility requires a dynamic encoder cache")

    def step(self, q=B):
        if current_thread() is not self._owner:
            raise RuntimeError("Use the owning serial inference lane")
        if type(q) is not int or q != B:
            raise ValueError("Native argmax requires exactly Q256")
        return super().step(q)

    def result(self):
        if self.status != "done":
            raise RuntimeError(f"Request is {self.status}")
        return tuple(self.tokens)

    def _advance(self, q):
        s, rows = self.sampler, self.rows
        head = NativeCanvas(self.ordinal, rows, s)
        base = self.origin + B * self.ordinal
        pos = tuple(range(base, base + B))
        dirty = set(rows.dirty)
        if (q != B or self.tail is not None or self.rebuild or self.cursor or
                s.phase != "sample" or not dirty <= rows.initialized or
                not {base+i for i in dirty}.issubset(self.bank.live)):
            raise RuntimeError("Not a fresh solo-head transaction")
        sample = head.can_sample
        if not sample and not (self.paid_verify and rows._native_stop):
            raise RuntimeError("Full-head endpoint requires an actual native hit")
        route = "dense_single" if sample else "verify_paid" if dirty else "verify"
        exposed = []
        with representation(self, True, exposed):
            event = self._apply([head], pos, set(pos) if sample else set(), (), (), q, route)
        part, = event["parts"]
        if (len(exposed) != len(self.model.model.decoder.layers)
                or part["rows"] != list(range(B)) or len(part["sampled"]) != (B if sample else 0)
                or part["dirty_before"] != sorted(dirty) or not set(part["dirty_before"]) <= FULL):
            raise RuntimeError("Incomplete Q256 projection or prior writeback")
        event.update(dirty_before_paid=sorted(dirty), native_draft=None,
                     unread_speculative_rows=[], delivered=None, stop_reason=None)
        if not sample:
            if rows.dirty:
                raise RuntimeError("Unpaid strict verification")
            event["delivered"] = tuple(self.tokens[B * head.ordinal:])
            event["stop_reason"] = "paid_verify"
        elif event["fullstop"]["hit"]:
            block = self.capture.draft
            if block is None or len(block) != B or not rows._native_stop:
                raise RuntimeError("Missing same-turn criterion argmax")
            event["native_draft"] = block
            self.capture.draft = None
            if not self.paid_verify:
                # Earlier writes were read above. This *new* post-sample
                # speculative canvas is never read again, including by E.
                event["unread_speculative_rows"] = sorted(rows.dirty)
                delivered, encoded = self._deliver_native(block)
                event.update(delivered=delivered, stop_reason="native_hit", encoded=encoded,
                             encoder_F=int(encoded), encoder_Q=B * int(encoded))
        elif s.age >= s.steps:
            raise RuntimeError("Clock limit is not a native criterion hit")
        event.update(generation_F=1 + event["encoder_F"],
                     physical_generation_Q=B + event["encoder_Q"])
        return event

    def _deliver_native(self, block):
        if len(block) != B or len(self.tokens) != B * self.ordinal:
            raise RuntimeError("Not at the complete head delivery boundary")
        end = next((i + 1 for i, value in enumerate(block) if value in self.eos), None)
        delivered = block if end is None else block[:end]
        self.tokens.extend(delivered)
        if end is not None or len(self.tokens) == self.max_new_tokens:
            self.status = "done"
            self.lane = self.bank = self.rows = self.sampler = None
            return delivered, False
        self.ids = torch.cat((self.ids, self.ids.new_tensor([block])), dim=1)
        self.mask = torch.nn.functional.pad(self.mask, (0, B), value=True)
        self._encode(B, first=False)
        self.ordinal += 1
        self._start_block()
        return delivered, True


def open_native_argmax_gemma(model, data, *, seed, max_new_tokens=2048):
    return NativeArgmaxRequest(model, data, seed=seed, max_new_tokens=max_new_tokens)
