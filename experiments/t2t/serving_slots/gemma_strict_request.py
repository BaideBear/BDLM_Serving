"""Single-request Gemma selected-row denoising. One actual decoder call per step.

This single-canvas path covers decoder caps 32/64/128/256. Q512 requires the
separate two-canvas path. A paid native encoder call after a nonterminal block
is not a decoder call. Its 256 rows are recorded on the committing step.
"""
import torch

from .gemma_strict_project import CanvasProjector
from .gemma_strict_rows import B, OFFERS, StrictRows
from .gemma_strict_sampler import SelectedRowSampler


def _valid_offer(q):
    return type(q) is int and q in OFFERS


class StrictRequest:
    """Batch one, no policy for choosing the supplied physical budget."""

    @torch.inference_mode()
    def __init__(self, model, data, *, seed, max_new_tokens=2048, prefix_quota=0, prefix_mode="right"):
        if (type(seed) is not int or type(max_new_tokens) is not int or max_new_tokens < B
                or max_new_tokens % B or type(prefix_quota) is not int or not 0 <= prefix_quota <= B
                or prefix_mode not in ("right", "oldest", "random", "zero")
                or (prefix_mode == "zero" and prefix_quota)):
            raise ValueError("Invalid B256 request settings or prefix policy")
        ids, mask = data["input_ids"], data["attention_mask"]
        if (model.training or model.config.canvas_length != B or ids.ndim != 2
                or ids.shape[0] != 1 or ids.shape[1] < 1 or ids.dtype != torch.long
                or ids.device != mask.device or mask.shape != ids.shape or not bool((mask == 1).all())
                or set(data) - {"input_ids", "attention_mask", "mm_token_type_ids"}
                or ("mm_token_type_ids" in data and bool(data["mm_token_type_ids"].count_nonzero()))):
            raise ValueError("Expected one unpadded text request on an eval-mode Gemma B256 model")
        self.model, self.ids, self.mask = model, ids, mask.bool()
        self.max_new_tokens, self.origin = max_new_tokens, ids.shape[1]
        self.seed, self.quota, self.mode = seed, prefix_quota, prefix_mode
        self.status, self.tick, self.ordinal = "ready", 0, 0
        self.tokens = []
        cfg, kwargs = model._prepare_generation_config(None, max_new_tokens=max_new_tokens)
        if kwargs or cfg.max_denoising_steps != SelectedRowSampler.steps:
            raise ValueError("Expected the pinned 48-step generation config")
        eos = cfg.eos_token_id
        self.eos = set() if eos is None else {eos} if isinstance(eos, int) else set(eos)
        self.cache = model._prepare_cache_for_generation(cfg, 1, self.origin + max_new_tokens - B)
        self._encode(self.origin, first=True)
        self._start_block()

    def _encode(self, length, *, first):
        pos = torch.arange(self.ids.shape[1]-length, self.ids.shape[1], device=self.ids.device,
                           dtype=torch.int32)[None]
        ids, mask = self.model._prepare_encoder_inputs(self.ids, self.mask, pos, self.cache, first, B, 1)
        if ids.shape != pos.shape:
            raise RuntimeError("Wrong physical encoder input")
        self.cache = self.model.model.encoder(input_ids=ids, attention_mask=mask,
                         past_key_values=self.cache, position_ids=pos).past_key_values
        if self.cache.get_seq_length() != self.ids.shape[1]:
            raise RuntimeError("Encoder cache did not reach the committed boundary")

    def _start_block(self):
        weight = self.model.model.decoder.embed_tokens.weight
        self.rows = StrictRows(SelectedRowSampler.steps, seed=self.seed, ordinal=self.ordinal)
        self.sampler = SelectedRowSampler(cap=B, vocab_size=weight.shape[0],
                    seed=(self.seed+self.ordinal) % (1 << 64), device=self.ids.device,
                    embedding_dtype=weight.dtype)
        self.lane = CanvasProjector(self.model, self.cache, origin=self.origin, ordinal=self.ordinal)
        self.cursor = 0

    def _commit(self):
        block = self.sampler.commit()
        stop = next((i+1 for i, t in enumerate(block) if t in self.eos), None)
        self.tokens.extend(block if stop is None else block[:stop])
        if stop is not None or len(self.tokens) == self.max_new_tokens:
            self.status = "done"
            self.lane = self.sampler = self.rows = None
            return False
        self.ids = torch.cat((self.ids, self.ids.new_tensor([block])), dim=1)
        self.mask = torch.nn.functional.pad(self.mask, (0,B), value=True)
        self._encode(B, first=False)
        self.ordinal += 1
        self._start_block()
        return True

    def _forward(self, rows, q):
        sc = self.sampler.sc
        seen = [r in sc for r in rows]
        signal = None
        if any(seen):
            sample = sc[next(r for r in rows if r in sc)]
            zero = torch.zeros_like(sample) if not all(seen) else None
            signal = torch.stack([sc.get(r, zero) for r in rows])[None]
        with self.lane as lane:
            return lane.project(self.sampler.ids, rows, q, conditioning=signal,
                                sc_valid=seen if signal is not None and not all(seen) else None)

    @torch.inference_mode()
    def step(self, q):
        if self.status != "ready":
            raise RuntimeError(f"Request is {self.status}")
        if not _valid_offer(q):
            raise ValueError("Unsupported single-canvas physical budget")
        try:
            return self._step(q)
        except BaseException:
            # A layer may already have written some KV; such a call is not replayable.
            self.status = "failed"
            raise

    def _step(self, q):
        rows, s = self.rows, self.sampler
        dirty = sorted(rows.dirty)
        record = dict(tick=self.tick, ordinal=self.ordinal, offer=q, prefix=[], sampled=[],
                      admitted=[], edited=[], revoked=[], dirty_before=dirty, encoded=False)
        if rows.status == "incomplete":
            raise RuntimeError("Denoising clock expired before covering the full canvas")
        if rows.status != "running":
            if s.phase == "sample":
                s.begin_close()
            take = tuple(range(self.cursor, min(B, self.cursor+q)))
            s.close_chunk(take, lambda r, ids, sc: self._forward(r, q))
            record.update(rows=list(take), phase="verify", q=len(take), stop_reason=rows.status)
            self.cursor += len(take)
            if self.cursor == B:
                record["encoded"] = self._commit()
        else:
            plan = rows.plan(q, prefix_quota=min(q, self.quota), prefix_mode=self.mode)
            if not set(dirty[:q]).issubset(plan.rows):
                raise RuntimeError("Missing required KV refresh")
            logits = self._forward(plan.rows, q)
            record.update(rows=list(plan.rows), q=len(plan.rows), phase="drain" if plan.drain_only else "sample",
                          prefix=list(plan.prefix), sampled=list(plan.sampled))
            if plan.drain_only:
                rows.finish(plan)
            else:
                idx = list(plan.sampled)
                old_ids = s.ids[0,idx]
                measured = logits if plan.sampled == plan.rows else logits[:,list(plan.sample_indices)]
                changes = s.consume(idx, measured)
                # Debt includes every changed token (even renoise), plus acceptance transitions.
                rows.finish(plan, accepted_after=s.accepted[0,idx].tolist(),
                            input_changed=(old_ids != s.ids[0,idx]).tolist())
                record.update(admitted=list(changes.admitted), edited=list(changes.edited),
                              revoked=list(changes.revoked))
            if rows.age != s.age:
                raise RuntimeError("Sampler and planner clocks diverged")
        self.tick += 1
        return record


def generate(model, data, *, seed, offers, prefix_quota=0, prefix_mode="right", max_new_tokens=2048):
    """Fixed tier or cyclic external-budget replay; record every charged event."""
    offers = tuple(offers)
    if not offers or not all(map(_valid_offer, offers)):
        raise ValueError("Expected a nonempty sequence of supported physical budgets")
    request = StrictRequest(model, data, seed=seed, max_new_tokens=max_new_tokens,
                            prefix_quota=prefix_quota, prefix_mode=prefix_mode)
    trace = []
    while request.status == "ready":
        trace.append(request.step(offers[request.tick % len(offers)]))
    return request.tokens, trace
