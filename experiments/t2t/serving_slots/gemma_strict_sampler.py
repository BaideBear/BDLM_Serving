"""Experimental batch-one, single-B256 sampler. NOT native sparse parity.

Pinned cd401955 / Transformers 5.12.0. Native entropy acceptance sorts B
positions, native stop uses whole-draft stability and mean entropy, and native
multinomial/renoise draw for B rows. gemma_unified._sample needs a previous
full raw/SC cache even to call its sparse path. None of those provides a cold
Q<B native sampler. WaveKV, separately, can begin with just Q visible keys.

project(rows, ids, previous_sc) must physically project exactly these *local*
rows, adding the request's absolute origin for position IDs and WaveKV.begin.
The caller owns encoder cache, attention, and projection accounting. First
visits pass None (native zero soft embedding); revisits pass stored processed
logits in embedding dtype. Schedule cold and warm rows in separate calls.

New protocol: locally rank only measured rows; one reverse clock per sampled
decoder call, RNG private to the block. Revisit accepted positions to permit
T2T and revocation. Off-Q state stays untouched. No adaptive native stop.
After coverage, the caller may freeze the canvas and sweep its argmax once in
chunks. These are extra decoder calls, not denoising clocks. A growing or
refreshed WaveKV makes the result order-dependent even with frozen ids/SC.
Full coverage requires ceil(256/cap) sampled ticks: cap <= 5 cannot complete
this particular 48-tick protocol even without revisits.
No encoder re-encode, EOS policy, quality estimate, or throughput claim here.
"""

from collections.abc import Callable
from typing import NamedTuple

import torch
from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
    EntropyBoundSampler,
    EntropyBoundSamplerConfig,
    LinearTemperatureScheduleLogitsProcessor,
)


Project = Callable[[tuple[int, ...], torch.Tensor, torch.Tensor | None], torch.Tensor]


class Changes(NamedTuple):
    clock: int
    admitted: tuple[int, ...]
    edited: tuple[int, ...]
    revoked: tuple[int, ...]


class SelectedRowSampler:
    block = 256
    steps = 48

    def __init__(self, *, cap, vocab_size, seed, device="cpu", embedding_dtype=torch.bfloat16):
        if type(cap) is not int or not 1 <= cap <= self.block:
            raise ValueError("Expected a decoder cap in [1,256]")
        if type(vocab_size) is not int or vocab_size < 2 or not embedding_dtype.is_floating_point:
            raise ValueError("Invalid decoder vocabulary or embedding dtype")
        self.cap, self.vocab, self.dtype = cap, vocab_size, embedding_dtype
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.ids = torch.randint(vocab_size, (1, self.block), generator=self.rng, device=device)
        self.accepted = torch.zeros_like(self.ids, dtype=torch.bool)
        self.sc = {}
        self.temp = LinearTemperatureScheduleLogitsProcessor(0.4, 0.8, self.steps)
        self.age = self.close_calls = 0
        self.sample_q = self.close_q = 0
        self.phase = "sample"
        self._sealed = {}

    def _project(self, rows, project, closing=False):
        rows = tuple(rows)
        if (not 1 <= len(rows) <= self.cap or any(type(r) is not int or not 0 <= r < self.block
                                                   for r in rows) or tuple(sorted(set(rows))) != rows):
            raise ValueError("Expected sorted unique in-block rows within decoder cap")
        if closing and self._sealed.keys() & set(rows):
            raise ValueError("Repeated terminal row")
        warm = [r in self.sc for r in rows]
        if any(warm) and not all(warm):
            raise ValueError("Separate cold (no SC) and warm decoder calls")
        idx = torch.tensor(rows, device=self.ids.device)
        ids = self.ids.index_select(1, idx)
        sc = torch.stack([self.sc[r] for r in rows], dim=0)[None] if all(warm) else None
        logits = project(rows, ids, sc)
        if (logits.shape != (1, len(rows), self.vocab) or logits.dtype != torch.float32
                or logits.device != ids.device):
            raise ValueError("Expected the actual selected float32 LM-head logits [1,Q,V]")
        return rows, idx, ids, logits

    @torch.inference_mode()
    def step(self, rows, project: Project):
        if self.phase != "sample" or self.age >= self.steps:
            raise RuntimeError("Sampling closed or reverse clock exhausted")
        rows, idx, ids, logits = self._project(rows, project)
        clock = self.steps - self.age
        t = torch.tensor(clock, device=ids.device, dtype=torch.int32)
        scores = self.temp(ids, logits, cur_step=t)
        probs = scores.softmax(dim=-1, dtype=torch.float32)
        sampled = torch.multinomial(probs.reshape(-1, self.vocab), 1, generator=self.rng).view_as(ids)
        sampler = EntropyBoundSampler(EntropyBoundSamplerConfig(0.1), len(rows), self.vocab, self.steps)
        proposed = sampler.accept_canvas(ids, sampled, scores, t)
        accepted = sampler.accepted_token_mask
        noise = torch.randint(self.vocab, ids.shape, generator=self.rng, device=ids.device)
        new_ids = torch.where(accepted, proposed, noise)
        old = self.accepted.index_select(1, idx)

        def positions(mask):
            return tuple(idx[mask[0]].tolist())

        changes = Changes(clock, positions(accepted & ~old), positions(old & accepted & (ids != new_ids)),
                          positions(old & ~accepted))
        self.accepted.index_copy_(1, idx, accepted)
        self.ids.index_copy_(1, idx, new_ids)
        for row, signal in zip(rows, scores[0].to(self.dtype).unbind(), strict=True):
            self.sc[row] = signal.clone()
        self.age += 1
        self.sample_q += len(rows)
        return changes

    def begin_close(self):
        if self.phase != "sample" or len(self.sc) != self.block:
            raise RuntimeError("Must sample every position before closing")
        self.phase = "closing"

    @torch.inference_mode()
    def close_chunk(self, rows, project: Project):
        if self.phase != "closing":
            raise RuntimeError("Terminal sweep not open")
        rows, _, _, logits = self._project(rows, project, closing=True)
        self._sealed.update(zip(rows, logits.argmax(dim=-1)[0].tolist(), strict=True))
        self.close_calls += 1
        self.close_q += len(rows)

    def commit(self):
        """Return the ordered block draft, never sampled ids; apply EOS only after this."""
        if self.phase != "closing" or len(self._sealed) != self.block:
            raise RuntimeError("Incomplete terminal sweep or block already committed")
        self.phase = "done"
        return [self._sealed[r] for r in range(self.block)]
