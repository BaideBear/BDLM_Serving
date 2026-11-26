"""Resident, block-causal KV for DiffusionGemma sparse/double requests.

The decoder of ``DiffusionGemmaForBlockDiffusion`` reads an encoder KV cache
and appends the K/V of the current canvas Q rows.  A wave scheduler keeps more
than one canvas resident, projects only selected Q rows, and needs the live
resident rows to remain visible in sorted absolute order.  This module provides
that bank and a request-local attention wrapper.

Contract
--------
``WaveKV(block_size)`` owns K/V for generated rows only.  Prompt rows live in
the native encoder cache and are passed through untouched on every layer.

``bank.begin(positions, canvas_origin, device, causal=True)`` declares the
selected Q rows for the next decoder forward.  ``positions`` are absolute,
sorted, unique, and include the prompt offset; they must match
``decoder_position_ids`` exactly.  ``canvas_origin`` is the absolute first
generated token of the logical request (the stable request origin, not the
moving head).  Block membership is ``(pos - canvas_origin) // block_size``.

``causal=True`` builds a block-causal mask: a query in block ``i`` sees keys
in blocks ``<= i``; same-block attention is fully bidirectional.  In particular
a head canvas cannot read a later tail canvas, while a tail canvas can read
the head.  ``causal=False`` opens every live row, which is the A/B
bidirectional case.  The prefix is always visible subject to the native mask.

``bank.drop_before(absolute_position)`` removes committed live rows and
compacts storage.  Call it between forwards, then call ``begin`` again.  The
caller must already have re-encoded committed tokens into the encoder prefix.

``with wave_attention(model, bank):`` patches only the decoder's
``self_attn.config._attn_implementation``.  Encoder attention is never
touched, and every patched config object is restored even if the body raises.

Caveats
-------
* Batch one only.  K/V storage dtype and shape are fixed at first write per
  layer.  Native masks must be boolean or ``None``.
* One decoder forward per ``begin``.  A second visit to the same layer raises.
  The caller can assert ``len(bank.visited) == len(model.model.decoder.layers)``.
* ``bank.bypass = True`` is a passthrough for dense native steps and never
  writes K/V.  Otherwise ``begin`` is required before attention.
* The attention alias is registry-local, but only one ``wave_attention`` may
  be active per process; it is not thread-safe.
* The bank is request-scoped.  Do not reuse one instance across models with
  different decoder layer counts.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import copy

import torch

__all__ = ["WaveKV", "wave_attention"]


_ALIAS = "bdlm_gemma_wave_kv_sdpa"


def _attention_functions():
    """Import lazily so the bank itself never requires Transformers."""
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        ALL_ATTENTION_FUNCTIONS,
    )

    return ALL_ATTENTION_FUNCTIONS


def _as_int_list(values, name):
    if isinstance(values, torch.Tensor):
        if values.dim() != 1:
            raise ValueError(f"{name} must be one-dimensional")
        values = values.detach().cpu().tolist()
    try:
        values = list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of integers") from exc
    out = []
    for value in values:
        if type(value) is not int:
            raise TypeError(f"{name} must contain Python ints, got {type(value).__name__}")
        out.append(value)
    return out


class WaveKV:
    """Resident generated-token K/V with block-causal visibility.

    The bank stores one K/V tensor pair per decoder layer.  Rows are addressed
    by absolute position and returned in sorted absolute order after the
    layer's native prefix.  Only selected Q rows are written on each forward;
    rows not selected stay resident from an earlier bootstrap or step.
    """

    def __init__(self, block_size):
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("block_size must be a positive int")
        self.block_size = block_size
        self.device = None
        self.canvas_origin = None
        self.causal = True
        self.positions = None
        self.live = []
        self.buffers = {}
        self.prefix_lengths = {}
        self._written = {}
        self._layer_count = None
        self._floor = 0
        self._base = None
        self._capacity = 0
        self._live_mask = None
        self.visited = set()
        self.bypass = False

    @property
    def base(self):
        """Absolute position of storage slot zero, or ``None`` before begin."""
        return self._base

    @property
    def capacity(self):
        """Number of absolute slots allocated per layer."""
        return self._capacity

    def begin(self, positions, canvas_origin, device, causal=True):
        """Declare the selected Q rows for the next decoder forward.

        ``positions`` must be sorted, unique, absolute Python ints and must
        match ``decoder_position_ids`` in order.  ``canvas_origin`` is the
        stable absolute first generated token of the request.
        """
        positions = _as_int_list(positions, "positions")
        if not positions:
            raise ValueError("positions must not be empty")
        if any(positions[i] >= positions[i + 1] for i in range(len(positions) - 1)):
            raise ValueError("positions must be sorted and unique")
        if type(canvas_origin) is not int or canvas_origin < 0:
            raise ValueError("canvas_origin must be a non-negative int")
        if type(causal) is not bool:
            raise TypeError("causal must be a bool")
        if device is None:
            raise ValueError("device is required")
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if self.device is not None and self.device != device:
            raise ValueError(f"device changed from {self.device} to {device}")
        if self.canvas_origin is not None and canvas_origin != self.canvas_origin and self.live:
            raise ValueError("canvas_origin must remain stable while live rows exist")
        if any(p < canvas_origin for p in positions):
            raise ValueError("positions must be >= canvas_origin")
        if any(p < self._floor for p in positions):
            raise ValueError(f"positions before drop floor {self._floor} cannot be reused")

        new_live = sorted(set(self.live) | set(positions))
        if any(p < canvas_origin for p in new_live):
            raise ValueError("live positions must be >= canvas_origin")

        # Every layer must be able to supply every resident row that is not
        # being written by this forward.  A missing layer is only legal when
        # all live rows are current.
        layers = range(self._layer_count) if self._layer_count is not None else list(self.buffers)
        for layer in layers:
            missing = set(new_live) - set(positions) - self._written.get(layer, set())
            if missing:
                raise RuntimeError(
                    f"layer {layer} is missing resident K/V for {sorted(missing)}")

        if not self.live:
            # A fresh request may start anywhere at or after the drop floor.
            self.buffers.clear()
            self._written.clear()
            self._base = new_live[0]
            self._capacity = 0
        elif new_live[0] < self._base:
            # A selected row may sit in a gap before the first live row, for
            # example the first uncommitted row after a prefix promotion.
            self._rebase(new_live[0])

        if new_live:
            needed = new_live[-1] - self._base + 1
            if needed > self._capacity:
                self._grow(max(needed, max(16, self._capacity * 2)))

        self.canvas_origin = canvas_origin
        self.causal = causal
        self.device = device
        self.positions = list(positions)
        self.live = new_live
        self.visited = set()
        self._live_mask = self._build_live_mask(positions, device)
        return self

    def drop_before(self, absolute_position):
        """Remove committed rows with ``abs < absolute_position``.

        Committed tokens must already have been re-encoded into the encoder
        prefix.  The bank only owns generated rows; it never rebuilds prefix.
        """
        if type(absolute_position) is not int or absolute_position < 0:
            raise ValueError("absolute_position must be a non-negative int")
        self._floor = max(self._floor, absolute_position)
        keep = [p for p in self.live if p >= absolute_position]
        if len(keep) == len(self.live):
            if self.positions is not None and any(p < absolute_position for p in self.positions):
                self.positions = None
                self._live_mask = None
                self.visited = set()
            return

        removed = set(self.live) - set(keep)
        self.live = keep
        for layer in list(self._written):
            self._written[layer] -= removed
            if not self._written[layer]:
                del self._written[layer]

        if keep:
            if self._base is None:
                self._base = keep[0]
            else:
                shift = keep[0] - self._base
                if shift > 0:
                    for layer, (k_store, v_store) in list(self.buffers.items()):
                        self.buffers[layer] = (
                            k_store[..., shift:, :].clone(),
                            v_store[..., shift:, :].clone(),
                        )
                    self._base = keep[0]
            if self.buffers:
                self._capacity = next(iter(self.buffers.values()))[0].shape[-2]
            else:
                self._capacity = 0
        else:
            self.buffers.clear()
            self._written.clear()
            self._base = absolute_position
            self._capacity = 0

        if self.positions is not None and any(p < absolute_position for p in self.positions):
            self.positions = None
        self._live_mask = None
        self.visited = set()

    def extend(self, layer, q, k, v, mask=None):
        """Write selected K/V and return ``(prefix+live K, V, mask)``.

        This is called by the patched attention function.  ``q`` holds only the
        selected Q rows.  ``k``/``v`` are ``[native prefix, current Q K/V]``.
        The prefix length is inferred per layer, so sliding and full layers
        can have different native cache lengths.
        """
        if self.positions is None or self.device is None:
            raise RuntimeError("WaveKV.begin is required before attention")
        if type(layer) is not int:
            raise TypeError("layer must be an int")
        if self._layer_count is not None and not 0 <= layer < self._layer_count:
            raise ValueError(f"layer {layer} outside the model's decoder layers")
        if not all(isinstance(t, torch.Tensor) for t in (q, k, v)):
            raise TypeError("q, k and v must be tensors")
        if q.dim() != 4 or k.dim() != 4 or v.shape != k.shape:
            raise ValueError("expected 4-D q/k/v with matching k/v shapes")
        if q.shape[0] != 1 or k.shape[0] != 1:
            raise ValueError("WaveKV requires batch one")
        if q.shape[-1] != k.shape[-1]:
            raise ValueError("q and k head_dim disagree")
        if not self.live:
            raise RuntimeError("live bank is empty")
        q_len = q.shape[-2]
        if q_len != len(self.positions):
            raise ValueError(
                f"decoder Q has {q_len} rows but begin() declared {len(self.positions)}")
        prefix_len = k.shape[-2] - q_len
        if prefix_len < 0:
            raise ValueError("k/v is shorter than the current Q")
        if v.shape[-2] != k.shape[-2]:
            raise ValueError("k/v sequence lengths disagree")
        if layer in self.visited:
            raise RuntimeError(f"layer {layer} visited twice in one begin()")

        missing = set(self.live) - set(self.positions) - self._written.get(layer, set())
        if missing:
            raise RuntimeError(f"layer {layer} is missing resident K/V for {sorted(missing)}")

        # Build the mask before writing so an invalid native mask cannot leave
        # a partially updated bank.
        mask_new = self._build_attention_mask(mask, q_len, prefix_len)

        if self._base is None:
            self._base = self.live[0]
        needed = self.live[-1] - self._base + 1
        if needed > self._capacity:
            self._grow(max(needed, max(16, self._capacity * 2)))

        if layer not in self.buffers:
            leading = k.shape[:-2]
            head_dim = k.shape[-1]
            self.buffers[layer] = (
                k.new_zeros((*leading, self._capacity, head_dim)),
                v.new_zeros((*leading, self._capacity, head_dim)),
            )
        else:
            k_store, v_store = self.buffers[layer]
            if k_store.shape[:-2] != k.shape[:-2]:
                raise ValueError(f"layer {layer} K/V leading shape changed")
            if k_store.shape[-1] != k.shape[-1] or v_store.shape[-1] != v.shape[-1]:
                raise ValueError(f"layer {layer} K/V head_dim changed")
            if k_store.dtype != k.dtype or v_store.dtype != v.dtype:
                raise TypeError(f"layer {layer} K/V dtype changed")
            if k_store.device != self.device or v_store.device != self.device:
                raise ValueError(f"layer {layer} K/V device changed")
        self.prefix_lengths[layer] = prefix_len

        k_store, v_store = self.buffers[layer]
        index = torch.as_tensor(
            [p - self._base for p in self.positions], dtype=torch.long, device=self.device
        )
        k_store.index_copy_(-2, index, k[..., -q_len:, :])
        v_store.index_copy_(-2, index, v[..., -q_len:, :])
        self._written.setdefault(layer, set()).update(self.positions)
        self.visited.add(layer)

        live_index = torch.as_tensor(
            [p - self._base for p in self.live], dtype=torch.long, device=self.device
        )
        k_live = k_store.index_select(-2, live_index)
        v_live = v_store.index_select(-2, live_index)
        k_new = torch.cat((k[..., :prefix_len, :], k_live), dim=-2)
        v_new = torch.cat((v[..., :prefix_len, :], v_live), dim=-2)
        return k_new, v_new, mask_new

    def _build_live_mask(self, positions, device):
        q_pos = torch.as_tensor(positions, dtype=torch.long, device=device)
        k_pos = torch.as_tensor(self.live, dtype=torch.long, device=device)
        q_block = (q_pos - self.canvas_origin) // self.block_size
        k_block = (k_pos - self.canvas_origin) // self.block_size
        visible = k_block[None, :] <= q_block[:, None]
        if not self.causal:
            visible = torch.ones_like(visible)
        return visible[None, None, :, :]

    def _build_attention_mask(self, native_mask, q_len, prefix_len):
        if native_mask is None:
            prefix_mask = torch.ones(
                (1, 1, q_len, prefix_len), dtype=torch.bool, device=self.device
            )
        else:
            if not isinstance(native_mask, torch.Tensor):
                raise TypeError("native attention mask must be a tensor or None")
            if native_mask.dtype != torch.bool:
                raise TypeError("only boolean native attention masks are supported")
            if native_mask.device != self.device:
                raise ValueError("native attention mask is on a different device")
            m = native_mask
            if m.dim() == 2:
                m = m[:, None, None, :]
            elif m.dim() == 3:
                m = m[:, None, :, :]
            elif m.dim() != 4:
                raise ValueError("native attention mask must be 2-D, 3-D or 4-D")
            if m.shape[0] != 1:
                raise ValueError("WaveKV requires batch one masks")
            if m.shape[2] not in (1, q_len):
                raise ValueError(f"native mask has {m.shape[2]} query rows, expected {q_len}")
            if m.shape[3] == prefix_len + q_len:
                prefix_mask = m[..., :prefix_len]
            elif m.shape[3] == prefix_len:
                prefix_mask = m
            else:
                raise ValueError(
                    f"native mask width {m.shape[3]} does not match "
                    f"prefix {prefix_len} + Q {q_len}"
                )
            if prefix_mask.shape[2] == 1:
                prefix_mask = prefix_mask.expand(-1, -1, q_len, -1)

        if self._live_mask is None:
            raise RuntimeError("live mask missing; call begin()")
        live_mask = self._live_mask
        if prefix_mask.shape[1] != 1:
            live_mask = live_mask.expand(-1, prefix_mask.shape[1], -1, -1)
        if prefix_mask.shape[0] != 1:
            live_mask = live_mask.expand(prefix_mask.shape[0], -1, -1, -1)
        return torch.cat((prefix_mask, live_mask), dim=-1)

    def _grow(self, new_capacity):
        if new_capacity <= self._capacity:
            return
        for layer, (k_store, v_store) in list(self.buffers.items()):
            new_k = k_store.new_zeros((*k_store.shape[:-2], new_capacity, k_store.shape[-1]))
            new_v = v_store.new_zeros((*v_store.shape[:-2], new_capacity, v_store.shape[-1]))
            if self._capacity:
                new_k[..., : self._capacity, :] = k_store
                new_v[..., : self._capacity, :] = v_store
            self.buffers[layer] = (new_k, new_v)
        self._capacity = new_capacity

    def _rebase(self, new_base):
        """Move storage slot zero backward while preserving existing rows."""
        if self._base is None:
            self._base = new_base
            return
        if new_base >= self._base:
            return
        shift = self._base - new_base
        new_capacity = self._capacity + shift
        for layer, (k_store, v_store) in list(self.buffers.items()):
            new_k = k_store.new_zeros((*k_store.shape[:-2], new_capacity, k_store.shape[-1]))
            new_v = v_store.new_zeros((*v_store.shape[:-2], new_capacity, v_store.shape[-1]))
            if self._capacity:
                new_k[..., shift:, :] = k_store
                new_v[..., shift:, :] = v_store
            self.buffers[layer] = (new_k, new_v)
        self._base = new_base
        self._capacity = new_capacity


@contextmanager
def wave_attention(model, bank):
    """Route decoder attention through ``bank`` for the duration of the block.

    Only ``model.model.decoder.layers[*].self_attn`` is patched.  Encoder
    attention is left untouched.  The original config objects and the registry
    alias are restored on normal exit and on exceptions.
    """
    if not isinstance(bank, WaveKV):
        raise TypeError("bank must be a WaveKV")
    try:
        layers = model.model.decoder.layers
    except AttributeError as exc:
        raise ValueError("model.model.decoder.layers is required") from exc
    impl = _attention_functions()
    if _ALIAS in impl:
        raise RuntimeError("wave attention is already active")
    original = impl["sdpa"]

    def sparse(module, q, k, v, mask, **kwargs):
        if bank.bypass:
            return original(module, q, k, v, mask, **kwargs)
        k_new, v_new, mask_new = bank.extend(module.layer_idx, q, k, v, mask)
        return original(module, q, k_new, v_new, mask_new, **kwargs)

    with ExitStack() as stack:
        impl[_ALIAS] = sparse

        def remove_alias():
            if impl.get(_ALIAS) is sparse:
                del impl[_ALIAS]

        stack.callback(remove_alias)
        for layer in layers:
            attn = layer.self_attn
            config = getattr(attn, "config", None)
            if config is None:
                raise ValueError("decoder attention module has no config")
            if getattr(config, "_attn_implementation", None) != "sdpa":
                raise ValueError("expected pinned SDPA decoder attention")
            stack.callback(setattr, attn, "config", config)
            attn.config = copy(config)
            attn.config._attn_implementation = _ALIAS
        bank._layer_count = len(layers)
        yield bank
