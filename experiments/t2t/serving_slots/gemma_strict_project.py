"""Project only the selected rows of a B256 Gemma canvas, including cold start."""
from contextlib import ExitStack

import torch

from .gemma_wave_kv import WaveKV, wave_attention


class CanvasProjector:
    """Single request, single canvas; caller owns token, sampler and encoder state."""

    def __init__(self, model, encoder_cache, *, origin, ordinal=0):
        if model.training or model.config.canvas_length != 256 or type(origin) is not int or origin < 1:
            raise ValueError("Expected an eval-mode B256 model and a positive prompt length")
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("Invalid canvas ordinal")
        self.model, self.cache, self.origin = model, encoder_cache, origin
        self.base = origin + 256 * ordinal
        if encoder_cache.get_seq_length() != self.base:
            raise ValueError("Encoder prefix must reach the start of the active canvas")
        self.bank = WaveKV(256)
        self.masks = dict.fromkeys(model.config.text_config.layer_types)
        self.active = False

    def __enter__(self):
        if self.active:
            raise RuntimeError("Already active")
        self.context = wave_attention(self.model, self.bank)
        self.context.__enter__()
        self.active = True
        return self

    def __exit__(self, *exc):
        self.active = False
        return self.context.__exit__(*exc)

    @torch.inference_mode()
    def project(self, ids, rows, offer, *, conditioning=None, sc_valid=None):
        """Return fresh [1, q, vocab] logits. No sampler, timer, or token write."""
        rows = list(rows)
        if (not self.active or type(offer) is not int or offer not in (32,64,128,256)
                or not 0 < len(rows) <= offer or any(type(i) is not int or not 0 <= i < 256 for i in rows)
                or rows != sorted(set(rows)) or ids.shape != (1,256) or ids.dtype != torch.long):
            raise ValueError("Expected a live decoder, a supported physical cap and sorted unique rows")
        q = len(rows)
        if self.cache.get_seq_length() != self.base:
            raise RuntimeError("Encoder cache changed under an active canvas")
        if conditioning is not None:
            if (conditioning.ndim != 3 or conditioning.shape[:2] != (1,q)
                    or conditioning.device != ids.device or conditioning.shape[2] !=
                    self.model.model.decoder.embed_tokens.weight.shape[0]):
                raise ValueError("Self-conditioning must contain exactly the selected rows")
            if sc_valid is not None and (len(sc_valid) != q or any(type(b) is not bool for b in sc_valid)):
                raise ValueError("Invalid row self-conditioning validity")
        elif sc_valid is not None:
            raise ValueError("No conditioning tensor to filter")
        pos = [self.base+i for i in rows]
        idx = torch.tensor(rows,device=ids.device,dtype=torch.long)
        self.bank.begin(pos,self.origin,ids.device,causal=True)
        with ExitStack() as stack:
            if sc_valid is not None and not any(sc_valid):
                conditioning = None
            elif sc_valid is not None and not all(sc_valid):
                valid = torch.tensor(sc_valid,device=ids.device)[None,:,None]
                hook = self.model.model.decoder.self_conditioning.register_forward_pre_hook(
                    lambda _,args: (args[0],args[1].masked_fill(~valid,0)))
                stack.callback(hook.remove)
            measured = self.model(decoder_input_ids=ids.index_select(1,idx),
                decoder_position_ids=torch.tensor([pos],device=ids.device,dtype=torch.int32),
                self_conditioning_logits=conditioning,decoder_attention_mask=self.masks,
                past_key_values=self.cache).logits
        if (measured.shape[:2] != (1,q)
                or len(self.bank.visited) != len(self.model.model.decoder.layers)):
            raise RuntimeError("Returned logits or physical decoder layers disagree with selected Q")
        return measured
