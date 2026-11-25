"""Attention-only intervention on nominal-window OBS; never clips query rows."""
import torch


def native_visible(prior, query_positions):
    # SelectedResidentCache returns a contiguous KV view starting at absolute 0.
    keys = torch.arange(prior.shape[-1], device=prior.device)
    return prior & ((query_positions[:, None, :, None] // 32) >= (keys // 32))


class BoundaryModel:
    def __init__(self, model):
        self.model = model

    def __call__(self, *args, **kw):
        # Prefill already uses this exact block mask, so intersection is identity.
        kw["attention_mask"] = native_visible(kw["attention_mask"], kw["position_ids"])
        return self.model(*args, **kw)
