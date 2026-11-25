"""SDAR reference sampling filters; keep ties at the kth score."""
import torch


def top_k_logits(logits, k):
    if k <= 0:
        return logits
    cutoff = logits.topk(k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < cutoff, -torch.inf)


def top_p_logits(logits, p):
    ordered, indices = logits.sort(dim=-1, descending=True)
    above = ordered.softmax(-1).cumsum(-1) > p
    drop = torch.zeros_like(above)
    drop[..., 1:] = above[..., :-1]
    return logits.masked_fill(torch.zeros_like(drop).scatter_(-1, indices, drop), -torch.inf)
