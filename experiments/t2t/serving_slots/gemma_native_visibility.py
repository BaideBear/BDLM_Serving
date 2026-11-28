"""Represent an unrestricted solo-head SDPA mask the same way as upstream."""
from contextlib import contextmanager
from unittest.mock import patch


def eligible(live, origin, head, *, size, native_mask, dynamic, solo, bidirectional):
    return (native_mask is None and dynamic and solo and bidirectional and
            bool(live) and all(origin+size*head <= p < origin+size*(head+1) for p in live))


@contextmanager
def representation(req, enabled, stats):
    # Attach only to the actual request's bank for this advance. No registry
    # or model mutation; encoder forwards and newly spawned banks are untouched.
    bank=req.bank
    old=bank.extend
    dynamic=not req.cache.is_compileable
    bidirectional=all(layer.self_attn.is_causal is False for layer in req.model.model.decoder.layers)
    def extend(layer,q,k,v,mask=None):
        kk,vv,mm=old(layer,q,k,v,mask)
        if eligible(bank.live,bank.canvas_origin,req.ordinal,size=256,native_mask=mask,
                    dynamic=dynamic,solo=req.tail is None and not req.rebuild,bidirectional=bidirectional):
            if not bool(mm.all()):
                raise ValueError("Visibility guard does not match physical mask")
            stats.append(layer)
            if enabled:
                mm=None
        return kk,vv,mm
    with patch.object(bank,"extend",extend):
        yield
