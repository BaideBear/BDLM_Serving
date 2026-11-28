"""Replace only the paid Q16 observer's choice of extra positions, request-locally."""
from types import FunctionType

from tools import llada_ibs_eos_q16_agreement as original


def current_observe(ledger, tape, excluded, apply, **kw):
    if not kw["positions"]:
        return ()
    original.require(apply, "current observer expects application")
    original.observe(ledger, tape, excluded, False, **kw)
    t=tape[-1]
    k=len(t["eligible"])
    choices=sorted((-prob,p) for p,v,prob,native in zip(
        t["positions"],t["proposals"],t["probabilities"],t["native"],strict=True)
        if not native and prob>.5 and v not in excluded)
    original.require(k<=len(choices), "ledger eligible outside current reject pool")
    # The stock writer applies a membership mask in physical (not rank) order.
    t["applied"]=tuple(sorted(p for _,p in choices[:k]))
    return t["applied"]


def install(fixed,arm,tape,excluded):
    if arm in ("stock","repeat"):
        return original.install_run(fixed,"stock" if arm=="stock" else "apply",tape,excluded)
    original.require(arm=="current","unknown arm")
    template=original.install_run
    private=FunctionType(template.__code__,dict(template.__globals__,observe=current_observe))
    return private(fixed,"apply",tape,excluded)
