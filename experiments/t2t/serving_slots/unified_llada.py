"""Opt-in LLaDA execution adapter for the unified row planner.

This file only *executes* ``bdlm_serving.methods.t2t.unified_rows.plan_rows``.
It contains no admission rule, no confidence ranking, no threshold override, no
endpoint guard and no two-block trick of its own: the planner decides which rows
are paid, the native LLaDA sampler/write protocol decides which of those rows
become tokens, and every paid call is one forward (``q = len(rows)``).  Extra
forwards are visible, never merged or hidden.

Mapping is deliberately mechanical:

* ``Block`` objects are built from live state only.  ``start`` is the first
  valid generated row of the span (``max(head, origin)``) and ``size`` the valid
  count, so ``next.start == head.start + head.size`` even when the first head is
  physically short; ``masks``/``dirty``/``resident`` are absolute positions
  intersected with that span.
* ``plan_rows`` output is copied into ``Plan``: ``rows`` keeps the planner's row
  set verbatim (sorted for the trace), ``dirty``/``mask``/``prefix`` are the
  same absolute sets intersected with the paid rows, and cold unresolved rows
  stay inside ``Plan.cold`` because the contract says ``.mask`` includes them.
* ``sampled`` and ``cold`` are taken from the planner, never rebuilt: the
  contract is ``sampled == sorted(mask | prefix)`` with no dirty row, and
  ``cold`` a subset of the selected nonresident mask.  ``drain_only`` returns no
  sampled rows at all.  A contract violation is reported with physical
  receipts (q/rows/sampled/cold) instead of being patched up.
* ``open_next`` means the next block was actually admitted, and it is sticky: it
  is fed back as ``opened`` and stays true once true.

Usage (opt-in, never a default)::

    from experiments.t2t.serving_slots.unified_llada import request
    cls = request(base_class)          # cross="quarter" for Q=64, "head" below
"""
import torch

from bdlm_serving.methods.t2t.budget_family import Plan
from bdlm_serving.methods.t2t.llada_budget_request import LLaDABudgetRequest

B = 32


def planner():
    """Resolve the contract lazily so this adapter imports without the planner."""
    from bdlm_serving.methods.t2t.unified_rows import Block, Params, plan_rows
    return Block, Params, plan_rows


def live_blocks(req, opened):
    """Build the head block (and next block when opened) from live request state."""
    Block, Params, _ = planner()
    mask = req.mask_id
    origin, end = req.origin, req.end

    def block(start, stop):
        stop = min(stop, end)
        span = range(start, stop)
        tokens = req.x[0, start:stop].tolist()
        masks = (frozenset() if getattr(req, "stopping", False) else
                 frozenset(p for i, p in enumerate(span) if tokens[i] == mask))
        return Block(start=start, size=stop - start, masks=masks,
                     dirty=frozenset(p for p in span if p in req.dirty),
                     resident=frozenset(p for p in span if p in req.bank.initialized))

    head_block = block(max(req.head, origin), req.head + B)
    if getattr(req, "stopping", False):
        # Closed successor after EOS: suppress only the *fresh* successor. The paid
        # window stays the physical head block, anchored in block coordinates.
        return head_block, None
    nxt_start = head_block.start + head_block.size
    # Ordinary mode: the successor is always handed over, visited or not. Whether it
    # may be admitted is the planner's decision; suppressing it here would hide debt
    # and mask a real contract violation instead of failing closed.
    return head_block, (block(nxt_start, nxt_start + B) if nxt_start < end else None)


def receipt(q, out, extra=None):
    return dict(q=q, rows=sorted(out.rows), sampled=sorted(out.sampled), cold=sorted(out.cold),
                mask=sorted(out.mask), prefix=sorted(out.prefix), dirty=sorted(out.dirty),
                open_next=bool(out.open_next), drain_only=bool(out.drain_only), **(extra or {}))


def execute(req, q, opened, params=None):
    """One paid call: returns (Plan, sampled, phase, cap, open_next)."""
    _, Params, plan_rows = planner()
    params = Params() if params is None else params
    head_block, next_block = live_blocks(req, opened)
    out = plan_rows(q, B, head_block, next_block, opened=opened, params=params)
    rows = tuple(sorted(out.rows))
    paid = set(rows)
    blocks = (head_block,) if next_block is None else (head_block, next_block)
    dirty, initialized = set(req.dirty), set(req.bank.initialized)
    planned = set().union(*(range(b.start, b.start + b.size) for b in blocks))
    if not paid <= planned | dirty:
        raise RuntimeError("Planner paid a row outside the planned blocks and the debt: %r"
                           % receipt(q, out))
    if len(paid) > q:
        raise RuntimeError("Planner exceeded the paid offer: %r" % receipt(q, out))
    want = (set(out.mask) | set(out.prefix)) - dirty
    if list(out.sampled) != sorted(out.sampled) or set(out.sampled) != want or set(out.sampled) & dirty:
        raise RuntimeError("Planner sampled set is not sorted(mask | prefix) minus debt: %r"
                           % receipt(q, out, dict(want=sorted(want))))
    if not set(out.cold) <= set(out.mask) or (set(out.cold) & initialized):
        raise RuntimeError("Planner cold rows are not selected nonresident mask: %r" % receipt(q, out))
    mask = set(out.mask) & paid
    cold = set(out.cold) & mask
    warm = mask - cold - dirty
    if out.drain_only and any(paid & set(out.mask)):
        raise RuntimeError("Drain plan must not sample unresolved rows: %r" % receipt(q, out))
    plan = Plan(rows, tuple(sorted(paid & dirty)), tuple(sorted(cold)),
                tuple(sorted(warm)), tuple(sorted(paid & set(out.prefix))), bool(out.drain_only))
    sampled = () if out.drain_only else tuple(out.sampled)
    if getattr(req, "stopping", False):
        if sampled:
            raise RuntimeError("EOS/tail call must not sample: %r" % receipt(q, out))
        window = set(range(head_block.start, head_block.start + head_block.size))
        outside = sorted(set(req.dirty) - window)
        if outside:
            raise RuntimeError("EOS tail debt outside the physical head window: %r"
                               % receipt(q, out, dict(outside=outside)))
        # In-window debt beyond this offer is legitimate: it drains on later rounds,
        # so the caller sees it as ``set(debt) - set(plan.rows)`` and no error is raised.
    return plan, sampled, ("sample" if sampled else "drain"), q, bool(out.open_next)


def request(base=LLaDABudgetRequest, *, params=None, on_step=None, **forced):
    """Opt-in request class; also the factory for the raw LLaDA request.

    One class serves all five offers (Q4/8/16/32/64) because the base is
    ``switchable``: the planner, not the tier, decides geometry.
    """
    class Unified(base):
        def __init__(self, *a, **k):
            opts = {**k, **forced}
            if opts.get("switchable") is not True:
                raise ValueError("Unified adapter requires switchable=True for all five offers")
            super().__init__(*a, **opts)

        def _unified_guard(self):
            if getattr(self, "_additional_admissions", None) is not None:
                raise ValueError("Unified adapter refuses a base admission hook")
            if not getattr(self, "switchable", False):
                raise ValueError("Unified adapter requires a switchable request")
            for name, expected in (("warm_head_sample", False), ("admission_scope", "global"),
                                   ("close_policy", "unchanged"), ("borrow_at_half", False)):
                if getattr(self, name, expected) != expected:
                    raise ValueError("Unified adapter does not carry a profile knob: %s" % name)

        def step(self, q):
            out = super().step(q)      # audits the record just written, terminal call included
            self._unified_audit()
            return out

        def _unified_audit(self):
            last, recs = getattr(self, "_unified_last", None), getattr(self, "records", None)
            if last is None or not recs:
                return
            e = recs[-1]
            if tuple(e["rows"]) != last[0] or tuple(e["sampled"]) != last[1]:
                raise RuntimeError("Observed rows/sampled differ from the planner decision: %r" % dict(
                    planner=dict(rows=last[0], sampled=last[1]),
                    observed=dict(rows=e["rows"], sampled=e["sampled"])))

        def _plan(self, q):
            self._unified_guard()
            self._unified_audit()
            stopping = bool(getattr(self, "stopping", False))
            head_block, nxt = live_blocks(self, False)
            admitted = getattr(self, "_unified_open_at", None)   # physical successor start
            opened = (not stopping) and nxt is not None and admitted == nxt.start
            plan, sampled, phase, cap, nxt_out = execute(self, q, opened, params)
            if stopping:
                self._unified_open_at = None            # closed: no successor survives EOS
            elif nxt is not None and (opened or nxt_out):
                self._unified_open_at = nxt.start        # remember the admitted boundary
            elif admitted is not None and (nxt is None or nxt.start != admitted):
                self._unified_open_at = None            # head moved past it
            self._unified_last = (tuple(plan.rows), tuple(sampled))
            if on_step is not None:
                on_step(dict(q=q, rows=plan.rows, sampled=sampled, phase=phase,
                             open_next=nxt_out, opened=opened, successor=None if nxt is None else nxt.start))
            return plan, sampled, phase, cap

        _switch_plan = _plan
        _cross_plan = _plan
    return Unified


def factory(params=None, on_step=None, **base_kwargs):
    """Simple factory on the raw request: no per-tier live control is involved."""
    base_kwargs["switchable"] = True
    return request(LLaDABudgetRequest, params=params, on_step=on_step, **base_kwargs)
