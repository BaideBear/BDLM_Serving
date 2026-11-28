"""Research-only KV-h horizon branch for LLaDA head-only Q8.

The selected rows, plan and sampler rule are inherited unchanged from
``LLaDABudgetRequest``.  The only difference is physical visibility: each
forward exposes KV only through its rightmost selected row (Optimus horizon),
while the resident cache keeps its high-water mark for future reuse/rebuild.
The causal mask is built from the horizon; global dirty accounting is the base
executor's.  The main executor and the current qualification scripts are
untouched.
"""
from __future__ import annotations

from types import SimpleNamespace

from bdlm_serving.methods.t2t.llada_budget_request import B, LLaDABudgetRequest
from bdlm_serving.methods.t2t.llada_sparse import _SelectedCache


class KVHorizonRequest(LLaDABudgetRequest):
    """Opt-in visibility branch: KV through the rightmost selected row only."""

    def _project(self, rows, cap, phase):
        if not 0 < len(rows) <= cap or rows != tuple(sorted(set(rows))) or rows[0] < 0:
            raise RuntimeError("Invalid physical Q or positions")
        pos = self.x.new_tensor(rows)
        self.bank.begin(pos, rows)
        state = SimpleNamespace(x=self.x, events=[], block=self.head//B,
                                committed_end=self.head, context_version=self.head//B)
        horizon = rows[-1] + 1
        if phase == "prefill" or not self.bank.buffers:
            # Prefill has no prior resident; the full prompt is the horizon.
            logits = self.engine._execute(
                state, pos, self.bank.visible_end, self.bank, phase,
                "prefill" if phase == "prefill" else "selected", True, budget=cap)
        else:
            # Slice the resident KV view to the horizon; writes still land in
            # the bank's full buffers, so the high-water mark is preserved.
            resident = tuple((key[..., :horizon, :], value[..., :horizon, :])
                             for key, value in self.bank.buffers)
            # ``start`` is the cache length seen by the model; keep it equal to
            # the visible horizon so a gap of zero is byte-identical to the
            # base bank's high-water path.
            cache = _SelectedCache(resident, pos, horizon, horizon)
            logits = self.engine._execute(
                state, pos, horizon, cache, phase,
                "prefill" if phase == "prefill" else "selected", True, budget=cap)
        event, = state.events
        self.bank.initialized.update(rows)
        self.bank.last_end = self.bank.visible_end
        event["kv_horizon"] = horizon
        event["kv_high_water"] = self.bank.visible_end
        event["kv_horizon_gap"] = self.bank.visible_end - horizon
        return event, logits


def request_class():
    """Return the research subclass; the evaluator swaps only this class."""
    return KVHorizonRequest
