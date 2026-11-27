"""Single-variable low-Q probes layered over the pinned LLaDA request."""
from bdlm_serving.methods.t2t.budget_family import Law
from bdlm_serving.methods.t2t.llada_budget_request import B, LLaDABudgetRequest


class LowStartRequest(LLaDABudgetRequest):
    def __init__(self, *args, arm, law=Law(0, 8), **kwargs):
        if arm not in ("eager", "resident", "credit") or law != Law(0, 8):
            raise ValueError(arm)
        super().__init__(*args, law=Law(0, 8), **kwargs)
        self.arm, self.building, self.credit, self.credit_head = arm, False, 0, None

    def _step(self, q):
        if q not in (4, 8, 32) or (q == 32 and self.arm == "credit"):
            raise ValueError("Fixed single-block diagnostic only")
        end = min(self.head+B, self.end)
        missing = end - self.bank.last_end
        if self.arm == "resident" and (self.building or missing > q):
            if self.dirty or self.stopping or missing <= 0:
                raise RuntimeError("Build must precede the first write of this block")
            rows = tuple(range(self.bank.last_end, min(end, self.bank.last_end+q)))
            e, _ = self._project(rows, q, "build")
            self.building = self.bank.last_end < end
            self._stall += 1
            e.update(tick=len(self.records), offer=q, q=len(rows), rows=list(rows), dirty_before=[],
                     dirty=[], cold=list(rows), mask=[], prefix=[], sampled=[], drain_only=True,
                     admitted=[], edited=[], revoked=[], written_absolute=[], dirty_after=[],
                     sample_clock=self.sample_clock, retired_blocks=[])
            self.records.append(e)
            return e

        credit_in = self.credit
        room = max(0, q-len(self.dirty))
        if self.arm == "credit":
            if self.credit_head != self.head:
                self.credit_head, self.credit = self.head, 0
            credit_in = self.credit
            masks, prefix, _ = self._inputs(q)
            eligible = bool(masks and not masks-self.bank.initialized and prefix and room >= 2)
            if eligible:
                self.credit += min(room, B)
            # At Q4/Q8 the 1/8 token bucket earns at most one row per warm call.
            self.law = Law(1, room) if eligible and self.credit >= 8 else Law(0, 8)
        e = super()._step(q)
        if self.arm == "credit":
            self.credit -= 8*len(e["prefix"])
            if not 0 <= self.credit < 8:
                raise RuntimeError("Bad one-seat credit ledger")
            e.update(credit_in=credit_in, credit_out=self.credit)
        if self.arm == "resident" and e["sampled"] and e["visible_kv"] < end:
            raise RuntimeError("Sampler read a logit without all physical block keys")
        return e
