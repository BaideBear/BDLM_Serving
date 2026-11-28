"""Budget-neutral production adapter: one paid prefix seat replaces a warm MASK.

The row decision is made before ``Resident.select``, so the decoder sees exactly
``offer`` rows.  The incoming row is an already resolved, resident, non-dirty
generated prefix row.  It is projected for fresh K/V but, because ``prior`` is
resolved, it is not passed to the admission sampler.  All dirty debt, the first
actual unresolved MASK, and the original right edge are preserved.  A changed
token anywhere remains subject to the normal real writeback debt.
"""
import numpy as np

from bdlm_serving.methods.t2t.prefix_seat import choose_prefix_seat


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class PrefixSeat:
    """At most one successful exchange per physical block."""

    schema = "gemma-paid-ibs-q64-prefix-seat-v1"

    def __init__(self):
        self.used_blocks = set()
        self.exchanges = []
        self.records = []

    @property
    def record(self):
        return dict(schema=self.schema, exchanges=self.exchanges,
                    blocks_used=sorted(self.used_blocks))

    def rows(self, *, local, prior, width, dirty, start, offer, ready, bank, tick):
        original = np.asarray(local, dtype=np.int64)
        local = tuple(map(int, local))
        prior = list(map(int, prior))
        block = start // 256
        unresolved = {i for i, value in enumerate(prior[:width]) if value == 0}
        due = {int(i) for i in dirty[start:start + width].nonzero()[0]}
        picked = set(local)
        resident = {i for i in range(width) if bank.initialized[start + i]}
        # Prefix selection is a no-op unless the complete state qualifies.  This
        # also avoids consuming the one-shot opportunity on an ineligible drain.
        if (block in self.used_blocks or len(local) != offer or not local
                or not unresolved or not due <= picked or ready <= 0
                or prior[:ready] != [1] * ready or due & set(range(ready))):
            return original
        exchange = choose_prefix_seat(local, unresolved=unresolved, resident=resident,
                                      due=due, paid_prefix_end=int(ready), offer=offer,
                                      used=block in self.used_blocks)
        if exchange is None:
            return original
        donor, prefix = map(int, exchange)
        require_state(local=local, prior=prior, width=width, dirty=dirty, start=start,
                      ready=ready, bank=bank, donor=donor, prefix=prefix)
        after = tuple(sorted((picked - {donor}) | {prefix}))
        require(len(after) == len(local) and after[-1] == local[-1]
                and prefix not in picked and donor in picked,
                "Seat exchange changed physical width or boundary")
        require(sum(1 for i in after if prior[i] == 0) >= 1,
                "Exchange left no selected unresolved MASK")
        self.used_blocks.add(block)
        item = dict(block=block, tick=int(tick), start=int(start), offer=int(offer),
                    ready=int(ready), donor=donor, prefix=prefix,
                    donor_absolute=start + donor, prefix_absolute=start + prefix,
                    right_edge=local[-1], before=list(local), after=list(after),
                    due=sorted(due), resident_prefix=prefix in resident)
        self.exchanges.append(item)
        # The pinned runtime performs ``start + local``; preserve its ndarray
        # representation and dtype rather than returning a tuple.
        return np.asarray(after, dtype=np.int64)


def require_state(*, local, prior, width, dirty, start, ready, bank, donor, prefix):
    require_local = lambda rows: bool(rows and tuple(sorted(set(rows))) == rows
                                     and all(0 <= i < width for i in rows))
    if not require_local(local) or not require_local((donor,)) or not require_local((prefix,)):
        raise RuntimeError("Invalid block-local exchange")
    if donor not in local or prefix in local or prefix >= donor:
        raise RuntimeError("Exchange is not an omitted prefix for an earlier selected row")
    if prior[donor] != 0 or prior[prefix] != 1 or dirty[start + donor] or dirty[start + prefix]:
        raise RuntimeError("Expected unresolved donor MASK and resolved clean prefix")
    if not bank.initialized[start + donor] or not bank.initialized[start + prefix]:
        raise RuntimeError("Both exchange rows must already have resident KV")
    if any(dirty[start:start + ready]):
        raise RuntimeError("Paid prefix contains debt")
