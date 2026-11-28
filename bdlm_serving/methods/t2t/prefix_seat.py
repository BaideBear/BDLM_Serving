"""Single optional MASK-to-prefix exchange, shared across fixed-budget adapters."""
from operator import index


def choose_prefix_seat(rows, *, unresolved, resident, due, paid_prefix_end, offer, used=False):
    """Return (outgoing warm MASK, incoming paid prefix), using block-local indices.

    The caller supplies the pre-projection state of one ordinary sample step.
    `resident` contains eligible generated rows only (no prompt). A successful
    exchange consumes the block's one opportunity; no-ops do not.
    """
    rows = tuple(map(index, rows))
    offer, end = index(offer), index(paid_prefix_end)
    if offer <= 0 or end < 0 or len(rows) > offer or tuple(sorted(set(rows))) != rows:
        raise ValueError("Invalid physical plan or prefix frontier")
    if rows and rows[0] < 0:
        raise ValueError("Expected block-local positions")
    if used or len(rows) != offer or not end:
        return None
    picked = set(rows)
    masks, warm, debt = map(set, (unresolved, resident, due))
    if not masks or not debt <= picked or min(masks) < end:
        return None
    selected_masks = picked & masks
    if len(selected_masks) < 2:
        return None
    donors = selected_masks & warm - debt - {min(masks), rows[-1]}
    prefix = {p for p in warm - debt - picked if 0 <= p < min(end, rows[-1])}
    if not donors or not prefix:
        return None
    return max(donors), max(prefix)
