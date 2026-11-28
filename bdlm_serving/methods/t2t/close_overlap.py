"""Optional successor work on a mask-free head's final sparse calls."""


def spare_successor_rows(*, offer, head_start, width, generated_end, selected_head,
                         dirty_head, resident_head, unresolved_head, generated_eos,
                         already_tried):
    """Return additional rows only. The caller keeps its full original head plan.

    Positions are absolute. ``generated_end`` is the exclusive output cap in
    the same coordinate system. A partial prompt contributes resident head
    rows, never a generated EOS. An attempt, even one later invalidated, spends
    this head's one opportunity.
    """
    if not (width > 0 and 0 < offer <= 2*width and head_start >= 0 and generated_end > head_start):
        raise ValueError("invalid head extent or decoder offer")
    head = set(range(head_start, head_start+width))
    original = tuple(selected_head)
    selected, due = set(original), set(dirty_head)
    resident, unresolved = set(resident_head), set(unresolved_head)
    if (not selected or len(original) != len(selected) or len(selected) > offer or
            not (selected | due | resident | unresolved) <= head):
        raise ValueError("invalid original head plan")
    if (already_tried or generated_eos or unresolved or not due <= selected or
            head != selected | resident):
        return ()
    end = head_start+width
    return tuple(range(end, min(end+width, generated_end, end+offer-len(selected))))


def may_promote_successor(*, head_sealed, head_changed, request_ended,
                          projection_context, promotion_context):
    """A positive result permits reusing the *paid inputs*, not newly sampled KV.

    The adapter owns context versions; change the version when an encoder or
    other dependency changes. A promoted sample that changes an input is
    still owed next round. False means discard tentative work, not a paid
    refresh or an authority to retract an already published token.
    """
    if projection_context is None or promotion_context is None:
        return False
    return (head_sealed and not head_changed and not request_ended and
            projection_context == promotion_context)
