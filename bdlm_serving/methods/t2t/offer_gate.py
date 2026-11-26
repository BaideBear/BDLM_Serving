"""Spend the high offer according to pre-forward state of the current physical block."""
from .slot_trace import BudgetOffer


def choose_native(info, offered, low, threshold, mode):
    if (not 0 < low < info.block_width or offered not in (low, info.block_width)
            or not 0 <= threshold <= info.block_width
            or not 0 <= info.masks <= info.block_width
            or not 0 <= info.mandatory <= info.block_width or info.tick < 0):
        raise ValueError("Invalid offer or physical block state")
    if mode == "all":
        accept = True
    elif mode == "tail":
        accept = info.masks <= threshold
    elif mode == "early":
        accept = info.masks > threshold
    else:
        raise ValueError("Unknown native offer rule")
    return (BudgetOffer(offered, native_full=True) if accept and offered!=low
            else BudgetOffer(low))
