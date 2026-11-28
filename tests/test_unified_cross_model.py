"""Compare state adapters, not the model outputs. No model or weights needed."""
from types import SimpleNamespace

import pytest
import torch

from bdlm_serving.methods.t2t.unified_rows import plan_rows
from experiments.t2t.serving_slots import unified_gemma as gemma
from experiments.t2t.serving_slots import unified_llada as llada


@pytest.mark.parametrize("b", (32, 256))
@pytest.mark.parametrize("opened", (False, True))
def test_two_encodings_of_the_same_physical_state(monkeypatch, b, opened):
    # Inject identical geometry only in this adapter test. Real checkpoints retain their B.
    monkeypatch.setattr(llada, "B", b)
    monkeypatch.setattr(gemma, "B", b)
    full = frozenset(range(b))
    monkeypatch.setattr(gemma, "FULL", full)
    start = b
    mask = {b-4, b-2}
    hs = SimpleNamespace(rows=SimpleNamespace(accepted=full-mask, dirty={2}, initialized=full),
                         can_sample=True)
    ns = (SimpleNamespace(rows=SimpleNamespace(accepted={0}, dirty={0}, initialized={0, 1}),
                          can_sample=True) if opened else None)
    # LLaDA stores actual MASK ids; even a Gemma unresolved position can carry
    # an ordinary vocabulary id, so the Gemma adapter must use acceptance bits.
    x = torch.full((1, 3*b), 17)
    x[0, [start+i for i in mask]] = 0
    x[0, 2*b:] = 0
    live, dirty = set(range(start, 2*b)), {start+2}
    if opened:
        x[0, 2*b] = 19
        live.update((2*b, 2*b+1))
        dirty.add(2*b)
    rq = SimpleNamespace(mask_id=0, origin=start, head=start, end=3*b, x=x, dirty=dirty,
                         stopping=False, bank=SimpleNamespace(initialized=live))
    lpair = llada.live_blocks(rq, opened)
    gpair = (gemma.block_view(start, hs, live, sampling=True),
             gemma.block_view(start+b, ns, live, sampling=True))
    assert lpair == gpair
    for offer in (b//8, b//4, b//2, b, 2*b):
        assert plan_rows(offer, b, *lpair, opened=opened) == plan_rows(offer, b, *gpair, opened=opened)
