from hashlib import sha256
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bdlm_serving.methods.optimus import fixed, upstream

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"experiments/t2t/serving_slots/controls/ibs-tail-overlap/fused.py"
spec = importlib.util.spec_from_file_location("ibs_joint",SRC)
joint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(joint)


class Spy:
    config = SimpleNamespace(num_hidden_layers=2)
    def __init__(self, edit=False, head_eos=False):
        self.edit, self.head_eos = edit, head_eos
        self.calls = []

    def __call__(self, x, *, position_ids, attention_mask, past_key_values, **kw):
        pos = position_ids[0]
        self.calls.append((pos.tolist(),x[0].tolist(),attention_mask[0,0].clone()))
        for i in range(2):
            v = x[:,None,:,None]
            past_key_values.update(v,v,i)
        ids = torch.where(x==0,torch.where((pos[None]<32) & (not self.head_eos),2,3),x)
        if self.edit and pos[0]<32 and pos[-1]>=32:
            ids[0,0] = 4
        logits = torch.full((*ids.shape,5),-8.)
        logits.scatter_(-1,ids[...,None],8.)
        return SimpleNamespace(logits=logits)


def run(monkeypatch, arm, edit=False, length=40, head_eos=False):
    if os.environ.get("BDLM_OPTIMUS_CONTROLS"):
        controls = Path(os.environ["BDLM_OPTIMUS_CONTROLS"])
        assert sha256((controls/"sequence.py").read_bytes()).hexdigest()=="99c80047d517113579e04fd57443471a759a738dc30b206ba1342e0d4f9b3da2"
        assert sha256((controls/"unmasking.py").read_bytes()).hexdigest()=="bbc3c3328e5497aa08aed6e04150ab0383f95dc06b4ff570a69adaed4a769746"
        monkeypatch.setattr(upstream,"control_directory",lambda _:controls)
    try:
        upstream.control_directory("Optimus")
    except (RuntimeError, FileNotFoundError):
        pytest.skip("set BDLM_OPTIMUS_CONTROLS to the pinned Optimus sources")
    spy, receipts = Spy(edit,head_eos), []
    engine = fixed.FixedOptimus(spy,"llada21",8,"ibs",physical_caps=(8,))
    call = (fixed.FixedOptimus.run if arm=="stock" else joint.install_run(fixed,arm=arm,receipt=receipts))
    x,ev,_ = call(engine,torch.full((1,31),1),length,0,0,[3])
    assert len(ev)==len(spy.calls) and all(e["q_tokens"]==len(c[0])<=8 for e,c in zip(ev,spy.calls))
    return x,ev,spy.calls,receipts


def test_stock_noop_and_joint_debt(monkeypatch):
    stock,noop = (run(monkeypatch,arm) for arm in ("stock","noop"))
    assert torch.equal(stock[0],noop[0]) and stock[1]==noop[1] and noop[3]
    _,ev,calls,_ = run(monkeypatch,"overlap")
    k = next(i for i,e in enumerate(ev) if "successor" in e)
    v=ev[k]["successor"]
    pos,_,mask=calls[k]
    assert pos==v["head_rows"]+v["tail_rows"] and v["layer_widths"]==[len(pos)]*2
    assert not mask[:len(v["head_rows"]),v["head_key_end"]:].any()
    assert mask[:len(v["head_rows"]),:v["head_key_end"]].all() and mask[len(v["head_rows"]):].all()
    assert v["promoted"] and v["tail_writes"] and not v["discarded_writes"]
    assert ev[k+1]["dirty_before"]==v["tail_writes"]
    inputs=dict(zip(calls[k+1][0],calls[k+1][1],strict=True))
    assert all(inputs[p]==3 for p in v["tail_writes"])


def test_roll_back_dirty_head_and_veto_eos_or_output_cap(monkeypatch):
    _,ev,calls,_=run(monkeypatch,"overlap",edit=True)
    k=next(i for i,e in enumerate(ev) if "successor" in e)
    v=ev[k]["successor"]
    assert v["head_changed"] and not v["promoted"] and v["discarded_writes"] and not v["tail_writes"]
    assert set(ev[k+1]["dirty_before"]).isdisjoint(v["tail_rows"])
    next_tail=next((p,ids) for p,ids,_ in calls[k+1:] if min(p)>=32)
    assert next_tail[1]==[0]*len(next_tail[0])
    for kw in ({"length":1},{"head_eos":True}):
        _,ev,_,r=run(monkeypatch,"overlap",**kw)
        assert not r and all("successor" not in e for e in ev)


def test_truncate_spare_positions_to_end_of_generation(monkeypatch):
    _,ev,_,_=run(monkeypatch,"overlap",length=3)
    v=next(e["successor"] for e in ev if "successor" in e)
    assert v["tail_rows"]==[32,33]
