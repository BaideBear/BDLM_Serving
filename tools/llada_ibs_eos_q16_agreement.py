"""Opt-in Q16 agreement over native strict IBS with a common paid EOS close."""
from inspect import getsource
from pathlib import Path
from textwrap import dedent

from bdlm_serving.methods.t2t.paid_agreement import EvidenceLedger
from tools.llada_ibs_eos_stop import B, FIXED_SHA, frozen_eos, require
from tools.llada_ibs_eos_eval import file_sha

Q = 16
ARMS = ("stock", "tracked", "apply")


def observe(ledger, tape, excluded, apply, *, block, tick, positions, proposals, probabilities, native):
    if not positions:
        return ()
    extra = ledger.observe(block=block, tick=tick, positions=positions, proposals=proposals,
                           probabilities=probabilities, native=native, excluded_ids=excluded)
    applied = extra if apply else ()
    tape.append(dict(block=block, tick=tick, positions=positions, proposals=proposals,
                     probabilities=probabilities, native=native, eligible=extra, applied=applied))
    return applied


def debt(dirty, *, start, ready, initialized):
    rows = sorted(dirty)
    require(len(rows) <= Q and all(start+ready <= x < start+B and x in initialized for x in rows),
            "Invalid suffix debt")
    return rows


def install_run(fixed, arm, receipt, excluded):
    require(arm in ARMS and file_sha(Path(fixed.__file__)) == FIXED_SHA, "Wrong arm or fixed source")
    source = dedent(getsource(fixed.FixedOptimus.run))
    ledger = EvidenceLedger(floor=0.5) if arm != "stock" else None

    def tap(**kw):
        return observe(ledger, receipt["ticks"], excluded, arm == "apply", **kw)

    additions = {
        "    p = prompt.shape[1]\n": (
            "    _check(self.kind == 'llada21' and self.mode == 'ibs' and self.chunk == 16 "
            "and self.caps == (16,) and not self.prefix_refresh, 'Expected strict IBS Q16')\n"
            "    certified_end = None\n"),
        "            now_mask = (x[0, start:start+width].cpu().numpy() != mask_id).astype(np.uint8)\n": (
            "            if _tracked:\n"
            "                measured = masked & (positions >= p) & (positions < p+gen_length)\n"
            "                if bool(measured.any()):\n"
            "                    abs_rows = positions[measured]\n"
            "                    local = selected[measured]\n"
            "                    extra = _tap(block=start//32, tick=tick, positions=abs_rows.tolist(),\n"
            "                        proposals=proposed[measured].tolist(), probabilities=confidence[measured].tolist(),\n"
            "                        native=accepted[local].tolist())\n"
            "                    if extra:\n"
            "                        slots = torch.as_tensor(extra, device=x.device, dtype=torch.long)\n"
            "                        picked_extra = torch.isin(positions, slots)\n"
            "                        _check(int(picked_extra.sum()) == len(extra) and "
            "bool((x[0, slots] == mask_id).all()), 'Extra was not measured and unresolved')\n"
            "                        x[0, positions[picked_extra]] = proposed[picked_extra]\n"
            "                        accepted[selected[picked_extra]] = True\n"),
        "            if ready >= 32:\n": (
            "            writes = events[-1]['written_absolute']\n"
            "            _receipt['writes'].append([[pos, int(x[0, pos].item())] for pos in writes])\n"
            "            eos_pos = _scan(lambda lo, hi: x[0, lo:hi].tolist(), prompt=p,\n"
            "                            cap=gen_length, start=start, ready=ready, eos=eos_ids)\n"
            "            if eos_pos is not None:\n"
            "                due = _debt(dirty,start=start,ready=ready,initialized=cache.initialized)\n"
            "                _receipt['certificate'] = dict(event=len(events)-1,eos=eos_pos,\n"
            "                    block=start//32,step=step,ready=ready,debt_before=due)\n"
            "                if due:\n"
            "                    pos = torch.as_tensor(due,device=x.device,dtype=torch.long)\n"
            "                    forward(pos,'certificate_debt')\n"
            "                    events[-1].update(block=start//32,step=step,physical_cap=offer,\n"
            "                        visible_kv=cache.visible_end,dirty_before=due,written_absolute=[],\n"
            "                        sampled_absolute=[],edited_positions=[])\n"
            "                    dirty.difference_update(due)\n"
            "                _check(not dirty,'Unpaid certificate debt')\n"
            "                _receipt['certificate']['debt_after'] = sorted(dirty)\n"
            "                certified_end = eos_pos+1\n"
            "                break\n"),
        "        if any(bool((x[:, p:start+32] == eos).any()) for eos in eos_ids):\n": (
            "        if certified_end is not None:\n"
            "            x = x[:, :certified_end]\n"
            "            break\n"),
    }
    for anchor, addition in additions.items():
        require(source.count(anchor) == 1, f"Unregistered insertion point: {anchor.strip()}")
        source = source.replace(anchor, addition + anchor)
    ns = dict(vars(fixed), _check=require, _receipt=receipt, _scan=frozen_eos, _debt=debt,
              _tracked=ledger is not None, _tap=tap)
    exec(compile(source, "<strict-ibs-q16-paid-agreement>", "exec"), ns)
    return ns["run"]
