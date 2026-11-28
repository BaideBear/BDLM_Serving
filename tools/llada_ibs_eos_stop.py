"""Opt-in terminal certificate for the pinned strict IBS Q4 executor."""
from hashlib import sha256
from inspect import getsource
from pathlib import Path
from textwrap import dedent

B, Q = 32, 4
FIXED_SHA = "b95003b5f8867bd6ef7d5126c9814e08fb6d03bab57f67642618028d8a1960d4"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def frozen_eos(read_frozen, *, prompt, cap, start, ready, eos):
    """The reader is called only on the generated part of this frozen block."""
    require(start % B == 0 and 0 <= ready <= B and cap >= 1, "Invalid frozen frontier")
    lo, hi = max(prompt, start), min(start + ready, prompt + cap)
    if lo >= hi:
        return None
    values = read_frozen(lo, hi)
    require(len(values) == hi - lo, "Short frozen read")
    return next((lo + i for i, v in enumerate(values) if v in eos), None)


def debt_rows(dirty, *, prompt, start, ready, eos_pos, initialized):
    require(max(start, prompt) <= eos_pos < start + ready, "EOS outside the certified prefix")
    rows = sorted(dirty)
    require(len(rows) <= Q and all(start + ready <= v < start + B and v in initialized for v in rows),
            "Unpaid prefix debt or invalid suffix debt")
    return rows


def install_run(fixed, arm, receipt):
    """Return a request-local opt-in method; the real planner and writes are unchanged."""
    require(arm in ("instrumented", "certificate"), "Only explicit opt-in arms are allowed")
    require(sha256(Path(fixed.__file__).read_bytes()).hexdigest() == FIXED_SHA, "Unexpected fixed IBS source")
    src = dedent(getsource(fixed.FixedOptimus.run))
    anchors = {
        "    p = prompt.shape[1]\n": (
            "    _check(self.kind == 'llada21' and self.mode == 'ibs' and self.chunk == 4 "
            "and self.caps == (4,) and not self.prefix_refresh, 'Expected unmodified IBS Q4')\n"
            "    certified_end = None\n"),
        "            if ready >= 32:\n": (
            "            writes = events[-1]['written_absolute']\n"
            "            _receipt['writes'].append([[pos, int(x[0, pos].item())] for pos in writes])\n"
            "            if _receipt['certificate'] is None:\n"
            "                eos_pos = _scan(lambda lo, hi: x[0, lo:hi].tolist(), prompt=p,\n"
            "                                cap=gen_length, start=start, ready=ready, eos=eos_ids)\n"
            "                if eos_pos is not None:\n"
            "                    due = _debt(dirty, prompt=p, start=start, ready=ready,\n"
            "                                eos_pos=eos_pos, initialized=cache.initialized)\n"
            "                    _receipt['certificate'] = dict(event=len(events)-1, eos=eos_pos,\n"
            "                        block=start//32, step=step, ready=ready, debt_before=due)\n"
            "                    if _stop:\n"
            "                        if due:\n"
            "                            pos = torch.as_tensor(due, device=x.device, dtype=torch.long)\n"
            "                            forward(pos, 'certificate_debt')\n"
            "                            events[-1].update(block=start//32, step=step, physical_cap=offer,\n"
            "                                visible_kv=cache.visible_end, dirty_before=due, written_absolute=[],\n"
            "                                sampled_absolute=[], edited_positions=[])\n"
            "                            dirty.difference_update(due)\n"
            "                        _check(not dirty, 'Certificate cannot leave unpaid debt')\n"
            "                        _receipt['certificate']['debt_after'] = sorted(dirty)\n"
            "                        certified_end = eos_pos + 1\n"
            "                        break\n"),
        "        if any(bool((x[:, p:start+32] == eos).any()) for eos in eos_ids):\n": (
            "        if certified_end is not None:\n"
            "            x = x[:, :certified_end]\n"
            "            break\n"),
    }
    for anchor, insert in anchors.items():
        require(src.count(anchor) == 1, f"Expected unique IBS hook point: {anchor.strip()}")
        src = src.replace(anchor, insert + anchor)
    ns = dict(vars(fixed), _check=require, _receipt=receipt, _stop=arm == "certificate",
              _scan=frozen_eos, _debt=debt_rows)
    exec(compile(src, "<pinned-ibs-q4-eos>", "exec"), ns)
    return ns["run"]
