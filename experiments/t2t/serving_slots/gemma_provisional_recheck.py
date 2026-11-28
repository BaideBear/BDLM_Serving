"""Opt-in, one-shot correction of repeat-extra IDs at real first payment."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np

from . import gemma_paid_agreement as rep
from .gemma_ibs_certified_eos import BASELINE_SHA, B, Stop, require


class Recheck(rep.Agreement):
    def __init__(self, *, correct, excluded_ids, torch, rng):
        super().__init__(apply=True, excluded_ids=excluded_ids, floor=.5)
        self.correct = correct
        self.torch, self.rng = torch, rng
        self.pending, self.replacements = {}, {}
        self.corrections, self.measurements, self.same_step = [], [], []

    def observe(self, *args):
        updated = super().observe(*args)
        step = self.observations[-1]
        extra = set(step["extra"])
        for p, token, native in zip(step["positions"], step["proposals"], step["native"], strict=True):
            if native or p in extra:
                require(p not in self.pending and p not in self.replacements, "Duplicate provisional write")
                self.pending[p] = (token, "extra" if p in extra else "native", step["tick"])
        return updated

    def first_payments(self, absolute, dirty, ids, logits, tick, stop, start, ready):
        paid = [(j, p) for j, p in enumerate(absolute.tolist()) if dirty[p]]
        require(all((p in self.pending) ^ (p in self.replacements) for _, p in paid),
                "True payment is missing or has two owners")
        newly_due = [(j, p) for j, p in paid if p in self.pending]
        tokens = ids[0, absolute.tolist()].tolist() if paid else []
        for j, p in paid:
            if p in self.replacements:
                e = self.replacements.pop(p)
                require(tokens[j] == e["top"] and tick > e["payment"], "Replacement was not physically paid")
                e["repayment"] = tick
        edits = []
        for begin in range(0, len(newly_due), 8):
            chunk = newly_due[begin:begin+8]
            prob = logits[[j for j, _ in chunk]].float().softmax(-1)
            best_p, best = prob.max(-1)
            written_p = prob.gather(-1, self.torch.tensor([[tokens[j]] for j, _ in chunk],
                                                         device=prob.device)).flatten().tolist()
            for (j, p), top, confidence, old_p in zip(chunk, best.tolist(), best_p.tolist(),
                                                     written_p, strict=True):
                old, role, write = self.pending.pop(p)
                require(write < tick and old == tokens[j], "Wrong first-payment decoder input ID")
                ordinary = top not in self.excluded_ids
                event = dict(role=role, pos=p, write=write, payment=tick, old=old, top=top,
                             top_p=confidence, written_p=old_p, ordinary=ordinary)
                self.measurements.append(event)
                if self.correct and role == "extra" and ordinary and top != old and confidence > .9:
                    hit = stop.trigger
                    require(start + ready <= p < start+B and
                            (hit is None or (hit["kind"] == "suffix_debt" and
                                             p > start + hit["eos_index"])),
                            "NO-GO: correction would cross a frozen prefix or certified EOS")
                    require(all(e["pos"] != p for e in self.corrections), "Second correction of one position")
                    edit = dict(event, stage="terminal_suffix" if hit is not None else "ordinary",
                                prior_ready=ready, repayment=None)
                    self.corrections.append(edit)
                    edits.append(edit)
        return edits

    def publish(self, edits, ids, resolved, dirty, frontier_prior, absolute, unresolved,
                tick, start, ready):
        # Compare all ordinary sampled output IDs (including rejected-row noise)
        # and the real post-sampler RNG before publishing disjoint replacements.
        sampled = absolute[unresolved].tolist()
        self.same_step.append(dict(tick=tick, positions=sampled,
                                   ids=ids[0, sampled].tolist() if sampled else [], rng=self.rng()))
        for e in edits:
            p = e["pos"]
            require(p >= start + ready and resolved[p] and not dirty[p] and p not in sampled,
                    "NO-GO: replacement touched a frozen, unresolved or newly sampled row")
            ids[0, p] = e["top"]
            dirty[p] = True
            frontier_prior[p-start] = 0
            self.replacements[p] = e


class RecheckStop(Stop):
    def __init__(self, offer):
        super().__init__("paid", offer)
        self.terminal_payments = []

    def rows(self, normal, prior, dirty, start, offer):
        if self.trigger and self.terminal_payments:
            debt = np.flatnonzero(dirty) - start
            require(offer == self.offer and len(self.terminal_payments) == 1 and
                    self.trigger["block"] * B == start and 0 < len(debt) <= offer and
                    all(self.trigger["eos_index"] < p < B for p in debt) and
                    prior[debt].all() and np.isin(debt, normal).all(),
                    "NO-GO: terminal repayment cannot use a bounded suffix-only projection")
            return debt
        return super().rows(normal, prior, dirty, start, offer)

    def after_update(self, ids, resolved, dirty, start, ready, eos, trace):
        if self.trigger is None:
            return super().after_update(ids, resolved, dirty, start, ready, eos, trace)
        t = trace[-1]
        hit = self.trigger
        debt = np.flatnonzero(dirty).tolist()
        prev = self.terminal_payments
        expected = prev[-1]["debt_after"] if prev else hit["debt"]
        require(hit["kind"] == "suffix_debt" and self.payment is None and len(prev) <= 1 and
                t["tick"] == hit["tick"] + len(prev) + 1 and t["positions"] == expected and
                t["offer"] == self.offer and not t["admitted"] and not t["accepted"] and
                start // B == hit["block"] and hit["eos_index"] < ready and
                ids[0, start + hit["eos_index"]].item() == hit["eos_id"] and
                set(debt) <= set(expected) and not (prev and debt),
                "NO-GO: terminal payment left debt or changed the certified prefix")
        self.payment = dict(tick=t["tick"], positions=t["positions"], Q=t["q"])
        prev.append(dict(self.payment, debt_after=debt))
        if debt:
            self.payment = None
            return None
        return ids[0, :start + hit["eos_index"] + 1].tolist()


@lru_cache(maxsize=1)
def _loop(rt):
    raw = Path(rt.__file__).read_bytes()
    require(sha256(raw).hexdigest() == BASELINE_SHA, "Unqualified IBS runtime")
    _, rest = raw.decode().split(rep._SIG)
    body = rest.split("\n\ndef ", 1)[0]
    pay = "                dirty[absolute] = False\n"
    publish = "            ctl.update_from_mask(frontier_prior, mark_ready_if_all_left_non_mask=True,\n"
    anchors = (rep._PLAN, rep._AFTER, rep._UNMASK, pay, publish)
    require(all(body.count(a) == 1 for a in anchors), "Ambiguous pinned-loop hook anchors")
    body = body.replace(rep._PLAN, "            local = _stop.rows(local, prior, dirty, start, offer)\n" + rep._PLAN)
    body = body.replace(rep._UNMASK, """\
                updated = _recheck.observe(unmask, raw, proposal, prior, sel, width, start, tick)
""")
    body = body.replace(pay, """\
                edits = _recheck.first_payments(absolute, dirty, ids, logits, tick, _stop, start, ctl.cached_prefix_len)
""" + pay)
    body = body.replace(publish, """\
            _recheck.publish(edits, ids, resolved, dirty, frontier_prior, absolute, unresolved,
                             tick, start, ctl.cached_prefix_len)
""" + publish)
    body = body.replace(rep._AFTER, """\
            delivery = _stop.after_update(ids, resolved, dirty, start, ctl.cached_prefix_len, eos, trace)
            if delivery is not None:
                return delivery, trace
""" + rep._AFTER)
    namespace = {}
    exec(compile(rep._SIG[:-2] + ", _stop, _recheck):" + body, "<provisional-recheck>", "exec"),
         vars(rt), namespace)
    return namespace["_generate"]


def generate(rt, model, data, controls, unmask, stop, recheck):
    bank = rt.Resident()
    bank.bypass = False
    with rt.resident_attention(model, bank):
        return _loop(rt)(bank, model, data, controls, unmask, "ibs", stop.offer, False,
                         (stop.offer,), stop, recheck)
