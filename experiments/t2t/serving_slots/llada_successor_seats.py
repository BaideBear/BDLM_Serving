"""Q64: spend unused seats on clean rows of an already open, still-active tail."""
from dataclasses import replace


def extra_rows(plan, sampled, split, limit, masks, resident):
    if plan.drain_only or plan.cold or not sampled:
        return ()
    tail = set(range(split, limit))
    if not (masks & tail) or not tail <= resident:
        return ()
    added = tail - set(plan.rows)
    if added & masks or len(plan.rows) + len(added) > 64:
        raise ValueError("Unexpected unserved MASK or physical-width overflow")
    return tuple(sorted(added))


def variants(control):
    class TailRead(control):
        edits_enabled = False

        def _cross_plan(self, q):
            plan, sample, phase, cap = super()._cross_plan(q)
            self.extra = ()
            if (q == 64 and (self._cross_evidence or {}).get("opened")
                    and sample and not plan.cold and not plan.drain_only):
                split = self.head + 32
                masks, prefix, _ = self._inputs(64)
                self.extra = extra_rows(plan, sample, split, min(split+32,self.end),
                                        masks, self.bank.initialized)
                if self.extra:
                    added = set(self.extra)
                    if added & self.dirty:
                        raise RuntimeError("Unpaid mandatory tail")
                    plan = replace(plan, rows=tuple(sorted(set(plan.rows)|added)),
                                   prefix=tuple(sorted(set(plan.prefix)|(added&set(prefix)))))
                    if self.edits_enabled:
                        sample = tuple(sorted(set(sample)|added))
            return plan, sample, phase, cap

        def _step(self, q):
            event = super()._step(q)
            if self.extra:
                event["supplemental_tail"] = self.extra
            return event

    class TailEdit(TailRead):
        edits_enabled = True

    return control, TailRead, TailEdit


def first(old, new, *, edit):
    from .llada_fixed_family_eval import require
    key = "supplemental_tail"
    for a,b in zip(old["trace"],new["trace"]):
        if key not in b:
            require(a==b, "Changed trajectory before supplemental tail selection")
            continue
        add = set(b[key])
        s = b["pre_state"]
        tail = set(range(s["head"]+32, min(s["head"]+64,s["end"])))
        require(add and s["last_end"]>=max(tail)+1 and set(b["rows"])>=tail>=add
                and add.isdisjoint(set(b["dirty_before"])|set(b["cold"])|set(b["mask"]))
                and b["tail_opened"] and not b["cold"] and a["sampled"]
                and all(a[k]==b[k] for k in ("pre_state","phase","head_q","head_mask_at_plan",
                                              "dirty","cold","mask","visible_kv","sample_clock"))
                and all(e["layer_physical_q"]==[e["q"]]*20 and e["q"]<=64 for e in (a,b)),
                "Unmatched warm-tail opportunity")
        if edit:
            require(a[key]==b[key] and a["rows"]==b["rows"] and a["prefix"]==b["prefix"]
                    and add.isdisjoint(a["sampled"]) and set(b["sampled"])==set(a["sampled"])|add
                    and a["admitted"]==b["admitted"],"Edit changed more than sampler eligibility")
        else:
            require(add.isdisjoint(a["rows"]) and set(b["rows"])==set(a["rows"])|add
                    and a["sampled"]==b["sampled"],"Read-only treatment changed incumbent seats")
        return dict(tick=b["tick"],width=b["q"],old_width=a["q"],added=sorted(add))
    require(len(old["trace"])==len(new["trace"]) and all(old[k]==new[k] for k in
            ("output_ids","F","Q","failure","score")), "Unexplained effect without supplemental seats")
    return None
