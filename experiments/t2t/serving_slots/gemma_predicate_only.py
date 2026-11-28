"""Serial Q256 experiment: remove local all-accepted as a separate close reason."""


def native_canvas(base):
    class PredicateCanvas(base):
        @property
        def can_sample(self):
            return not self.rows._native_stop and self.rows.age < self.rows.max_steps
    return PredicateCanvas


def first_opportunity(events):
    """Index of the completed sample after which off would close but on continues."""
    return next((i for i,e in enumerate(events) if e["route"] == "dense_single" and
                 e["accepted"] == 256 and not e["hit"] and e["age"] < 48), None)


def compare(off, on, off_audit, on_audit, off_result, on_result):
    j = first_opportunity(off)
    if j != first_opportunity(on):
        raise ValueError("First close opportunity not matched")
    if j is None:
        if off != on or off_audit != on_audit or off_result != on_result:
            raise ValueError("Changed before any stop opportunity")
    else:
        if (off[:j+1] != on[:j+1] or off_audit["ticks"][:j+1] != on_audit["ticks"][:j+1]
                or off_audit["init"][:1+off[j]["ordinal"]] != on_audit["init"][:1+off[j]["ordinal"]]
                or off[j+1]["route"] not in ("verify", "verify_paid") or on[j+1]["route"] != "dense_single"
                or not (off[j]["ordinal"] == off[j+1]["ordinal"] == on[j+1]["ordinal"])
                or off[j+1]["age"] != off[j]["age"] or on[j+1]["age"] != off[j]["age"] + 1):
            raise ValueError("Not a matched verify versus real next sample")
    return j
