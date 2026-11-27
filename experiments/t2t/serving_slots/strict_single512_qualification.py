"""Compare the public handle with a sealed fixed ledger and interrupt a real stop."""
import gzip
from hashlib import sha256
import json

PROFILE = "dense_stop_joint_single512"
BOUNDARIES = ("after_top_stop", "interrupt_top_256", "interrupt_top_32")


def require(value, message):
    if not value:
        raise ValueError(message)


def reference(path):
    r = json.loads(path.read_bytes())
    tape = path.with_suffix(".ledger.json.gz").read_bytes()
    require(r["schema"] == "single512-v1" and r["on"] is True
            and sha256(tape).hexdigest() == r["ledger_sha256"], "Wrong or unsealed fixed reference")
    return r, json.loads(gzip.decompress(tape))["trace"]


class Boundary:
    def __init__(self, suite, ref, trace):
        self.suite, self.ref, self.trace = suite, ref, trace
        self.target = next((h for h in ref["stop"]["hits"] if h["actionable"]), None)
        require(suite == "fixed512" or (suite in BOUNDARIES and self.target), "No actionable reference")
        self.prefix = 0
        self.diverged = False
        self.events = []

    def offer(self, tick, head):
        if self.suite == "fixed512":
            return 512
        t = self.target["tick"]
        if self.suite == "after_top_stop":
            return 32 if tick > t and head == self.target["ordinal"] else 512
        return int(self.suite.rsplit("_", 1)[1]) if tick == t else 512

    def observe(self, e):
        t, q = e["tick"], e["offer"]
        self.diverged |= q != 512
        if not self.diverged:
            expected = self.trace[t]
            keys = ("tick","offer","ordinal","route","positions","sampled_absolute","written_absolute",
                    "dirty_before_absolute","dirty_after_absolute","head_mask_before","head_initialized_before",
                    "opened","encoded","promoted","discarded","decoder_F","decoder_Q","encoder_F","encoder_Q")
            require(all(e[k] == expected[k] for k in keys), f"Physical fixed prefix differs at tick {t}")
            parts = [{k:v for k,v in p.items() if k != "tail_updates"} for p in expected["parts"]]
            require(e["parts"] == parts, f"Fixed row transaction differs at tick {t}")
            s, m = expected["single512"], e["fullstop"]
            require((s["qualified"],s["hit"],s["applied"],s["epoch"],s["resident"]) ==
                    (m["in_scope"],m["hit"],m["latched"],m["epoch"],m["resident"]), "Fixed stop differs")
            self.prefix += 1
        if self.suite == "fixed512" or e["ordinal"] != self.target["ordinal"] or t < self.target["tick"]-1:
            return
        p = e["parts"][0]
        self.events.append(dict(tick=t, offer=q, route=e["route"], full=e["fullstop"]["in_scope"],
                                hit=e["fullstop"]["latched"], age=p["age_after"], width=e["decoder_Q"],
                                masks=e["head_mask_before"]-len(p["admitted"])+len(p["revoked"]),
                                debt_before=len(e["dirty_before_absolute"]), debt_after=len(e["dirty_after_absolute"])))

    def finish(self):
        if self.suite == "fixed512":
            require(self.prefix == len(self.trace), "Incomplete fixed ledger")
            return dict(prefix_ticks=self.prefix, ledger_sha256=self.ref["ledger_sha256"])
        t = self.target["tick"]
        events = self.events
        require(len(events) >= 3 and events[0]["tick"] == t-1 and events[0]["full"] and not events[0]["hit"],
                "Missed previous qualified top tick")
        cur, following = events[1:3]
        require(cur["tick"] == t and following["tick"] == t+1, "Missing interruption")
        if self.suite == "after_top_stop":
            require(self.prefix == t+1 and cur["hit"] and cur["masks"] > 0 and cur["debt_after"] > 0
                    and cur["age"] < 48 and following["debt_before"] == cur["debt_after"], "No live latch/debt")
            closing = events[2:]
            require(all(p["offer"] == 32 and p["age"] == cur["age"] and
                        p["route"] in ("drain","verify","verify_paid") for p in closing)
                    and sum(p["width"] for p in closing if p["route"].startswith("verify")) == 256
                    and closing[-1]["debt_after"] == 0, "Latched head was not freshly verified at Q32")
        else:
            require(self.prefix == t and cur["offer"] == self.offer(t,self.target["ordinal"])
                    and cur["full"] == (cur["offer"] == 256) and not cur["hit"]
                    and following["offer"] == 512 and following["full"] and not following["hit"],
                    "Cross-offer stability or missing real restart")
        return dict(prefix_ticks=self.prefix, ledger_sha256=self.ref["ledger_sha256"],
                    fixed_actionable_tick=t, ordinal=self.target["ordinal"], epoch=self.target["epoch"],
                    events=events)
