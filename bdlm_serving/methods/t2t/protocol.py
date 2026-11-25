"""Shared frozen30 protocol: one case list, explicit budgets, no free closure work."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Arm:
    budget: int
    fresh: int
    mask: int
    selection: str

    @property
    def edit(self):
        return self.budget - self.fresh - self.mask

    @property
    def policy(self):
        return f"t2t_q{self.budget}_f{self.fresh}_m{self.mask}_{self.selection}"

    @classmethod
    def parse(cls, text, block):
        q, f, m, selection = text.split(":")
        obj = cls(int(q), int(f), int(m), selection)
        # An explicitly labelled, single-block B32 saturation measurement, not
        # two-block Q64. Do not silently accept oversize legacy configurations.
        saturated = block == 32 and obj == cls(64, 0, 64, "hard_noedit")
        if (not 1 <= obj.budget <= block and not saturated) or min(obj.fresh, obj.mask, obj.edit) < 0:
            raise ValueError(f"Invalid Q or quota: {text}")
        modes = ("targeted", "random", "noedit", "hard_frontier", "hard_random", "hard_noedit",
                 "hard_slack", "hard_trigger") + (("near_miss", "gate_random", "hard_oracle1") if block == 32
                                                    else ("fresh_random",) if block == 256 else ())
        if selection not in modes:
            raise ValueError(f"Unknown selection: {selection}")
        if selection.startswith("hard_") and obj.fresh != 0:
            raise ValueError("Hard refresh selects every fresh row; no fixed fresh quota is allowed")
        if selection == "hard_oracle1" and obj.edit != 1:
            raise ValueError("The shadow oracle verifies exactly one candidate swap")
        if (selection in ("noedit", "fresh_random", "hard_noedit")) != (obj.edit == 0):
            raise ValueError("noedit/fresh_random require zero reserved edit seats; other arms reserve at least one")
        return obj


def matrix(model):
    # Same nominal Q within each triplet. No-edit returns the edit quota to MASK.
    # Budget controls the sparse steps, not initialization/check/commit.
    splits = ([(16, 4, 8), (8, 2, 4), (16, 8, 6)] if model == "llada21" else
              [(128, 32, 80), (64, 16, 40), (128, 64, 56)])
    arms = []
    for q, f, m in splits:
        for mode in ("targeted", "random", "noedit"):
            arm = Arm(q, f, q-f if mode == "noedit" else m, mode)
            if arm not in arms:
                arms.append(arm)
    return arms


def verify_bridge(row, reference):
    for k in ("index", "dataset", "task_id", "seed", "output_ids", "output_tokens",
              "generation_forwards", "generation_query_rows", "generation_physical_padded_rows",
              "max_q"):
        if row[k] != reference[k]:
            raise ValueError(f"Full-width parity failed on request {row['index']}: {k}")


def verify_work(row):
    n, f, q, phys = (row[k] for k in ("output_tokens", "generation_forwards",
                                       "generation_query_rows", "generation_physical_padded_rows"))
    if f <= 0 or q < f or q != phys or n < 0:
        raise ValueError(f"Unexpected unpadded work: {(n,f,q,phys)}")
    if abs(row["tpf"] - n / f) > 1e-10 or abs(row["token_efficiency"] - n / q) > 1e-10:
        raise ValueError("Work counters disagree with reported metrics")
