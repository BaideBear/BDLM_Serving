"""Shared frozen30 protocol: one case list, explicit budgets, no free closure work."""
from dataclasses import dataclass
from .exchange import ONLINE_MODES as EXCHANGE
from .context_gate import CONTEXT, CROWDED, FIXED_CONTEXT, GATES
from .frontier import GEOMETRY_POLICIES, HARD_POLICIES
from .physical_prefix import GEMMA_ACTION_LOCK, PREFIX_MODES, prefix_quota
from .seat_audit import GEMMA_CONDITIONAL, GEMMA_FACTORIAL, MODES


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

    def prefix_executable(self, block):
        if self.selection in PREFIX_MODES and (block, self.budget) == (32, 64):
            return Arm(block, self.fresh, block - self.fresh - self.edit, self.selection)
        return self

    @classmethod
    def parse(cls, text, block):
        q, f, m, selection = text.split(":")
        obj = cls(int(q), int(f), int(m), selection)
        # An explicitly labelled, single-block B32 saturation measurement, not
        # two-block Q64. Do not silently accept oversize legacy configurations.
        saturated = block == 32 and obj == cls(64, 0, 64, "hard_noedit")
        if selection in GEMMA_ACTION_LOCK:
            if block != 256 or obj.budget >= block:
                raise ValueError("Gemma prefix action-lock requires a sub-block budget on B256")
            prefix_quota(block, obj.budget, obj.fresh, obj.mask, GEMMA_ACTION_LOCK[selection])
            return obj
        if selection in PREFIX_MODES:
            if block not in (32, 256):
                raise ValueError("Physical prefix policies are supported on B32 and B256 only")
            effective = obj.prefix_executable(block)
            prefix_quota(block, effective.budget, effective.fresh, effective.mask, selection)
            return obj
        if (not 1 <= obj.budget <= block and not saturated) or min(obj.fresh, obj.mask, obj.edit) < 0:
            raise ValueError(f"Invalid Q or quota: {text}")
        modes = ("targeted", "random", "noedit", *HARD_POLICIES) + (("near_miss", "gate_random", "hard_oracle1", *GEOMETRY_POLICIES) if block == 32
                                                    else ("fresh_random",) if block == 256 else ())
        if selection in (*GEMMA_FACTORIAL, GEMMA_CONDITIONAL) and block != 256:
            raise ValueError("Candidate persistence factorial is Gemma B256 only")
        if selection in (*MODES, *GEMMA_FACTORIAL, GEMMA_CONDITIONAL):
            if (block, obj.budget) not in ((32, 4), (256, 32)) or obj.fresh or obj.edit:
                raise ValueError("One-seat studies use 4:0:4 (B32) or 32:0:32 (B256)")
            return obj
        if block == 32:
            modes += ("hard_hazard_gate", "hard_hazard_fixed", *CONTEXT, *EXCHANGE)
        elif block == 256:
            modes += ("hard_trigger_refresh",)
        if selection not in modes:
            raise ValueError(f"Unknown selection: {selection}")
        if selection.startswith("hard_") and obj.fresh != 0:
            raise ValueError("Hard refresh selects every fresh row; no fixed fresh quota is allowed")
        if selection in FIXED_CONTEXT and (obj.budget, obj.edit) not in ((4, 1), (8, 1), (16, 2)):
            raise ValueError("Context rollout is bounded to Q4/1, Q8/1, Q16/2")
        if selection in (*GATES, *CROWDED) and (obj.budget, obj.edit) != (8, 1):
            raise ValueError("Context gate rollout is bounded to Q8/1")
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
