"""Publish the fixed K0 reference without selecting a best arm at each budget."""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fixed_rows(source):
    for model, slots in (("llada21", (4, 8, 16, 32)), ("gemma", (16, 32, 64, 128, 256))):
        for q in slots:
            path = source / f"{model}-q{q}-k1-right.csv"
            with path.open() as f:
                rows = [r for r in csv.DictReader(f) if r["scope"] == "all"]
            if ([int(r["requests"]) for r in rows] != [30, 60, 90] or any(
                    r["model"] != model or int(r["nominal_q"]) != q or
                    r["baseline_policy"] != f"t2t_q{q}_f0_m{q}_hard_prefix_zero" for r in rows)):
                raise ValueError(f"Unexpected fixed control: {path}")
            for field in ("correct","N","F","physical_Q"):
                vals = [int(r["baseline_"+field]) for r in rows]
                if sum(vals[:2]) != vals[2]:
                    raise ValueError(f"Disjoint cohorts disagree: {path} {field}")
            for r in rows:
                yield dict(model=model, slot=q, role="diagnostic" if (model,q)==("gemma",16)
                           else "primary", cohort=r["cohort"], requests=r["requests"],
                           **{k:r["baseline_"+k] for k in ("correct","macro_tpf","pooled_token_efficiency",
                                                            "N","F","physical_Q","q_per_f")})


if __name__ == "__main__":
    rows = list(fixed_rows(HERE.parent/"prefix_window"))
    with (HERE/"fixed-k0.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
