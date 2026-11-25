"""Publish all measured logit-free single-block policies vs their same-Q noedit control."""
from pathlib import Path

from tools.render_hard_study import ROOT, all_rows
from tools.t2t_results import table


def main():
    root = ROOT / "experiments/t2t"
    rows, overview = [], []
    for model, budgets in (("llada21",(4,8,16,32,64)), ("gemma",(16,32,64,128,256))):
        measured = {}
        for d in ("hard", "adaptive", "placement", "placement/geometry-online"):
            path = root / d / f"{model}-online-summary.csv"
            if not path.exists():
                continue
            for r in all_rows(path):
                name = r["policy"]
                if "_hard_" not in name or name.endswith("oracle1"):
                    continue
                if name in measured or int(r["requests"]) != 30:
                    raise ValueError(f"Duplicate or incomplete {name}")
                measured[name] = r
        for q in budgets:
            base = measured[f"t2t_q{q}_f0_m{q}_hard_noedit"]
            candidates = []
            for name, r in measured.items():
                if int(r["nominal_q"]) != q:
                    continue
                mode = name[name.index("hard_"):]
                delta = int(r["correct"])-int(base["correct"])
                dt = float(r["tpf"])-float(base["tpf"])
                datum = dict(model=model, q=q, policy=name, mode=mode, old_cap=int(r["edit"]),
                    correct=int(r["correct"]), tpf=float(r["tpf"]), efficiency=float(r["token_efficiency"]),
                    actual_Q_F=float(r["q_per_f"]), N=int(r["N"]), F=int(r["F"]), Q=int(r["physical_Q"]),
                    delta_correct=delta, delta_TPF=dt, delta_Q=int(r["physical_Q"])-int(base["physical_Q"]))
                rows.append(datum)
                if mode != "hard_noedit" and "placebo" not in mode and "random" not in mode and delta>0 and dt>=0:
                    candidates.append(datum)
            winner = sorted(candidates, key=lambda r:(-r["correct"],-r["tpf"],r["Q"],r["policy"]))[:1]
            overview.append(dict(model=model, nominal_q=q,
                control_correct=base["correct"], control_tpf=base["tpf"], control_eff=base["token_efficiency"],
                control_Q=base["physical_Q"], both_up_count=len(candidates),
                illustrative_policy=winner[0]["policy"] if winner else "",
                correct=winner[0]["correct"] if winner else "", tpf=winner[0]["tpf"] if winner else "",
                efficiency=winner[0]["efficiency"] if winner else "", physical_Q=winner[0]["Q"] if winner else "",
                note="single B32 saturation; see dual64/ for real two-block Q64" if (model,q)==("llada21",64)
                else "single block saturated" if (model,q) in (("llada21",32),("gemma",256)) else ""))
    table(root/"budget-index.csv", rows)
    table(root/"budget-verdicts.csv", overview)


if __name__ == "__main__":
    main()
