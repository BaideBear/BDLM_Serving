"""Validate a neutral rerun against the archived noedit path and audit actual old-row refreshes."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

from tools.t2t_results import table


def main(run, control, out):
    meta = json.loads((run / "run.json").read_text())
    model = meta["model"]
    q = {"llada21": 4, "gemma": 32}[model]
    ref = f"t2t_q{q}_f0_m{q}_hard_noedit"
    fields = ("index", "task_id", "dataset", "seed", "output_ids", "output_tokens",
              "generation_forwards", "generation_query_rows", "generation_physical_padded_rows",
              "forwards_by_phase", "q_by_phase")
    rows, counts = [], {}
    for arm in ("base", "rank", "random"):
        policy = f"t2t_q{q}_f0_m{q}_hard_once_{arm}"
        if json.loads((run / f"complete-{policy}.json").read_text()) != dict(policy=policy, requests=30):
            raise ValueError("Only complete arms")
        counts[arm] = 0
        for i in range(30):
            row = json.loads((run / f"{i:02d}-{policy}.json").read_text())
            if arm == "base":
                old = json.loads((control / f"{i:02d}-{ref}.json").read_text())
                if any(old[k] != row[k] for k in fields) or len(old["events"]) != len(row["events"]):
                    raise ValueError(f"{model} neutral cannot reproduce reference on request {i}")
                for a, b in zip(old["events"], row["events"], strict=True):
                    if any(b.get(k) != v for k, v in a.items()):
                        raise ValueError(f"{model} changed neutral physical/decision event on {i}")
            anchor = [e["intervention"] for e in row["events"] if "intervention" in e]
            if not anchor:
                continue
            p, = anchor
            counts[arm] += 1
            selected_key = {"base":"victim", "rank":"candidate", "random":"random_candidate"}[arm]
            selected = p[selected_key]
            if selected not in p["executed"]:
                raise ValueError("Compared seat must actually be recomputed")
            dr, = [x for x in p["resident_l15"] if x["row"] == selected]
            if not all(math.isfinite(dr[k]) and dr[k] >= 0 for k in ("dK", "dV")):
                raise ValueError("Invalid drift value")
            rows.append(dict(model=model, arm=arm, index=i, dataset=row["dataset"],
                stratum=p["stratum"], clock=p["clock"], selected=selected,
                selected_is_old=int(selected not in set(p["masks"])), dK=dr["dK"], dV=dr["dV"],
                strict=int(selected in p["strict_rows"]), revoked=int(selected in p["revoked_rows"]),
                rank_random_collision=int(p["candidate"] == p["random_candidate"])))
    if len(set(counts.values())) != 1:
        raise ValueError("Unequal opportunity counts")
    out.mkdir(exist_ok=True, parents=True)
    if rows:
        table(out / f"{model}-row-refresh.csv", rows)
    summaries = []
    for arm in counts:
        for stratum in ("all", "slack", "mask"):
            sub = [x for x in rows if x["arm"] == arm and (stratum == "all" or x["stratum"] == stratum)]
            summaries.append(dict(model=model, arm=arm, stratum=stratum, requests=30, anchors=len(sub),
                old=sum(x["selected_is_old"] for x in sub), strict=sum(x["strict"] for x in sub),
                revoked=sum(x["revoked"] for x in sub),
                nonzero=sum(max(x["dK"], x["dV"]) > 0 for x in sub),
                median_dK=median(x["dK"] for x in sub) if sub else "",
                median_dV=median(x["dV"] for x in sub) if sub else "",
                collisions=sum(x["rank_random_collision"] for x in sub)))
    table(out / f"{model}-row-refresh-summary.csv", summaries)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "control", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    main(a.run, a.control, a.out)
