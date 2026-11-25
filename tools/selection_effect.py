"""Pair fully scored budget variants with no-edit and charge all shadow probes."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def read(p):
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def pair_rows(scores, baseline, run, model):
    lookup = {(int(r["nominal_q"]), int(r["index"])):r for r in baseline
              if r["model"] == model and r["policy"].endswith("_hard_noedit")}
    by = defaultdict(list)
    for r in scores:
        if r["policy"] != "native":
            by[r["policy"]].append(r)
    for policy, rows in by.items():
        assert {int(r["index"]) for r in rows} == set(range(30))
        n = f = q = same = wins = losses = 0
        hits = seats = df = dq = screens = screen_hits = 0
        tpf, ref_tpf, ideal_tpf = [], [], []
        ref_q = ref_f = ref_n = correct = ref_correct = 0
        diagnostic = policy.endswith("_hard_oracle1")
        for r in rows:
            idx = int(r["index"])
            b = lookup[int(r["nominal_q"]),idx]
            assert all(r[k] == b[k] for k in ("index","dataset","task_id"))
            data = json.loads((run/f"{idx:02d}-{policy}.json").read_text())
            assert data["index"] == idx and data["task_id"] == r["task_id"]
            base = (c := int(r["correct"])) - (bc := int(b["correct"]))
            wins += base > 0
            losses += base < 0
            same += base == 0
            correct += c
            ref_correct += bc
            ni, fi, qi = (int(r[k]) for k in ("N","F","Q"))
            bn, bf, bq = (int(b[k]) for k in ("N","F","Q"))
            assert (fi, qi) == (sum(data["forwards_by_phase"].values()), sum(data["q_by_phase"].values()))
            di = dqi = 0
            stages = Counter()
            for event in data["events"]:
                stage = event["stage"]
                if stage.startswith("oracle_"):
                    assert diagnostic and event["diagnostic"] and event["phase"] == "diagnostic"
                    assert event["actual_q"] == event["physical_q"] == len(event["absolute_positions"])
                    assert event["F"] == 1 and all(v == event["actual_q"] for v in event["layer_physical_q"])
                    di += 1
                    dqi += event["actual_q"]
                    screens += event["screened_candidates"] if stage == "oracle_union" else 0
                    screen_hits += event["screen_hits"] if stage == "oracle_union" else 0
                    stages[stage] += 1
                if stage == "selected":
                    es = event["reserved_extra_edit_positions"]
                    positives = event["reserved_strict_edit_positions"]
                    assert set(positives) <= set(es)
                    if diagnostic:
                        assert es == positives and len(es) <= 1
                    hits += len(positives)
                    seats += len(es)
            assert di == sum(data["forwards_by_phase"].get(s,0) for s in ("oracle_union","oracle_trial"))
            assert dqi == sum(data["q_by_phase"].get(s,0) for s in ("oracle_union","oracle_trial"))
            tpf.append(ni/fi)
            ref_tpf.append(bn/bf)
            ideal_tpf.append(ni/(fi-di))
            n += ni
            f += fi
            q += qi
            ref_n += bn
            ref_f += bf
            ref_q += bq
            df += di
            dq += dqi
        assert wins+losses+same == len(rows)
        measured, reference = mean(tpf), mean(ref_tpf)
        yield dict(model=model, policy=policy, nominal_q=r["nominal_q"], requests=len(rows),
                   correct=correct, reference_correct=ref_correct, only_policy=wins, only_reference=losses,
                   tpf=measured, reference_tpf=reference, delta_tpf=measured-reference,
                   quality_up_tpf_not_lower=int(correct>ref_correct and measured>=reference),
                   N=n, F=f, Q=q, q_per_f=q/f, reference_q_per_f=ref_q/ref_f,
                   token_efficiency=n/q, reference_token_efficiency=ref_n/ref_q,
                   reserved_seats=seats, reserved_strict_hits=hits, oracle_F=df, oracle_Q=dq,
                   union_candidates=screens, union_hits=screen_hits,
                   # Still this rollout's outputs, NOT a possible deployed rate.
                   idealized_tpf_without_probes=mean(ideal_tpf) if diagnostic else "",
                   idealized_efficiency_without_probes=n/(q-dq) if diagnostic else "")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("scores","baseline","run","out"):
        p.add_argument("--"+flag, type=Path, required=True)
    p.add_argument("--model", choices=["llada21","gemma"], required=True)
    args = p.parse_args()
    rows = list(pair_rows(read(args.scores), read(args.baseline), args.run, args.model))
    if not rows:
        raise ValueError("No completed non-native policies")
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
