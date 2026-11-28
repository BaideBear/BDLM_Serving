"""Audit a paired 18-case screen and select only complete per-model panels."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
from math import prod
from pathlib import Path

from experiments.t2t.serving_slots.shared_fixed_shard import identity, shard_cases
from experiments.t2t.serving_slots.shared_point_metrics import BLOCK, _cohort, _receipt
from tools.frozen90_six_shards import BOARDS, MANIFEST, OFFERS
from tools.shared_anchor_report import costs, read_receipts, source_paths
from tools.shared_screen_sweep import CHOICES, build


def choose(scores, control):
    """Compare exact macro TPF; the product ranks geometric means for five equal tiers."""
    base = scores[control]
    eligible = {}
    for dose, points in scores.items():
        if dose == control:
            continue
        diffs = [(points[q][0] - ok, points[q][1] - tpf) for q, (ok, tpf) in base.items()]
        if all(a >= 0 and b >= 0 for a, b in diffs) and any(a or b for a, b in diffs):
            eligible[dose] = points
    if not eligible:
        return dict(selected=control, qualified=[], reason="retain_control")
    if len(eligible) == 1:
        return dict(selected=next(iter(eligible)), qualified=list(eligible), reason="sole_qualifier")
    if any(not tpf for _, tpf in base.values()):
        return dict(selected=None, qualified=list(eligible), reason="undefined_control_TPF_ratio")

    def rank(dose):
        ratios = [eligible[dose][q][1] / base[q][1] for q in base]
        return min(ratios), prod(ratios), sum(c for c, _ in eligible[dose].values()), -Fraction(dose)

    return dict(selected=max(eligible, key=rank), qualified=list(eligible), reason="registered_tiebreak")


def summarize_model(cells, model, control):
    scores, table = {}, {}
    for dose in CHOICES:
        scores[dose], table[dose] = {}, {}
        for q in OFFERS[model]:
            by_id, ref = cells[q, dose], cells[q, control]
            if len(by_id) != 18 or by_id.keys() != ref.keys():
                raise ValueError("unpaired screen")
            rows = [by_id[i] for i in sorted(by_id)]
            totals = _cohort(rows)
            exact_tpf = sum(Fraction(r["N"], r["F"]) for r in rows) / len(rows)
            scores[dose][q] = totals["correct"], exact_tpf
            charge = Counter()
            for r in rows:
                charge.update(costs(r))
            table[dose][q] = dict(**totals, physical_Q_by_role=dict(charge),
                                  gained_correct=sum(by_id[i]["correct"] and not ref[i]["correct"] for i in by_id),
                                  lost_correct=sum(ref[i]["correct"] and not by_id[i]["correct"] for i in by_id),
                                  mean_actual_Q=totals["Q"] / sum(r["decoder_F"] for r in rows))
    return dict(decision=choose(scores, control), offers_by_dose=table)


def audit(root, samples, base, panel, anchor_root=None):
    axis = panel["axis"]
    fixed = [panel["arms"][m]["0"]["prefix_share"] for m in OFFERS] if axis == "open" else []
    if panel != build(base, axis, *fixed):
        raise ValueError("panel does not match the base registration, wrapper or arm configuration")
    version, source = identity(panel), identity(base)
    subsets = {b: {r["index"]: r for r in shard_cases(samples, b, "screen18")} for b in BOARDS}
    if any(len(ids) != 3 for ids in subsets.values()) or len(set().union(*subsets.values())) != 18:
        raise ValueError("invalid per-board calibration assignment")
    control = "1/8" if axis == "prefix" else "1/4"
    result = dict(version=version, source=source, axis=axis, control=control, target=720, seen=0, models={})
    anchor = None if anchor_root is None else read_receipts(anchor_root, samples, base)[1]
    for model, offers in OFFERS.items():
        cells, boards, deferred = defaultdict(dict), Counter(), []
        for board, subset in subsets.items():
            paths = source_paths(root, board, model)
            if len(paths) > 1:
                raise ValueError(f"ambiguous ledger for {board}/{model}")
            for path in paths:
                with path.open("rb") as f:
                    for line_no, line in enumerate(f, 1):
                        if not line.endswith(b"\n"):
                            deferred.append(f"{board}/{model}:line{line_no}")
                            break
                        try:
                            r = json.loads(line)
                            idx, q, dose = r["index"], r["offer"], r["dose"]
                            if (type(idx) is not int or idx not in subset or type(q) is not int or q not in offers
                                    or dose not in CHOICES or idx in cells[q, dose]
                                    or (r["stage"], r["axis"], r["version"], r["source"], r["model"], r["board"])
                                    != ("screen18", axis, version, source, model, board)
                                    or r["params"] != panel["arms"][model][dose]
                                    or any(r[k] != subset[idx][k] for k in ("dataset", "task_id", "seed", "prompt_sha256"))):
                                raise ValueError("wrong identity, duplicate, arm, or registration")
                            if not {"max_decoder_Q", "decoder_Q", "encoder_Q"} <= r.keys():
                                raise ValueError("missing physical charge")
                            _receipt(r, model, q)
                            roles = costs(r)
                            if (type(r["encoder_Q"]) is not int or r["encoder_Q"] != BLOCK[model] * r["encoder_F"]
                                    or r["Q"] < r["max_decoder_Q"] + r["decoder_F"] - 1
                                    or (int(q * Fraction(r["params"]["prefix_share"])) == 0 and roles["prefix"])):
                                raise ValueError("impossible encoder, decoder peak or prefix charge")
                            cells[q, dose][idx] = r
                            boards[board] += 1
                        except (TypeError, KeyError, ValueError) as e:
                            raise ValueError(f"{path}:{line_no}: {e}") from e
        seen = sum(boards.values())
        out = dict(seen=seen, target=360, by_board={b: boards[b] for b in BOARDS}, deferred=deferred)
        if anchor is not None:
            replay = dict(expected=0, matched=0, discrepancies=[])
            if panel["arms"][model][control] == base["params"][model]:
                replay["expected"] = 90  # 18 paired cases at five offers.
                for q in offers:
                    for idx, row in cells[q, control].items():
                        ref = anchor.get((model, q, idx))
                        if ref is None:
                            continue
                        replay["matched"] += 1
                        diffs = [k for k in ("status", "correct", "N", "F", "Q", "decoder_F", "encoder_F",
                                             "decoder_Q", "max_decoder_Q", "roles") if ref[k] != row[k]]
                        if diffs:
                            replay["discrepancies"].append(dict(index=idx, offer=q, fields=diffs))
            out["anchor_replay"] = replay
        if seen == 360 and not deferred:
            out.update(summarize_model(cells, model, control))
        result["models"][model] = out
        result["seen"] += seen
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "base", "panel"):
        p.add_argument(field, type=Path)
    p.add_argument("--require-complete", action="store_true")
    p.add_argument("--anchor-root", type=Path, help="compare like-for-like control receipts to formal90")
    args = p.parse_args()
    report = audit(args.root, json.loads(MANIFEST.read_bytes())["samples"],
                   json.loads(args.base.read_bytes()), json.loads(args.panel.read_bytes()), args.anchor_root)
    if args.require_complete and any("decision" not in m for m in report["models"].values()):
        p.error("incomplete dose screen; no final release")
    if args.require_complete and args.anchor_root:
        for m in report["models"].values():
            rep = m["anchor_replay"]
            if rep["matched"] != rep["expected"] or rep["discrepancies"]:
                p.error("missing or inconsistent control replay; inspect without --require-complete")
    print(json.dumps(report, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
