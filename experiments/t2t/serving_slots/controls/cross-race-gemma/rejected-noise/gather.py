"""Run on the receipt host with the qualified code on PYTHONPATH. No inference."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from experiments.t2t.serving_slots.gemma_rejected_noise_eval import CONTROL, TIERS, check, pair, qualify
from experiments.t2t.serving_slots.gemma_rejected_noise_report import report


def write_csv(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ("root", "history", "out"):
        ap.add_argument("--" + flag, type=Path, required=True)
    args = ap.parse_args()
    qual = qualify(args.root, args.history)
    summaries, audited = [], []
    signatures, tasks = set(), {}
    for name, ids in (("frozen90", range(90)), ("original30", range(30)), ("new60", range(30, 90))):
        tiers, compact = report(args.root, ids, complete=True)
        if name == "frozen90":
            for row in compact:
                c, i = row["budget"], row["index"]
                a, b = pair(args.root, c, i)
                h = json.loads((args.history / f"C{c}-fuse0-{i:02d}.json").read_text())
                check(h["schema"] == "paid-close-v1" and h["budget"] == c and h["fuse"] is False
                      and h["gate"] == "off" and h["prefix"] == [1, 8] and h["slack"] is False,
                      f"Unmatched history C{c}/{i}")
                diff = [k for k in (*CONTROL, "torch", "transformers") if a[k] != h[k]]
                check(not diff, f"Off differs from history C{c}/{i}: {diff}")
                identity = tuple(a[k] for k in ("dataset", "task_id", "seed", "input_sha256"))
                check(tasks.setdefault(i, identity) == identity, f"Mixed identity at {i}")
                row.update(cohort="original30" if i < 30 else "new60", history_exact=1,
                           score_delta=int(b["correct"])-int(a["correct"]), ledger_sealed_arms=2)
                for arm, r in (("off", a), ("on", b)):
                    sig = {k: r[k] for k in ("source_sha256", "policy", "gpu_visible", "checkpoint",
                                              "torch", "transformers", "samples_sha256", "scorer_sha256")}
                    signatures.add(json.dumps(sig, sort_keys=True))
                    check(r["max_decoder_Q"] <= c and r["F"] == r["decoder_F"] + r["encoder_F"]
                          and sum(r["route_F"].values()) == r["decoder_F"]
                          and sum(r["route_Q"].values()) == r["decoder_Q"]
                          and r["route_F"].get("verify", 0) > 0
                          and r["route_F"].keys() <= {"sparse", "drain", "verify"},
                          f"Invalid physical ledger C{c}/{i}/{arm}")
                    hit = r["noise"]["first_exposure"] or {}
                    row.update({f"{arm}_TPF": round(r["N"]/r["F"], 8),
                                f"{arm}_first_block": hit.get("ordinal"),
                                f"{arm}_first_rows_count": len(hit.get("rows", [])),
                                f"{arm}_sparse_F": r["route_F"].get("sparse", 0),
                                f"{arm}_verify_F": r["route_F"]["verify"]})
                audited.append(row)
        for c in TIERS:
            t = tiers[c]
            grouped = [r for r in audited if r["index"] in ids and r["budget"] == c]
            expected = dict.fromkeys(("gsm8k", "math500", "humaneval"), len(ids)//3)
            check(Counter(r["dataset"] for r in grouped) == expected, f"Unbalanced {name}/C{c}")
            s = dict(cohort=name, budget=c, matched=t["matched"], expected=t["expected"],
                     history_exact=sum(r["history_exact"] for r in grouped), ids_changed=t["ids_changed"],
                     rescues=len(t["rescues"]), losses=len(t["losses"]))
            for arm in ("off", "on"):
                s.update({f"{arm}_{k}": v for k, v in t[arm].items()})
                for k in ("sparse_F", "verify_F"):
                    s[f"{arm}_{k}"] = sum(r[f"{arm}_{k}"] for r in grouped)
            s.update(delta_macro_TPF=t["delta_macro_TPF"], tpf_ci95_lo=t["TPF_CI95"][0],
                     tpf_ci95_hi=t["TPF_CI95"][1], mean_delta_F=(s["on_F"]-s["off_F"])/len(ids),
                     mean_delta_F_ci95_lo=t["F_delta_CI95"][0], mean_delta_F_ci95_hi=t["F_delta_CI95"][1])
            summaries.append(s)
    check(len(audited) == 270 and len(signatures) == 1 and len(set(tasks.values())) == 90,
          "Incomplete or mixed 90-case population")
    provenance = json.loads(signatures.pop())
    check(provenance["policy"] == dict(top_gate="off", head_first=False, prefix_order="right",
          reuse_paid_masks=False, fuse_first_verify=False, fullstop=None, prefix=[1, 8])
          and provenance["gpu_visible"] == "GPU-8c80a093-8254-7eb4-db21-6a1bbe2d5d83",
          "Different policy or device than the GPU6 registration")
    provenance.update(qualification=qual, history_fields=[*CONTROL, "torch", "transformers"],
                      sealed_arms=540, historical_exact=270, root=str(args.root), history=str(args.history))
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "paired.csv", audited)
    write_csv(args.out / "cohorts.csv", summaries)
    tmp = args.out / "provenance.tmp"
    tmp.write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n")
    tmp.replace(args.out / "provenance.json")


if __name__ == "__main__":
    main()
