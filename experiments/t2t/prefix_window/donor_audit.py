"""Check donor choice on the common trajectory and score paired-backfill against paired-MASK.

Run with --data /private/prefix90 --q 128 --out /private/donor-audit; publish aggregate CSVs only.
--opportunities-only reads the corresponding zero arm without waiting for new online arms.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import re

from tools.prefix_cohorts import score_files
from bdlm_serving.methods.t2t.protocol import verify_work
from . import exposure
from .score_run import snapshot


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def opportunity(request, policy, q):
    verify_work(request)
    by = Counter()
    for e in request["events"]:
        if e["stage"] != "selected":
            continue
        exposure.classify("gemma", 256, policy, request["index"], e)
        p = e["prefix_plan"]
        old = min(q // 4, len(p["eligible_prefix_rows"]), len(p["zero_backfill_rows"]))
        paired = min(old, max(0, len(p["zero_mask_rows"])-1))
        by.update(selected=1, paired_rows=paired, paired_steps=int(paired > 0),
                  backfill_rows=old, backfill_steps=int(old > 0))
    by.update(paired_requests=int(by["paired_rows"] > 0), backfill_requests=int(by["backfill_rows"] > 0))
    return by


def first_exchange(old, mask):
    require(all(old[k] == mask[k] for k in ("task_id", "index", "seed", "dataset", "model")),
            "Donor-pair task mismatch")
    for t, (a, b) in enumerate(zip(old["events"], mask["events"])):
        ap, bp = a.get("prefix_plan", {}), b.get("prefix_plan", {})
        if not (ap.get("added_rows") or bp.get("added_rows")):
            require(a == b, f"Pre-intervention event differs at {t}")
            continue
        require(a["stage"] == b["stage"] == "selected" and a["q"] == b["q"]
                and a["block"] == b["block"] and a["clock"] == b["clock"], "Different entry or width")
        for key in ("prefix_pre_canvas_sha256", "prefix_pre_draft_sha256"):
            require(a[key] == b[key] and re.fullmatch("[0-9a-f]{64}", a[key]), f"Bad common {key}")
        omit = {"evicted_rows", "selected_rows"}
        require({k:v for k,v in ap.items() if k not in omit} ==
                {k:v for k,v in bp.items() if k not in omit}, "Different common state or recipients")
        n = len(ap["added_rows"])
        require(n > 0 and ap["evicted_rows"] == ap["zero_backfill_rows"][-n:]
                and bp["evicted_rows"] == bp["zero_mask_rows"][1:][-n:]
                and len(ap["evicted_rows"]) == len(bp["evicted_rows"]) == n,
                "Wrong first donor classes")
        return True
    require({k:v for k,v in old.items() if k != "policy"} ==
            {k:v for k,v in mask.items() if k != "policy"}, "Unexposed request changed")
    return False


def run(data, q, out, opportunities_only=False):
    sel = {32: "right", 128: "random"}[q]
    out.mkdir(parents=True, exist_ok=True)
    private = data / ".donor-scores"
    cases_path = data / "samples.json"
    cases = exposure.load_cases(cases_path)
    digest = exposure.sha256(cases_path)
    zero = f"t2t_q{q}_f0_m{q}_hard_prefix_zero"
    ref = data / ("gemma-q32" if q == 32 else "gemma-grid-k1")
    folders = [snapshot(ref, zero, digest, private / ".inputs")]
    pids = [zero]
    if not opportunities_only:
        online = data / f"gemma-donor-{q}"
        for d in ("paired_backfill", "paired_mask"):
            p = f"t2t_q{q}_f0_m{q-q//4}_hard_prefix_{d}_{sel}"
            pids.append(p)
            folders.append(snapshot(online, p, digest, private / ".inputs"))
    totals = {cohort:Counter() for cohort, _ in exposure.COHORTS}
    for i in range(90):
        rs = [json.loads((folder / f"{i:02d}-{pid}.json").read_text())
              for folder, pid in zip(folders, pids, strict=True)]
        for r, p in zip(rs, pids, strict=True):
            require(r["policy"] == p and r["model"] == "gemma" and r["status"] == "success"
                    and all(r[k] == cases[i][k] for k in ("index","dataset","task_id","seed")),
                    f"Invalid request {i}")
        counts = opportunity(rs[0], zero, q)
        if not opportunities_only:
            for r in rs[1:]:
                verify_work(r)
                mode = exposure.parse_policy(r["policy"])["mode"]
                for e in r["events"]:
                    if e["stage"] == "selected":
                        exposure.classify("gemma", 256, r["policy"], i, e)
                        exposure.check_donor_recipients(e["prefix_plan"], mode, cases[i]["seed"])
            counts["online_requests_first_exchange_verified"] = int(first_exchange(*rs[1:]))
            require(counts["paired_requests"] == counts["online_requests_first_exchange_verified"],
                    f"Zero trajectory and captured first intervention disagree at {i}")
        for cohort, ids in exposure.COHORTS:
            if i in ids:
                totals[cohort].update(counts)
    write(out / f"gemma-donor-opportunity-{q}.csv", [
        dict(q=q, k=q//4, cohort=c, requests=len(ids), samples_sha256=digest, **totals[c])
        for c,ids in exposure.COHORTS])
    if not opportunities_only:
        scores, _, _ = score_files(folders[2], folders[1], cases_path, private / f"q{q}-direct.csv",
                                    baseline_policy=pids[2], arm_policy=pids[1])
        write(out / f"gemma-donor-direct-{q}.csv", scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--q", required=True, choices=(32, 128), type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--opportunities-only", action="store_true")
    args = parser.parse_args()
    run(args.data, args.q, args.out, args.opportunities_only)
