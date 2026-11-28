"""CPU-only: freeze attempt-02 on n164. No generation, scoring, or GPU imports."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory


def main(root):
    src = root / "llada-ibs4-frontier-swap-a2-v1"
    attempt = src / "attempt-02"
    dest = root / "llada-ibs4-frontier-swap-a2-partial19-v1"
    sys.path.insert(0, str(src))
    from ibs4_frontier_report import report, VERSION
    diag = report(attempt, diagnostic=True)
    ids = list(range(15)) + [20, 30, 50, 70]
    assert not dest.exists() and diag["sealed"] == {"pilot": True, "remaining": False}
    assert diag["verified_bare_noop_ids"] == ids
    assert diag == json.loads((attempt / "same-card.diagnostic.19.json").read_bytes())
    with TemporaryDirectory(dir=root, prefix=".partial19-") as tmp:
        pkg = Path(tmp) / dest.name
        receipts = pkg / "attempt-02"
        tools = pkg / "tools"
        receipts.mkdir(parents=True)
        tools.mkdir()
        for name in ("registration.json", "pilot.ok", "same-card.diagnostic.06.json", "same-card.diagnostic.19.json",
                     "pilot.verify", "remaining.verify", "pilot.stderr", "remaining.stderr",
                     "pilot.receiver.stderr", "remaining.receiver.stderr"):
            shutil.copyfile(attempt / name, receipts / name)
        for name in ("ibs4_frontier_collect.py", "ibs4_frontier_swap_a2.py", "ibs4_frontier_report.py",
                     "ibs_frontier_exposure.py", "run_ibs4_frontier_a2.sh", "relay_ibs4_frontier_a2.same-card.sh"):
            shutil.copyfile(src / name, tools / name)
        for name in ("seal.py", "README.md"):
            shutil.copyfile(Path(__file__).with_name(name), pkg / name)
        pairs = []
        for i in ids:
            for suffix in ("json.gz", "metrics.json"):
                name = f"{i:02}.{suffix}"
                shutil.copyfile(attempt / name, receipts / name)
            with gzip.open(receipts / f"{i:02}.json.gz") as f:
                pkt = json.load(f)
            m = json.loads((receipts / f"{i:02}.metrics.json").read_bytes())
            a, b, c = (pkt["raw"][k] for k in ("bare", "noop", "swap"))
            assert pkt["registration"] == VERSION and pkt["index"] == i and a == b
            u, v = a["output_ids"], c["output_ids"]
            first = next((j for j, xy in enumerate(zip(u, v)) if xy[0] != xy[1]),
                         None if len(u) == len(v) else min(len(u), len(v)))
            assert (first is None) == (m["bare"]["ids"] == m["swap"]["ids"])
            row = dict(index=i, dataset=m["dataset"], cohort="original30" if i < 30 else "new60",
                       bare_noop_equal=True, output_ids_differ=u != v, first_output_difference_0based=first,
                       bare_output_digest=m["bare"]["ids"], swap_output_digest=m["swap"]["ids"],
                       bare_id_count=len(u), swap_id_count=len(v), swap_first_tick=m["first"]["tick"],
                       bare_eligible=m["eligible"][0], swap_eligible=m["eligible"][1])
            for key in ("correct", "N", "F", "Q", "eos", "truncated", "edits", "prefill_Q", "peak"):
                for arm in ("bare", "swap"):
                    row[f"{arm}_{key}"] = m[arm][key]
                row[f"delta_{key}"] = m["swap"][key] - m["bare"][key]
            row.update(bare_status=m["bare"]["status"], swap_status=m["swap"]["status"])
            pairs.append(row)
        with (pkg / "pairs.csv").open("x") as f:
            w = csv.DictWriter(f, fieldnames=list(pairs[0]))
            w.writeheader()
            w.writerows(pairs)
        groups = {}
        risk = set(json.loads((receipts / "registration.json").read_bytes())["historical_progress_failures"])
        for group, total in (("all_observed", 90), ("original30", 30), ("new60", 60),
                             ("historical_risk6", 6), ("historical_other84", 84)):
            s = diag["same_card"][group]
            gs = [r for r in pairs if group in ("all_observed", r["cohort"],
                  "historical_risk6" if r["index"] in risk else "historical_other84")]
            assert len(gs) == s["n"]
            groups[group] = dict(frozen_denominator=total, output_ids_differ=sum(r["output_ids_differ"] for r in gs), **s)
        (pkg / "cohorts.json").write_text(json.dumps(groups, indent=2) + "\n")
        assert report(receipts, diagnostic=True) == diag
        files = sorted(p for p in pkg.rglob("*") if p.is_file())
        (pkg / "SHA256SUMS").write_text("".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(pkg)}\n" for p in files))
        for p in pkg.rglob("*"):
            p.chmod(0o500 if p.is_dir() else 0o400)
        os.rename(pkg, dest)
        dest.chmod(0o500)
    print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
