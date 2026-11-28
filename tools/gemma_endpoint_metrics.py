"""Export compact, paired fixed-Q256 metrics from the qualified H0 receipt stream."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
import sys


def check(ok, message):
    if not ok:
        raise ValueError(message)


def export(root, level):
    stream = (root/"receipts.ndjson").read_bytes()
    check(sha256(stream).hexdigest() == "b95a7a3aec58ac20e2abe98b7ef5f4932e0a643674c6ac9469bc26096a1bdb02",
          "Unexpected input receipt stream")
    rows = sorted(map(json.loads, stream.splitlines()), key=lambda x: x["index"])
    check([r["index"] for r in rows] == list(range(90)) and
          all(all(r["matches"].values()) and r["paid_prefix"] and
              r["ids_sha"]["step"] == r["ids_sha"]["native"] and
              r["encoder_sha"]["step"] == r["encoder_sha"]["native"] for r in rows), "Invalid endpoint")
    if level == "pairs":
        for r in rows:
            for arm,m in sorted(r["arms"].items()):
                yield dict(index=r["index"], dataset=r["dataset"], arm=arm, correct=int(m["correct"]),
                           N=m["N"], F=m["F"], D=m["D"], D_Q=m["D_Q"], E=m["E"], E_Q=m["E_Q"],
                           paid_ids_equal=r["paid_ids_equal"], ids_sha=r["ids_sha"][arm], raw_sha=r["raw_sha"])
        return
    for block, group in (("original30",rows[:30]),("new60",rows[30:]),("all90",rows)):
        for arm in ("native","step","paid"):
            ms = [r["arms"][arm] for r in group]
            sums = {k:sum(m[k] for m in ms) for k in ("N","F","D_Q","E","E_Q","correct")}
            yield dict(cohort=block,arm=arm,n=len(ms),**sums,TPF=mean(m["N"]/m["F"] for m in ms),
                       TE=sums["N"]/sums["D_Q"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("level",choices=("pairs","cohorts"))
    a = p.parse_args()
    data = list(export(a.root, a.level))
    writer = csv.DictWriter(sys.stdout, fieldnames=list(data[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(data)
