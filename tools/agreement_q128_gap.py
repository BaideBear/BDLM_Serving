"""Join certified A3 receipts to the same frozen questions in historical Live."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean


def compare(directory, certified, live):
    rows = []
    with live.open(newline="") as f:
        refs = list(csv.DictReader(f))
    if len(refs) != 90 or {int(r["index"]) for r in refs} != set(range(90)):
        raise ValueError("invalid Live cohort")
    verified = json.loads(certified.read_text())
    hashes = verified["hashes"]
    if verified["model"] != "gemma" or verified["fixed_q"] != 128:
        raise ValueError("unexpected paired reference model or Q")
    for r in refs:
        i = int(r["index"])
        path = directory / f"case-{i:02}.json"
        data = path.read_bytes()
        t = json.loads(data)
        if (sha256(data).hexdigest() != hashes[path.name] or t["index"] != i or
            (t["dataset"], t["task_id"], t["seed"], int(r["q"])) !=
            (r["dataset"], r["task_id"], int(r["seed"]), 128)):
            raise ValueError(f"unmatched case {i}")
        row = {"index": i, "dataset": r["dataset"]}
        row["live"] = bool(int(r["correct"])), int(r["N"]), int(r["on_F"]), int(r["on_decoder_Q"])
        for k in ("stock","apply"):
            a = t["arms"][k]
            row[k] = a["correct"], a["N"], a["bill"]["F"], a["bill"]["Q"]
        rows.append(row)

    def cohort(group):
        counts = {}
        for k in ("live","stock","apply"):
            ok,n,f,q = zip(*(r[k] for r in group),strict=True)
            counts[k] = dict(correct=sum(ok), N=sum(n), F=sum(f), Q=sum(q),
                             tpf=mean(x/y for x,y in zip(n,f,strict=True)), te=sum(n)/sum(q))
        return dict(n=len(group), counts=counts, patterns={
            "".join(str(int(v)) for v in bits):
                [r["index"] for r in group if tuple(r[k][0] for k in ("live","stock","apply"))==bits]
            for bits in sorted({tuple(r[k][0] for k in ("live","stock","apply")) for r in group})
        })

    return dict(live_reference_sha256=sha256(live.read_bytes()).hexdigest(),
                certified_sha256=sha256(certified.read_bytes()).hexdigest(),
                bit_order="historical Live, A3 IBS, A3 repeat; device/stacks not matched across Live vs A3",
                all90=cohort(rows), datasets={d:cohort([r for r in rows if r["dataset"] == d])
                                               for d in sorted({r["dataset"] for r in rows})})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("directory","certified","live"):
        p.add_argument(name,type=Path)
    print(json.dumps(compare(**vars(p.parse_args())),indent=2))
