"""Render one compact CSV row per sealed triplet. No model, raw IDs or logits."""
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

D=Path(__file__).resolve().parent


def main():
    done=json.loads((D/"complete.json").read_bytes())
    rows=[json.loads(line) for line in (D/"compact.ndjson").read_text().splitlines()]
    canonical=json.dumps(rows,sort_keys=True,separators=(",",":"),allow_nan=False)
    if sha256(canonical.encode()).hexdigest()!=done["sha"] or [r["index"] for r in rows]!=done["cases"]:
        raise ValueError("Incomplete triplets")
    w=csv.writer(sys.stdout,lineterminator="\n")
    keys=("correct","N","F","Q","extra","paid_D","paid_Q")
    arms=("stock","repeat","current")
    w.writerow(["index","dataset","seed","first_tick","first_k","repeat_positions","current_positions",
                "first_output_offset",*(f"{a}_{k}" for a in arms for k in keys)])
    for r in rows:
        f=r["first"] or {}
        w.writerow([r["index"],r["dataset"],r["seed"],f.get("tick"),f.get("k"),
                    "/".join(map(str,f.get("repeat",[]))),"/".join(map(str,f.get("current",[]))),
                    r["first_output_offset"],*(r["stats"][a][k] for a in arms for k in keys)])


if __name__=="__main__":
    main()
