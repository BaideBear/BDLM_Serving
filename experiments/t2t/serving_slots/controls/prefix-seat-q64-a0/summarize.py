"""Validate and compact the six-case A0 prefix-seat receipts; no inference."""
import argparse
import hashlib
import json
from pathlib import Path

SIX = (7, 13, 25, 33, 55, 71)
ARMS = ("stock", "observer", "seat")


def summarize(root: Path):
    summary = json.loads((root / "summary.json").read_bytes())
    if (summary["cases"] != list(SIX) or summary["arms"] != list(ARMS)
            or summary["gate"] != "passed" or summary["failed"]):
        raise ValueError("Incomplete or failed prefix-seat run")
    rows, digests = [], {}
    for i in SIX:
        path = root / f"case-{i:02}.json"
        payload = path.read_bytes()
        digests[path.name] = hashlib.sha256(payload).hexdigest()
        r = json.loads(payload)
        if r["index"] != i or r["error"] or not r["stock_observer_identical"]:
            raise ValueError(f"Invalid control pair at {i}")
        arms = {k: r["arms"][k] for k in ARMS}
        seat = arms["seat"]["seat"] or {"exchanges": []}
        exchanges = seat["exchanges"]
        if exchanges and (not r["first_fork"] or r["first_fork"]["kind"] != "trace"
                          or r["first_fork"]["tick"] != exchanges[0]["tick"]):
            raise ValueError(f"Seat fork mismatch at {i}")
        if not exchanges and r["first_fork"] is not None:
            raise ValueError(f"Unexposed seat diverged at {i}")
        row = dict(index=i, task_id=r["task_id"], dataset=r["dataset"], seed=r["seed"],
                   first_fork=r["first_fork"], exchanges=exchanges,
                   blocks_used=seat.get("blocks_used", []))
        for arm, v in arms.items():
            row[arm] = dict(N=v["N"], F=v["bill"]["F"], decoder_Q=v["bill"]["Q"],
                            encoder_Q=v["bill"]["encoder_Q"], correct=v["correct"],
                            scoring=v["scoring"])
        stock, probe = row["stock"], row["seat"]
        row["seat_delta"] = {k: probe[k] - stock[k] for k in ("N", "F", "decoder_Q")}
        row["correct_delta"] = int(probe["correct"]) - int(stock["correct"])
        rows.append(row)
    for name in ("summary.json", "raw.log"):
        path = root / name
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(source=root.name, sha256=digests, cases=rows)


if __name__ == "__main__":
    print(json.dumps(summarize(Path(__file__).parent if False else
                               argparse.ArgumentParser(description=__doc__).parse_args().root),
                      indent=2))
