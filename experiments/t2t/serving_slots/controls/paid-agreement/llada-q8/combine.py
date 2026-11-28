"""Combine sealed summaries without counting the A1 endpoint as an arm."""
import argparse
import json
from pathlib import Path

from study import ARMS, DEV, REST, aggregate, check


def read(path):
    return json.loads(path.read_bytes())


def cohort(cases):
    return dict(n=len(cases), indices=[c["index"] for c in cases], totals=aggregate(cases),
                answer_gains=[c["index"] for c in cases
                              if c["arms"]["apply"]["correct"] > c["arms"]["stock"]["correct"]],
                answer_losses=[c["index"] for c in cases
                               if c["arms"]["apply"]["correct"] < c["arms"]["stock"]["correct"]],
                same_output_ids=sum(c["arms"]["apply"]["ids_sha"] == c["arms"]["stock"]["ids_sha"]
                                    for c in cases),
                cases_exposed=sum(c["fork"] is not None for c in cases))


def combine(six, remaining):
    check(six["expand"] and all(six["gate"].values()), "Development gate not sealed")
    x, y = six["registration"], remaining["registration"]
    for k in ("code", "checkpoint", "gpu", "q", "samples", "policy"):
        check(x[k] == y[k], "Stage provenance differs: " + k)
    dev, rest = six["cases"], remaining["cases"]
    check(tuple(c["index"] for c in dev) == DEV and tuple(c["index"] for c in rest) == REST,
          "Incorrect development or non-development split")
    for cases, slip in ((dev, six), (rest, remaining)):
        check(aggregate(cases) == slip["totals"] and slip["gate"]["audit"], "Wrong sealed bill")
        check(set(slip["totals"]) == set(ARMS), "Wrong comparison arms")
    return {
        "status": "sealed90",
        "development_six": cohort(dev),
        "non_development_84": cohort(rest),
        "all90": cohort(sorted([*dev, *rest], key=lambda c: c["index"])),
        "non_development_original_27": cohort([c for c in rest if c["index"] < 30]),
        "non_development_added_57": cohort([c for c in rest if c["index"] >= 30]),
        "gpu": x["gpu"],
        "pids": {"development": x["pid"], "non_development": y["pid"]},
        "source": x["code"],
        "raw_sha256": {"development": six["raw_sha256"], "non_development": remaining["raw_sha256"]},
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("six", type=Path)
    p.add_argument("remaining", type=Path)
    a = p.parse_args()
    print(json.dumps(combine(read(a.six), read(a.remaining)), indent=2, sort_keys=True))
