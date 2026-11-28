"""Verify sealed raw receipts and export compact results without raw output IDs."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
import sys

CASES = (7, 13, 25, 33, 55, 71)
ARMS = ("repeat", "observer", "recheck")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(obj):
    return sha(json.dumps(obj, separators=(",", ":")).encode())


def check(ok, why):
    if not ok:
        raise ValueError(why)


def main(raw_path, reference, original):
    raw = raw_path.read_bytes()
    records = [json.loads(line[8:]) for line in raw.splitlines() if line.startswith(b"RECEIPT ")]
    start = records[0]
    check(start["kind"] == "start" and records[-1] == dict(kind="complete", cases=list(CASES))
          and start["cases"] == list(CASES) and start["q"] == 128, "Incomplete or wrong cohort")
    release, = [line.decode() for line in raw.splitlines() if line.startswith(b"RELEASE ")]
    source_hashes = {Path(path).name: checksum for checksum, path in
                     re.findall(r"(?m)^([0-9a-f]{64})  (/gs/work/\S+)$", raw.decode())}
    check(set(source_hashes) == {"gemma_paid_agreement_eval.py", "gemma_provisional_recheck.py",
                                 "launch.sh", "run.py"}, "Missing launch source pins")
    check({"rc=0", f"gpu={start['gpu']}", "memory_mib=0", "apps=''", "locks=closed"} <= set(release.split()),
          "Missing idle release")
    compact, totals, tpf = [], {k: Counter() for k in ARMS}, {k: [] for k in ARMS}
    old_hits = 0
    for j, i in enumerate(CASES):
        seq = records[1+4*j:5+4*j]
        stages, row = seq[:3], seq[3]
        check(row["kind"] == "case" and row["index"] == i and row["order"] ==
              [s["arm"] for s in stages] and set(row["order"]) == set(ARMS)
              and all(s["kind"] == "stage" and s["index"] == i for s in stages), f"Stages for #{i}")
        hist_path = reference / f"case-{i:02}.json"
        old_path = original / hist_path.name
        check(sha(hist_path.read_bytes()) == start["references"][str(i)], "Historical receipt hash")
        hist, old = [json.loads(p.read_bytes()) for p in (hist_path, old_path)]
        check(hist["arms"]["apply"] == old["arms"]["apply"], "Observer does not reconcile original repeat")
        arms = row["arms"]
        a, b, c = [arms[k] for k in ARMS]
        check(all(a[k] == b[k] for k in ("ids", "trace", "decoder", "encoder", "rng_after", "bill", "N",
                                        "correct", "trigger", "payment")), f"Observer changed #{i}")
        check(all(a[k] == hist["arms"]["apply"][k] for k in
                  ("bill", "trigger", "payment", "correct", "scoring", "N", "rng_after"))
              and [m for m in a["agreement"]["observations"] if m["extra"]] ==
              hist["arms"]["apply"]["agreement"]["extra_events"], "Repeat history mismatch")
        check(b["measurements"] == hist["payment_observer"]["events"], "First-payment history mismatch")
        old_hits += sum(m["role"] == "extra" and m["ordinary"] and m["old"] != m["top"] and m["top_p"] > .9
                        for m in b["measurements"])
        out = dict(index=i, order=row["order"], original_reference_sha256=sha(old_path.read_bytes()),
                   ids_recheck_equal=a["ids"] == c["ids"], arms={}, corrections=[],
                   terminal_payments=c["terminal_payments"])
        for k, v in arms.items():
            bill = v["bill"]
            widths = [t["q"] for t in v["trace"]]
            check(widths[0] == 128 and all(0 < w <= 128 for w in widths)
                  and bill["Q"] == sum(widths) and bill["D"] == len(widths) == len(v["decoder"])
                  and bill["E"] == len(v["encoder"])-1 and bill["F"] == bill["D"]+bill["E"], f"Bill #{i}/{k}")
            for tick, turn in enumerate(v["decoder"]):
                if turn["sample"] is None:
                    after = v["decoder"][tick+1]["rng"] if tick+1 < len(widths) else v["rng_after"]
                    check(turn["rng"] == after and (k == "repeat" or
                          v["same_step"][tick]["rng"] == after and not v["same_step"][tick]["positions"]),
                          f"Pure payment sampled or changed RNG: #{i}/{k}/{tick}")
            out["arms"][k] = dict(bill=bill, N=v["N"], correct=v["correct"], ids_sha256=digest(v["ids"]),
                                  trigger=v["trigger"], payment=v["payment"])
            totals[k].update(bill)
            totals[k].update(N=v["N"], correct=int(v["correct"]))
            tpf[k].append(v["N"]/bill["F"])
        first = row["first"]
        if first:
            tick = first["tick"]
            check(b["decoder"][:tick+1] == c["decoder"][:tick+1] and b["trace"][:tick] == c["trace"][:tick]
                  and b["same_step"][:tick+1] == c["same_step"][:tick+1], "First sampled step drift")
            out["first"] = dict(tick=tick, width=first["q"], positions=first["positions"],
                                sampled_rows=len(first["post_sample"]["positions"]),
                                identical_base_sample_and_rng=True)
        elif not c["corrections"]:
            check(b == c, "Unexposed treatment differed")
        else:
            check(False, "Correction without recorded fork")
        required = [m for m in c["measurements"] if m["role"] == "extra" and m["ordinary"] and
                    m["old"] != m["top"] and m["top_p"] > .9]
        check(required == [{k: e[k] for k in m} for m, e in zip(required, c["corrections"])]
              and len(required) == len(c["corrections"]), "Eligible correction missing or changed")
        for e in c["corrections"]:
            p, payment, repayment = e["pos"], e["payment"], e["repayment"]
            check(e["role"] == "extra" and e["ordinary"] and e["top_p"] > .9 and e["old"] != e["top"]
                  and payment < repayment and p in c["states"][payment]["debt"] and
                  p not in c["states"][repayment]["debt"], "Correction lacks valid dirty lifecycle")
            for tick, token in ((payment, e["old"]), (repayment, e["top"])):
                pos = c["trace"][tick]["positions"]
                check(c["decoder"][tick]["ids"][pos.index(p)] == token, "Wrong physical payment ID")
            out["corrections"].append(dict(e, payment_Q=c["trace"][payment]["q"],
                                          repayment_Q=c["trace"][repayment]["q"]))
        check(len({e["pos"] for e in out["corrections"]}) == len(out["corrections"]), "Repeated correction")
        compact.append(out)
    check(len(records) == 2+4*len(CASES), "Unexpected extra or unverified stages")
    summary = {}
    for k, v in totals.items():
        summary[k] = {name: value for name, value in v.items() if name != "peak_decoder_Q"}
        summary[k].update(macro_TPF=mean(tpf[k]), pooled_TE=v["N"]/v["Q"], peak_decoder_Q=128)
    print(json.dumps(dict(schema="gemma-provisional-recheck-six-v1", start=start, release=release,
                          source_sha256=source_hashes,
                          raw_sha256=sha(raw), repeat_candidates=old_hits, totals=summary,
                          cases=compact), indent=2))


if __name__ == "__main__":
    main(*(Path(x) for x in sys.argv[1:]))
