"""Validate the complete A3 stream on n164 and export token-free aggregates."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import sys

from prepare import CASES, prepare
from seat import B, FULL, Q, check, digest, sha256

GPU = "GPU-94c61e79-d80b-cae2-d85f-6bd2a2c99a1c"
FIELDS = ("ids", "events", "route_F", "physical_positions", "encoder",
          "projection_widths_verified", "terminal_eos", "physical", "D", "E", "F", "Q", "N", "correct")


def audit_arm(arm, notes, treat, label):
    check(not arm["error"] and arm["projection_widths_verified"] == 85, (label, "physical hooks"))
    es = arm["events"]
    noted = {n["tick"]: n for n in notes}
    check(len(noted) == len(notes) == len({n["head"] for n in notes}), (label, "duplicate exchange"))
    applied = set()
    seen, accepted, dirty = set(), set(), set()
    previous_drain = False
    age = h = 0
    origin = arm["physical"][0][1]
    ledger = [["encoder", origin]]
    widths, physical_positions = [], []
    for t, e in enumerate(es):
        scope = (label, t)
        p, = e["parts"]
        rows, sample, prefix = [set(p[k]) for k in ("rows", "sampled", "prefix")]
        width = len(rows)
        base = origin+B*h
        absolute = [base + x for x in sorted(rows)]
        unresolved = FULL - accepted
        check(rows <= FULL and p["ordinal"] == e["ordinal"] == h and e["tick"] == t
              and e["offer"] == Q and 0 < width <= Q and e["q"] == e["decoder_Q"] == width
              and e["positions"] == absolute and e["history_Q"] == width and e["decoder_F"] == 1
              and e["opened"] is None and e["promoted"] is None and dirty <= rows
              and e["head_initialized_before"] == len(seen) and len(unresolved) == e["head_mask_before"]
              and p["dirty_before"] == sorted(dirty)
              and e["dirty_before_absolute"] == [base+x for x in sorted(dirty)]
              and p["age_before"] == age and p["age_after"] == age+bool(sample)
              and e["sampled_absolute"] == [base+x for x in sorted(sample)]
              and prefix <= sample <= rows, (scope, "invalid physical pre-state"))
        stock_rows, stock_prefix = rows, prefix
        note = noted.get(t)
        if treat and note:
            stock_rows = (rows - {note["take"]}) | {note["drop"]}
            stock_prefix = prefix | {note["drop"]}
            check(note["drop"] not in rows and note["take"] in rows and rows == sample,
                  (scope, "missing measured exchange"))
        available = (unresolved & seen) - stock_rows - dirty
        donors = stock_prefix - dirty
        eligible = (previous_drain and seen == FULL and e["route"] == "sparse"
                    and donors and available and min(unresolved) in stock_rows and h not in applied)
        check(bool(eligible) == bool(note), (scope, "missed or spurious trigger"))
        if note:
            check(not dirty and note["debt"] == [] and note["previous_tick"] == t-1
                  and all(x < min(unresolved) for x in donors)
                  and (note["head"], note["drop"], note["take"], note["frontier"], note["width"])
                  == (h, min(donors), min(available), min(unresolved), Q)
                  and (note["encoder_end"], note["visible_start"], note["visible_end"], note["age"])
                  == (base, base, base+B, age)
                  and note["stock_rightmost_query"] == max(stock_rows)
                  and note["trial_rightmost_query"] == max((stock_rows-{note["drop"]})|{note["take"]}),
                  (scope, "seat identity or boundary"))
            applied.add(h)
        previous_drain = (seen == FULL and e["route"] == "drain" and width == Q
                          and rows == dirty and not sample and not prefix)
        admitted, revoked, changed = map(set, (p["admitted"], p["revoked"], p["input_changed"]))
        writes = admitted | revoked | changed
        check(admitted <= unresolved and revoked <= accepted and writes <= sample
              and e["written_absolute"] == [base+x for x in sorted(writes)], (scope, "new writes"))
        dirty = (dirty - rows) | writes
        check(p["dirty_after"] == sorted(dirty)
              and e["dirty_after_absolute"] == [base+x for x in sorted(dirty)], (scope, "writeback"))
        if e["route"] in ("verify", "verify_paid"):
            check(not sample and not dirty and p["phase"] == "verify", (scope, "terminal verifier"))
        accepted = (accepted | admitted) - revoked
        seen.update(rows)
        age = p["age_after"]
        physical_positions.append(absolute)
        widths.append(width)
        ledger.append(["decoder", width])
        if e["encoded"]:
            check(e["route"] in ("verify", "verify_paid") and not dirty
                  and e["encoder_F"] == 1 and e["encoder_Q"] == B, (scope, "encoding"))
            ledger.append(["encoder", B])
            h, age = h+1, 0
            dirty, seen, accepted = set(), set(), set()
            previous_drain = False
    check(not dirty and arm["physical"] == ledger and arm["physical_positions"] == physical_positions
          and arm["E"] == h == len(arm["encoder"])-1 and arm["D"] == len(widths)
          and arm["Q"] == sum(widths) and arm["F"] == h+len(widths)
          and arm["route_F"] == Counter(e["route"] for e in es), (label, "result bill"))


def summarize(root, source):
    raw = (root/"raw.log").read_bytes()
    ref = (root/"reference.json").read_bytes()
    refs = prepare(source)
    check(json.loads(ref) == refs and (root/"pipeline.exit").read_text().split() == ["0", "0", "0"],
          "Unsealed run")
    release, = [l.decode() for l in raw.splitlines() if l.startswith(b"RELEASE ")]
    release_check = (root/"release-check.txt").read_text().strip()
    check(all(x in release for x in ("rc=0", f"state={GPU}\\,\\ 0\\,", "apps=''", "lock=closed"))
          and all(x in release_check for x in (f"state={GPU}\\,\\ 0\\,\\ 0", "apps=''",
                                               "worker=absent", "scratch=absent", "lock=free")),
          "Unverified A3 release")
    records = [json.loads(l[8:]) for l in raw.splitlines() if l.startswith(b"RECEIPT ")]
    start, *body, done = records
    check(start["kind"] == "start" and start["gpu"] == GPU and start["q"] == Q
          and start["profile"] == "dense_stop_joint_verified_eos" and start["cases"] == list(CASES)
          and start["reference_sha256"] == sha256(ref).hexdigest()
          and done == dict(kind="complete", requests=len(CASES)) and len(body) == 2*len(CASES),
          "Incomplete A3 stream")
    rows = []
    for j, (i, r) in enumerate(zip(CASES, refs)):
        control, result = body[2*j:2*j+2]
        check(control["kind"] == "controls" and result["kind"] == "case"
              and i == control["index"] == result["index"], "Wrong pair order")
        b, n, t = control["bare"], control["noop"], result["treat"]
        stock = r["case"]["live"]
        check(all(b[k] == n[k] == stock[k] for k in FIELDS), (i, "stock/no-op/historical mismatch"))
        notes, swaps = control["opportunities"], result["swaps"]
        audit_arm(n, notes, False, (i, "noop"))
        audit_arm(t, swaps, True, (i, "treat"))
        check(bool(notes) == bool(swaps), (i, "exposure mismatch"))
        fork = None
        if swaps:
            fork = swaps[0]
            k = fork["tick"]
            check(notes[0] == fork and n["events"][:k] == t["events"][:k]
                  and n["physical_positions"][k] != t["physical_positions"][k], (i, "first fork"))
            fork = dict(fork, predecessor_tape_sha256=digest(n["events"][:k]))
        else:
            check(all(n[k] == t[k] for k in FIELDS), (i, "unexposed changed"))
        def compact(a):
            return dict(**{k: a[k] for k in ("N", "D", "E", "F", "Q", "correct", "terminal_eos")},
                        peak_Q=max(e["q"] for e in a["events"]), ids_sha256=digest(a["ids"]))
        rows.append(dict(index=i, dataset=r["case"]["dataset"], bare=compact(b), noop=compact(n),
                         treat=compact(t), stock_opportunities=len(notes), swaps=len(swaps),
                         first_fork=fork))
    totals = {}
    for arm in ("bare", "noop", "treat"):
        rs = [r[arm] for r in rows]
        agg = {k: sum(a[k] for a in rs) for k in ("N", "D", "E", "F", "Q", "correct")}
        totals[arm] = dict(**agg, macro_tpf=mean(a["N"]/a["F"] for a in rs), pooled_te=agg["N"]/agg["Q"])
    return dict(schema="gemma-live-q128-post-drain-mask-a3-six-v1", source_sha256=sha256(raw).hexdigest(),
                reference_sha256=sha256(ref).hexdigest(), gpu=GPU, launcher_release=release,
                independent_release=release_check,
                cases=list(CASES), totals=totals, exposed=sum(bool(r["swaps"]) for r in rows),
                swaps=sum(r["swaps"] for r in rows),
                changed_ids=sum(r["noop"]["ids_sha256"] != r["treat"]["ids_sha256"] for r in rows),
                gains=[r["index"] for r in rows if not r["noop"]["correct"] and r["treat"]["correct"]],
                losses=[r["index"] for r in rows if r["noop"]["correct"] and not r["treat"]["correct"]],
                rows=rows)


if __name__ == "__main__":
    print(json.dumps(summarize(*map(Path, sys.argv[1:])), indent=2))
