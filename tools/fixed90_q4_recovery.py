"""Audit the two A3 LLaDA Q4 attempts in place; emit only compact receipts.

Run from the repo: python -m tools.fixed90_q4_recovery ROOT MANIFEST
ROOT is the n164 Q4 directory, not either individual attempt.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import re
from statistics import mean

from tools.fixed_admission_rows import B, FAMILY, MANIFEST_SHA, MASK, ROLES, SAMPLES, check, inspect_tick

ROOT = "/data1/project_data/gx-erqw-20261123/results/llada21-live-fullblock-fixed90-v1/Q4"
FIRST = "attempt-20261128T163518Z-a3"
RESUME = "attempt-20261128T165314Z-a3-resume"
Q, CAP, RECOVERED = 4, 2048, 20
INDICES = set(range(90))
PRIOR = set(range(RECOVERED + 1))
EXECUTOR = "7c584c5426d77fad3eec4aa397c1f58a86b7ea093ccded2574ee8b76fa44c993"
BUNDLE = "00176188a04430a5ed805dbdc387b0e7fcff5b6c58725a40dddf6db85646e7ed"
RUNNER = "4ece0e2598be8eb705a438eb696376599d89b00e70e771c3137335996f323feb"
CHECKPOINT = "d4b48682fb49b315db221a96e288fb01ea20499f6aa4e4ef705ec9a2b328a918"
SCORER = "a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692"
DEVICE = "GPU-94c61e79-d80b-cae2-d85f-6bd2a2c99a1c"
RECOVERED_SEAL = "cb280ade4e691462f613b4912873f1bc6d27e39fc1dd95b44ea392d8a7bd7c03"
RECOVERED_IDS = "b12430a404e4b31dc0dd4b20c606290cc8e5b82d7ddd00ade9bae1fbfd4281a3"
LOCK = {("llada21", Q): (EXECUTOR, BUNDLE, CHECKPOINT, SCORER)}


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(blob):
    return sha256(blob).hexdigest()


def verified_file(root, name, expected):
    blob = (root / name).read_bytes()
    check(digest(blob) == expected, ("receipt checksum", name))
    return blob


def verify_listing(root, name, expected):
    listing = {}
    for line in verified_file(root, name, expected).decode().splitlines():
        value, label = line.split("  ", 1)
        path = Path(label)
        check(not path.is_absolute() and ".." not in path.parts and path not in listing
              and re.fullmatch(r"[0-9a-f]{64}", value), ("invalid checksum entry", label))
        verified_file(root, path, value)
        listing[path] = value
    return listing


def signed_seal(blob, row, expected, *, new_seal):
    actual = digest(blob)
    if actual != expected:
        # Only fresh seals went through the integer->string transport sort.
        widths = row["widths"]
        check(new_seal and blob == (packed(row) + "\n").encode()
              and all(re.fullmatch(r"[1-9][0-9]*", k) for k in widths),
              "unexpected seal transport bytes")
        original = dict(row, widths={int(k): v for k, v in widths.items()})
        check(digest((packed(original) + "\n").encode()) == expected, "producer seal checksum")
    return actual


def indexed(events, label, expected):
    entries = [e for e in events if e.get("kind") == label]
    by_index = {e["index"]: e for e in entries}
    check(len(entries) == len(by_index) and set(by_index) == expected, (label, "case coverage"))
    return by_index


def one(events, key, label):
    matches = [e for e in events if e.get(key) == label]
    check(len(matches) == 1, ("lifecycle receipt", label))
    return matches[0]


def partition(first, resume, samples):
    active = (indexed(first, "case_start", PRIOR),
              indexed(resume, "case_start", INDICES - PRIOR))
    for starts in active:
        check(all(e["seed"] == samples[i]["seed"] for i, e in starts.items()), "worker seed drift")
    indexed(first, "prior_reused", set())
    return indexed(resume, "prior_reused", PRIOR)


def verify_worker(events, attempt, case_ids, *, finished):
    reg = one(events, "transport", "registered")
    cp = one(events, "transport", "checkpoint_verified")
    worker = one(events, "kind", "worker")
    check((reg["id"], reg["offer"], reg["indices"], reg["mode"], reg["runner_sha256"],
           reg["bundle_sha256"], reg["pid"], cp["fingerprint"], cp["scorer"], worker["uuid"])
          == (attempt, Q, 90, "run", RUNNER, BUNDLE, worker["pid"], CHECKPOINT, SCORER, DEVICE),
          ("source, worker or checkpoint drift", attempt))
    sealed = [e["name"] for e in events if e.get("kind") == "transport_sealed"]
    names = [f"cases/{i:02}.json" for i in case_ids]
    if finished:
        names.append("complete.json")
        check(one(events, "kind", "complete")["count"] == 90, "worker did not complete")
    else:
        check(not any(e.get("kind") == "complete" for e in events), "unexpected first-attempt completion")
    check(Counter(sealed) == Counter(names), ("transport seal coverage", attempt))
    return worker


def check_origin(row, i, attempt, reused, digest_on_disk):
    check(row["raw"] == f"{attempt}/case-{i:02}.ndjson", (i, "wrong generation attempt"))
    check((row.get("recovered_from_retirement_tape") is True) == (i == RECOVERED),
          (i, "wrong scoring origin"))
    if reused is not None:
        check(reused["receipt_sha256"] == digest_on_disk, (i, "copied seal differs from reuse claim"))


def audit_tape(path, row, i):
    receipt, totals, bills, phases, widths = sha256(), Counter(), Counter(), Counter(), Counter()
    canvas, debt, starts, retired = {}, set(), set(), []
    origin = row["prefill"]["decoder_q"]
    clock = 0
    with path.open("rb") as stream:
        for tick, line in enumerate(stream):
            receipt.update(line)
            tape = json.loads(line)
            e = tape["event"]
            clock += bool(e["sampled"])
            check(e["tick"] == tick and e["sample_clock"] == clock and e["encoder_Q"] == 0,
                  (i, tick, "clock or encoder bill"))
            billed, debt = inspect_tick(tape, Q, debt, origin, origin + CAP)
            width = billed["Q"]
            totals.update({k: billed[k] for k in (*ROLES, "Q", "F", "over_offer_Q")})
            bills.update(F=1, Q=width, encoder_F=0, encoder_Q=0, paid_rows=billed["writeback"],
                         written_rows=len(tape["written"]))
            phases[e["phase"]] += 1
            widths[str(width)] += 1
            for p, token in zip(e["rows"], tape["ids"]):
                check(p not in canvas or canvas[p] == token, (i, tick, "canvas identity"))
                canvas[p] = token
            canvas.update((int(p), token) for p, token in tape["written"].items())
            for block in e["retired_blocks"]:
                check(block == origin // B + len(retired), (i, tick, "retirement order"))
                lo, hi = max(origin, block * B), min(origin + row["output_ids"], (block + 1) * B)
                check(lo < hi and all(p not in debt and p in canvas and canvas[p] != MASK
                                      for p in range(lo, hi)), (i, tick, "unpaid retirement"))
                retired.append(block)
            if billed["admission_F"]:
                start = e["admission_start"]
                check(start not in starts, (i, "duplicate admission"))
                starts.add(start)
                bills.update(build_F=1, build_Q=width, new_block_Q=billed["new_block"],
                             over_offer_Q=billed["over_offer_Q"])
                totals.update(build_F=1, build_Q=width, admission_overflows=int(width > Q))
            else:
                check(width <= Q, (i, "ordinary overflow"))
                totals.update(ordinary_Q=width)
    size = row["output_ids"]
    ids = [canvas.get(p) for p in range(origin, origin + size)]
    check(not debt and receipt.hexdigest() == row["raw_sha256"]
          and row["F"] == row["decoder_F"] == bills["F"] > 0
          and row["Q"] == row["decoder_Q"] == bills["Q"]
          and dict(bills) == row["bills"] and dict(phases) == row["phases"] and dict(widths) == row["widths"],
          (i, "unreconciled tape"))
    check(type(row["N"]) is type(size) is int and 0 <= row["N"] <= size <= CAP
          and ((row["end_reason"] == "eos" and row["N"] < size)
               or (row["end_reason"] == "cap" and size == CAP))
          and len(retired) == (origin + size - 1) // B - origin // B + 1
          and all(type(token) is int and token >= 0 and token != MASK for token in ids)
          and digest(packed(ids).encode()) == row["ids_sha"], (i, "terminal IDs"))
    return totals, max(map(int, widths))


def report(root, manifest_blob):
    check(digest(manifest_blob) == MANIFEST_SHA, "frozen manifest checksum")
    manifest = json.loads(manifest_blob)
    first, resume = root / FIRST, root / RESUME
    bundle = first / "bundle"
    check(resume.joinpath("bundle").resolve() == bundle.resolve(), "changed resume bundle")
    verify_listing(bundle, "bundle.sha256", BUNDLE)
    frozen = json.loads(verified_file(bundle, "samples.json", SAMPLES))["samples"]
    check(manifest["samples_sha256"] == SAMPLES and len(frozen) == len(manifest["samples"]) == 90,
          "fixed samples coverage")
    fields = ("index", "task_id", "dataset", "seed")
    for i, (task, sample) in enumerate(zip(frozen, manifest["samples"])):
        check(task["index"] == i and all(task[k] == sample[k] for k in fields)
              and digest(task["prompt"].encode()) == sample["prompt_sha256"], (i, "frozen task identity"))
    verified_file(bundle, "fixed90.py", RUNNER)
    verified_file(bundle, "src/experiments/t2t/serving_slots/llada_response_executor.py", EXECUTOR)
    verified_file(bundle, "src/bdlm_serving/evaluation/scoring.py", SCORER)
    verification = verify_listing(resume, "verification.sha256",
                                   "be799033db25d2755e951d2773200246918b2302bff5675a675b61ea9ead72b1")
    resume_receipts = verify_listing(resume, "resume.sha256", verification[Path("resume.sha256")])
    w1_blob = verified_file(first, "worker.ndjson",
                            "36621c2686ef814d23acac73f21d764478111a690d88e03f78ace3070119938c")
    w2_blob = (resume / "worker.ndjson").read_bytes()
    w1, w2 = ([json.loads(line) for line in blob.splitlines()] for blob in (w1_blob, w2_blob))
    exits = [(p / "pipeline.exit").read_text().split() for p in (first, resume)]
    check(exits == [["0", "1", "0"], ["0", "0", "0"]], "unexpected attempt exit")
    prior_scored, new = PRIOR - {RECOVERED}, INDICES - PRIOR
    workers = [verify_worker(w1, FIRST, prior_scored, finished=False),
               verify_worker(w2, RESUME, new, finished=True)]
    check(workers[0]["pid"] != workers[1]["pid"], "attempts lack distinct workers")
    reused = partition(w1, w2, frozen)
    scored = {**indexed(w1, "sealed", prior_scored), **indexed(w2, "sealed", new)}
    failure = indexed(w1, "case_error", {RECOVERED})[RECOVERED]
    indexed(w2, "case_error", set())
    check(all(s in failure["reason"] for s in ("Code evaluation sandbox unavailable", "bwrap", "Permission denied")),
          "unexpected recovery reason")
    close = json.loads((resume / "complete.json").read_bytes())
    expected = close["case_sha256"]
    check(close["completed"] == 90 and close["source_sha256"] == BUNDLE
          and set(expected) == {f"{i:02}.json" for i in INDICES}, "complete seal coverage")
    for directory, indices in ((first, prior_scored), (resume, INDICES)):
        check({p.name for p in (directory / "cases").glob("*.json")} == {f"{i:02}.json" for i in indices}
              and {p.name for p in directory.glob("case-*.ndjson")}
              == {f"case-{i:02}.ndjson" for i in (PRIOR if directory == first else new)},
              ("unexpected duplicate or missing on-disk cases", directory.name))

    cases, seals, counts = {}, {}, Counter(ordinary_overflows=0)
    for i, task in enumerate(frozen):
        case = f"cases/{i:02}.json"
        generation = first if i in PRIOR else resume
        original = "recovered-20.json" if i == RECOVERED else case
        blob = (generation / original).read_bytes()
        row = json.loads(blob)
        check(all(row[k] == task[k] for k in fields)
              and (row["status"], row["offer"], row["mode"], row["samples_sha256"], row["source_sha256"],
                   row["checkpoint"], row["scorer"], row["uuid"])
              == ("complete", Q, "full_block", SAMPLES, BUNDLE, CHECKPOINT, SCORER, DEVICE)
              and type(row["correct"]) is bool, (i, "case identity"))
        seal = expected[f"{i:02}.json"]
        received = signed_seal(blob, row, seal, new_seal=i in new)
        check_origin(row, i, generation.name, reused.get(i), received)
        if i in PRIOR:
            check(blob == (resume / case).read_bytes() and received == resume_receipts[Path(case)],
                  (i, "copied seal changed"))
        if i == RECOVERED:
            check(received == RECOVERED_SEAL and row["ids_sha"] == RECOVERED_IDS and row["end_reason"] == "eos"
                  and row["bills"] == failure["completed_bills"], "recovered score/retirement receipt")
        else:
            check(all(row[k] == scored[i][k] for k in ("N", "F", "Q", "correct")), (i, "worker scoring receipt"))
        pre = row["prefill"]
        ref = json.loads(verified_file(bundle, f"reference/{i:02}-native32.json", row["ref_sha256"]))
        check(all(ref[k] == task[k] for k in fields) and ref["generation_cap"] == CAP
              and pre == dict(decoder_f=1, decoder_q=len(ref["prompt_ids"]), encoder_f=0, encoder_q=0)
              and pre["decoder_q"] > 0 and digest(packed(ref["prompt_ids"]).encode()) == row["prompt_sha256"],
              (i, "native prompt or prefill"))
        raw = generation / f"case-{i:02}.ndjson"
        check(raw.resolve() == generation.resolve() / raw.name, (i, "redirected raw receipt"))
        billed, peak = audit_tape(raw, row, i)
        counts.update(billed)
        counts.update(prefill_F=1, prefill_Q=pre["decoder_q"])
        seals[str(i)] = seal
        cases[str(i)] = dict(task_id=row["task_id"], seed=row["seed"], correct=row["correct"],
                             terminal=row["end_reason"], N=row["N"], F=row["F"], Q=row["Q"],
                             decoder_F=row["F"], encoder_F=0, peak=peak, over=billed["over_offer_Q"],
                             build_Q=billed["new_block"], admission_Q=billed["build_Q"],
                             admission_F=billed["build_F"], prefill_Q=pre["decoder_q"],
                             attempt=generation.name, source_seal=original, received_sha256=received,
                             raw_sha256=row["raw_sha256"], ids_sha256=row["ids_sha"])

    def subtotal(rows):
        return dict(N=sum(r["N"] for r in rows), correct=sum(r["correct"] for r in rows),
                    TPF=mean(r["N"] / r["F"] for r in rows),
                    TE=sum(r["N"] for r in rows) / sum(r["Q"] for r in rows))

    ordered = list(cases.values())
    full = subtotal(ordered)
    terminal_counts = dict(Counter(row["terminal"] for row in ordered))
    normalized = sum(seals[i] != row["received_sha256"] for i, row in cases.items())
    summary = json.loads((resume / "summary-verified.json").read_bytes())
    old = summary["frozen90"]
    check(summary["normalized_width_key_order"] == normalized == 69 and old["terminal"] == terminal_counts
          and all(counts[k] == old[k] for k in ("F", "Q", "build_Q", "build_F", "prefill_Q", "prefill_F",
                                               "over_offer_Q", "ordinary_Q", "ordinary_overflows",
                                               "admission_overflows"))
          and counts["new_block"] == old["new_block_Q"]
          and all(full[k] == old[k] for k in ("N", "correct"))
          and isclose(full["TPF"], old["macro_TPF"]) and isclose(full["TE"], old["TE"]),
          "independent n164 summary disagreement")
    return dict(model="llada21", q=Q, cohort="frozen90", family=FAMILY, manifest_sha256=MANIFEST_SHA,
                executor_sha256=EXECUTOR, runner_sources_sha256=BUNDLE, checkpoint=CHECKPOINT,
                scorer=SCORER, native_prompts_verified=True, complete=True, verified=90, missing=[],
                sealed_failures=[], counts=dict(counts), peak=max(r["peak"] for r in ordered),
                terminals=terminal_counts, **full,
                cohorts=dict(original30=subtotal(ordered[:30]), new60=subtotal(ordered[30:])),
                provenance=dict(host="n164", root=ROOT, device_uuid=DEVICE, fixed_runner_sha256=RUNNER,
                                samples_sha256=SAMPLES, normalized_width_key_seals=normalized,
                                final_seal_sha256=verification[Path("complete.json")],
                                attempts={p.name: dict(pid=w["pid"], generation_indices=span, exit=rc,
                                                       worker_sha256=digest(blob))
                                          for p, w, span, rc, blob in
                                          zip((first, resume), workers, ([0, 20], [21, 89]), exits, (w1_blob, w2_blob))},
                                recovered_20=dict(scoring_host="n164", method="retired_ids_same_scorer_bwrap",
                                                  original_failure="bwrap temporary path permission denied",
                                                  script=f"{RESUME}/recover.py",
                                                  script_sha256=resume_receipts[Path("recover.py")],
                                                  seal=f"{FIRST}/recovered-20.json", seal_sha256=RECOVERED_SEAL)),
                seal_digests=seals, cases=cases)


def print_report(doc):
    # One line per case for reviewing all 90 rows without exposing the tapes.
    print("{")
    for k, value in doc.items():
        if k != "cases":
            print(f"  {json.dumps(k)}: {json.dumps(value, allow_nan=False)},")
    print('  "cases": {')
    for i, (key, row) in enumerate(doc["cases"].items()):
        print(f"    {json.dumps(key)}: {json.dumps(row, allow_nan=False)}" + ("," if i < 89 else ""))
    print("  }\n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print_report(report(args.root, args.manifest.read_bytes()))
