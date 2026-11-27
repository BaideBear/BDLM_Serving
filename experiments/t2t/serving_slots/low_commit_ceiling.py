"""Fixed-path COMMIT-elision ceiling, not an alternate decoder measurement.

Run from the checkout on the data host:
    python -m experiments.t2t.serving_slots.low_commit_ceiling \
        --data ../t2t-native-1125/prefix90

N sums recorded non-special output-token counts. F and physical Q are
reconstructed from charged forwards, excluding prompt prefill only. Delete
only the physical COMMIT for each eligible pair; retain VERIFY and the logical
commit. Pooled TPF is a ratio of totals; macro TPF gives each request
equal weight, as in the frozen90 report. Both projected TPFs are trace-only.

Eligibility tests the observable necessary conditions of verified_commit.py.
Old traces cannot establish KV object identity, tensor equality, or parity
after skipping a forward. Source digests fingerprint exact input bytes, not
runtime parity. No output text or token IDs are compared.
"""

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from math import fsum, isclose
from pathlib import Path

from bdlm_serving.evaluation.frozen90 import load_tasks
from experiments.t2t.serving_slots.llada_capped import audit as audit_cap
from experiments.t2t.serving_slots.promotion90 import SAMPLES, peers
from experiments.t2t.serving_slots.slot_costs import COHORTS, count, require


CONTROLS = Path(__file__).resolve().parent / "controls"
OFFERS = (4, 8, 16, 32)
B = 32
LABEL = "trace_only_bound_not_measured"
IDENTITY = ("index", "seed", "dataset", "task_id")
PHASE = {"prefill": "prefill", "bootstrap": "denoise", "selected": "denoise",
         "verify": "denoise", "commit": "commit"}
CAP_GROUP = {4: "low", 8: "endpoints", 16: "endpoints", 32: "anchor"}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(entries):
    """SHA256 of ordered ASCII lines '<filename> <sha256-of-file>\\n'."""
    return digest("".join(f"{name} {sha}\n" for name, sha in entries).encode("ascii"))


def read(path, entries):
    raw = path.read_bytes()
    entries.append((path.name, digest(raw)))
    return json.loads(raw)


def check_keys(row, expected, source):
    require(all(row.get(key) == val for key, val in expected.items()), f"Wrong source: {source}")


def full_head(event):
    block = event.get("block")
    version = event.get("context_version")
    if type(block) is not int or block < 0 or type(version) is not int or version < 0:
        return False
    start = B * block
    return all(event.get(key) == val for key, val in dict(
        q_start=start, q_end=start+B, prefix_kv_length=start, visible_kv=start+B,
        absolute_positions=list(range(start, start+B)), F=1, q_tokens=B, actual_q=B,
        physical_q=B, layer_physical_q=[B]*20, approximate_same_block_kv=False,
    ).items())


def eligible_pair(verify, commit):
    """Call only on adjacent events of a single original request."""
    return (
        (verify.get("phase"), verify.get("stage"), verify.get("store_kv"))
        == ("denoise", "verify", False)
        and (commit.get("phase"), commit.get("stage"), commit.get("store_kv"))
        == ("commit", "commit", True)
        and full_head(verify) and full_head(commit)
        and verify["block"] == commit["block"]
        and verify["context_version"] == commit["context_version"]
        and all(type(verify.get(key)) is int and verify[key] == 0
                for key in ("masked_before", "masked_after", "accepted", "edited"))
    )


def trace_work(row, offer):
    events = row["events"]
    require("logical_commits" not in row, "Expected unoptimized trace")
    require(all(e.get("stage") in PHASE and e.get("phase") == PHASE[e["stage"]]
                and e.get("F") == 1 and e.get("q_budget") == offer
                for e in events), "Expected fixed single-block K0 forward schedule")
    require(type(row["output_tokens"]) is int and row["output_tokens"] >= 0,
            "Invalid recorded N")
    work, widths = count(row, "llada21", offer)
    require(work["Q"] == row["generation_query_rows"] and max(widths) == row["max_q"],
            "Old request projection ledger mismatch")
    commits = [e["block"] for e in events if e["stage"] == "commit"]
    require(commits and commits == list(range(commits[0], commits[0]+len(commits))),
            "Expected one physical COMMIT per serial block")
    work["k"] = sum(eligible_pair(v, c) for v, c in zip(events, events[1:]))
    return work


def source_run(folder, policy, headers, expected):
    run = read(folder / "run.json", headers)
    check_keys(run, expected, folder)
    require(policy in run["policies"], f"Unregistered policy: {folder}/{policy}")
    seal = read(folder / f"complete-{policy}.json", headers)
    require(seal == dict(policy=policy, requests=90), f"Unsealed source: {folder}/{policy}")
    return run


def source_rows(folder, policy, tasks):
    for task in tasks:
        entries = []
        path = folder / f"{task['index']:02d}-{policy}.json"
        row = read(path, entries)
        check_keys(row, dict(model="llada21", policy=policy, **{k: task[k] for k in IDENTITY}),
                   path)
        yield row, entries[0]


def old_source(data, offer, tasks, meta):
    folder, policy = peers(data, offer)
    headers = []
    run = source_run(folder, policy, headers, dict(meta, cap=2048, seed_offset=0))
    qualification = read(folder / "qualified.json", headers)
    require(qualification == dict(model="llada21", full_width=B, matched=90),
            f"Missing B32 qualification: {folder}")
    require(run["arms"].count(dict(budget=offer, fresh=0, mask=offer,
                                   selection=policy if offer in (8, 16) else "hard_prefix_zero")) == 1,
            f"Not the registered K0 arm: {folder}/{policy}")
    work = []
    for row, entry in source_rows(folder, policy, tasks):
        require(row["status"] == "success", f"Incomplete old K0 trace: {entry[0]}")
        work.append((trace_work(row, offer), entry))
    return folder.name, policy, headers, work


def capped_work(row, offer):
    """Cross-check the existing cap audit against the actual batch-one calls."""
    check_keys(row, dict(phase="formal", generation_cap=2048), row["policy"])
    require(row["status"] in ("success", "generation_failed"), "Unknown capped completion status")
    events = [e for e in row["events"] if e.get("physical_forwards", 1)]
    audit_cap(row, (offer,))
    work = Counter(requests=1)
    for i, (event, call) in enumerate(zip(events, row["calls"], strict=True)):
        require(event.get("physical_forwards", 1) == call["batch"] == 1,
                "Not a single-request physical call")
        if event["phase"] == "prefill":
            require(i == 0, "Unexpected free prefill")
            work["prefill"] += call["physical_rows"]
        else:
            work.update(F=1, Q=call["physical_rows"])
    require(work["prefill"] == row["prefill_rows"] and work["F"] == row["generation_forwards"]
            and work["Q"] == row["generation_physical_padded_rows"], "Capped call ledger mismatch")
    return work


def capped_source(data, offer, mode, tasks, meta):
    folder = data / f"llada21-optimus-hardcap-{CAP_GROUP[offer]}"
    policy = f"optcap_{mode}_{offer}"
    # An in-progress arm, even with an entire original30, is not a sealed reference.
    if not (folder / f"complete-{policy}.json").is_file():
        return None
    headers = []
    source_run(folder, policy, headers, dict(meta, hard_q_cap=True,
               cap_clock="all_decoder_forwards_request_local", cap_excludes="prompt_prefill_only"))
    proof = read(data / "llada21-optimus-hardcap-anchor" / "replay-qualified.json", headers)
    require(proof == dict(model="llada21", control="optimus_ibs_c32", samples=SAMPLES, matched=90),
            "Missing cap replay qualification")
    work = [(capped_work(row, offer), entry) for row, entry in source_rows(folder, policy, tasks)]
    return folder.name, policy, headers, work


def table(path, optional=False):
    if optional and not path.is_file():
        return None, {}
    raw = path.read_bytes()
    return digest(raw), list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))


def published(controls, offer, mode, policy):
    nominal = f"optimus_{mode}_c{offer}"
    cap = f"optcap_{mode}_{offer}"
    reference_sha, reference = table(controls / f"llada21-q{offer}-original-{mode}.csv")
    if offer == 32:
        cap_sha, cap_rows = table(controls / "llada21-original-vs-capped-q32.csv")
        cap_match = dict(arm_policy=cap, baseline_policy=nominal)
        cap_column = "arm_physical_Q"
    else:
        cap_sha, cap_rows = table(controls / f"llada21-q{offer}-vs-capped-{mode}.csv", optional=True)
        # Q8's published comparison uses the separately named K0 run; verify
        # its aggregate N/F/Q against the registered slottrace before use.
        cap_match = dict(arm_policy=f"t2t_q{offer}_f0_m{offer}_hard_prefix_zero", baseline_policy=cap)
        cap_column = "baseline_physical_Q"

    def select(rows, cohort, length, labels):
        matches = [r for r in rows if all(r.get(k) == v for k, v in dict(
            model="llada21", nominal_q=str(offer), cohort=cohort, scope="all",
            window=str(length), requests=str(length), **labels).items())]
        require(len(matches) == 1, f"Missing or duplicate published comparison: {labels}/{cohort}")
        return matches[0]

    return {cohort: (
        select(reference, cohort, stop-start, dict(arm_policy=policy, baseline_policy=nominal)),
        reference_sha,
        select(cap_rows, cohort, stop-start, cap_match) if cap_sha else None,
        cap_sha, cap_column,
    ) for start, stop, cohort in COHORTS}


def pool(rows, start, stop):
    totals = Counter()
    for work, _ in rows[start:stop]:
        totals.update(work)
    return totals


def check_published_k0(row, work, macro_tpf):
    require(all(int(row["arm_"+col]) == work[key]
                for col, key in (("N", "N"), ("F", "F"), ("physical_Q", "Q")))
            and isclose(float(row["arm_macro_tpf"]), macro_tpf, rel_tol=1e-12, abs_tol=0),
            "Published comparison does not reconcile to old K0 work and macro TPF")


def macro_tpfs(requests):
    require(requests and all(0 <= r["k"] < r["F"] for r in requests),
            "Undefined per-request TPF")
    return (fsum(r["N"]/r["F"] for r in requests)/len(requests),
            fsum(r["N"]/(r["F"]-r["k"]) for r in requests)/len(requests))


def projection(work, offer):
    n, f, q, k = (work[key] for key in ("N", "F", "Q", "k"))
    require(0 <= k <= work["commit_F"] < f and B*k < q, "Invalid elision arithmetic")
    saved = B*k
    over_f = work["over_F"] - (k if offer < B else 0)
    over_q = work["over_Q"] - k*max(0, B-offer)
    return dict(N=n, F=f, physical_Q=q, commit_F=work["commit_F"], eligible_k=k,
                saved_physical_Q=saved, saved_Q_pct=100*saved/q,
                observed_pooled_tpf=n/f, projected_F=f-k, projected_physical_Q=q-saved,
                projected_pooled_tpf=n/(f-k), projected_calls_over_slot=over_f,
                projected_rows_over_slot=over_q,
                bootstrap_Q=work["bootstrap_Q"], verify_Q=work["verify_Q"],
                selected_Q=work["selected_Q"], projected_commit_Q=work["commit_Q"]-saved)


def gaps(old_q, projected_q, baseline_q):
    if baseline_q is None:
        return None, None, None
    before, after = old_q-baseline_q, projected_q-baseline_q
    return before, after, 100*after/baseline_q


def report(data, controls=CONTROLS):
    tasks = load_tasks(data / "samples.json")
    meta = dict(model="llada21", suite="frozen90", samples_sha256=SAMPLES, block=B, batch_size=1)
    fixed_sha, fixed = table(controls / "fixed-k0-costs.csv")
    result = []
    for offer in OFFERS:
        name, policy, headers, traces = old_source(data, offer, tasks, meta)
        caps = {m: capped_source(data, offer, m, tasks, meta) for m in ("ibs", "obs")}
        refs = {m: published(controls, offer, m, policy) for m in caps}
        for start, stop, cohort in COHORTS:
            work = pool(traces, start, stop)
            old_macro, projected_macro = macro_tpfs([r for r, _ in traces[start:stop]])
            ledger = [r for r in fixed if (r["model"], r["offer"], r["cohort"])
                      == ("llada21", str(offer), cohort)]
            keys = ("requests", "N", "F", "Q", "over_Q", "over_F", "bootstrap_Q",
                    "verify_Q", "selected_Q", "commit_F", "commit_Q")
            require(len(ledger) == 1 and all(int(ledger[0][k]) == work[k] for k in keys),
                    f"Published K0 ledger mismatch: Q{offer}/{cohort}")
            out = dict(model="llada21", offer=offer, cohort=cohort, requests=stop-start,
                       result_type=LABEL, old_policy=policy, old_source=name, **projection(work, offer),
                       observed_macro_tpf=old_macro, projected_macro_tpf=projected_macro)
            for mode, cap in caps.items():
                nominal, nominal_sha, cap_pub, cap_pub_sha, cap_col = refs[mode][cohort]
                check_published_k0(nominal, work, old_macro)
                nominal_q = int(nominal["baseline_physical_Q"])
                cap_q = cap_name = cap_policy = cap_sha = None
                if cap:
                    cap_name, cap_policy, cap_headers, cap_rows = cap
                    cap_q = pool(cap_rows, start, stop)["Q"]
                    cap_sha = fingerprint(cap_headers + [e for _, e in cap_rows[start:stop]])
                if cap_pub:
                    require(cap_q == int(cap_pub[cap_col]),
                            f"Published cap Q mismatch: Q{offer}/{mode}/{cohort}")
                    if offer != 32:
                        check_published_k0(cap_pub, work, old_macro)
                    else:
                        require(nominal_q == int(cap_pub["baseline_physical_Q"]),
                                f"Published nominal Q mismatch: Q32/{mode}/{cohort}")
                for label, baseline_q in (("capped", cap_q), ("nominal", nominal_q)):
                    stem = f"{mode}_{label}"
                    before, after, pct = gaps(work["Q"], out["projected_physical_Q"], baseline_q)
                    out.update({f"{stem}_physical_Q": baseline_q, f"{stem}_old_gap_Q": before,
                                f"{stem}_projected_gap_Q": after, f"{stem}_projected_gap_pct": pct})
                out.update({f"{mode}_cap_status": "sealed_full90" if cap else "no_sealed_full90",
                            f"{mode}_cap_source": cap_name, f"{mode}_cap_policy": cap_policy,
                            f"{mode}_cap_source_sha256": cap_sha, f"{mode}_cap_published_sha256": cap_pub_sha,
                            f"{mode}_nominal_published_sha256": nominal_sha})
            out.update(samples_sha256=SAMPLES, old_source_sha256=fingerprint(
                headers + [e for _, e in traces[start:stop]]), published_k0_sha256=fixed_sha)
            result.append(out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Registered prefix90 runtime directory")
    parser.add_argument("--output", type=Path, help="New CSV file; defaults to stdout")
    args = parser.parse_args()
    rows = report(args.data)

    def emit(stream):
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    if args.output:
        with args.output.open("x", newline="") as stream:
            emit(stream)
    else:
        emit(sys.stdout)


if __name__ == "__main__":
    main()
