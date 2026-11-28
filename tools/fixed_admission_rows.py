"""Audit row roles on fixed-offer LLaDA full-admission tapes, without model inference."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean

B, MASK = 32, 156895
ROLES = ("writeback", "new_block", "other_mask", "clean_decoded")
SAMPLES = "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed"
MANIFEST_SHA = "2e67d12f5045d486d1676a4b9cd35718b0791ad1b2a3c81fdec86a27bd773e62"
FAMILY = "live-full-successor-v1"
A1_BUNDLE = "30e6ff4f9e55291fab31a99c57838509095202d0afe25ff4d8554656038e84d4"
A1_RUNNER = "4ece0e2598be8eb705a438eb696376599d89b00e70e771c3137335996f323feb"
A1_EXECUTOR = "7c584c5426d77fad3eec4aa397c1f58a86b7ea093ccded2574ee8b76fa44c993"
A1_PAYLOAD = "d4b48682fb49b315db221a96e288fb01ea20499f6aa4e4ef705ec9a2b328a918"
H4_CHECKPOINT = "aac79bb98f76cfe422079843f1dcc53a9e8c9b5cf56008d43f3ad4a4d3791518"
SCORER = "a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692"
CAP = 2048
EXECUTOR_FILE = "experiments/t2t/serving_slots/llada_response_executor.py"


def check(condition, note):
    if not condition:
        raise ValueError(note)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def packed(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def case_digests(data, row, expected=None, *, transport=False):
    received = sha256(data).hexdigest()
    if expected is None or received == expected:
        return received, received
    # A1's receiver sorted decimal string keys; the producer sorted integer keys.
    check(transport and data == (packed(row) + "\n").encode(), "case seal bytes")
    widths = row["widths"]
    check(isinstance(widths, dict) and all(re.fullmatch(r"[1-9][0-9]*", k) for k in widths),
          "noncanonical width keys")
    original = dict(row, widths={int(k): v for k, v in widths.items()})
    producer = sha256((packed(original) + "\n").encode()).hexdigest()
    check(producer == expected, "case seal digest")
    return received, producer


def output_ids(r, writes):
    size, origin = r["output_ids"], r["prefill"]["decoder_q"]
    check(type(r["N"]) is type(size) is int and 0 <= r["N"] <= size <= CAP
          and (size == CAP if r["end_reason"] == "cap" else r["N"] < size), "output length")
    ids = [writes.get(p) for p in range(origin, origin + size)]
    check(all(type(token) is int and token >= 0 and token != MASK for token in ids)
          and sha256(packed(ids).encode()).hexdigest() == r["ids_sha"], "output token digest")
    return ids


def tape_path(root, row, index, *, transport=False):
    if transport:
        name = f"case-{index:02}.ndjson"
        check(row["raw"] == f"{root.name}/{name}", "transported raw path")
        path = root / name
        check(path.resolve() == root.resolve() / name, "redirected raw path")
        return path
    path = (root / row["raw"]).resolve()
    check(root.resolve() in path.parents, "raw path")
    return path


def a1_registration(root):
    bundle = root / "bundle"
    check(digest(bundle/"bundle.sha256") == A1_BUNDLE
          and digest(bundle/"fixed90.py") == A1_RUNNER
          and digest(bundle/"src"/EXECUTOR_FILE) == A1_EXECUTOR
          and digest(bundle/"src/bdlm_serving/evaluation/scoring.py") == SCORER
          and digest(bundle/"samples.json") == SAMPLES
          and (root/"pipeline.exit").read_text().split() == ["0", "0", "0"], "A1 source or exit")
    with (root/"worker.ndjson").open() as stream:
        events = [json.loads(line) for line in stream]
    registered, checkpoint, worker, done = (
        [e for e in events if e.get(key) == label] for key, label in
        (("transport", "registered"), ("transport", "checkpoint_verified"),
         ("kind", "worker"), ("kind", "complete")))
    check(all(len(m) == 1 for m in (registered, checkpoint, worker, done)), "A1 lifecycle receipts")
    reg, cp, w = registered[0], checkpoint[0], worker[0]
    check((reg["offer"], reg["indices"], reg["mode"], reg["id"], reg["runner_sha256"], reg["bundle_sha256"])
          == (8, 90, "run", root.name, A1_RUNNER, A1_BUNDLE)
          and reg["pid"] == w["pid"] and done[0]["count"] == 90
          and (cp["fingerprint"], cp["qualified_h4_repository_fingerprint"], cp["scorer"])
          == (A1_PAYLOAD, H4_CHECKPOINT, SCORER), "A1 registration or checkpoint/scorer")
    seals = [e["name"] for e in events if e.get("kind") == "transport_sealed"]
    check(Counter(seals) == Counter(["complete.json", *(f"cases/{i:02}.json" for i in range(90))]),
          "A1 transport coverage")
    scored = [e for e in events if e.get("kind") == "sealed"]
    scores = {e["index"]: e for e in scored}
    check(len(scored) == 90 and set(scores) == set(range(90)), "A1 scoring coverage")
    return w["uuid"], scores


def inspect_tick(tape, q, debt, origin, end):
    e = tape["event"]
    rows = e["rows"]
    work = set(rows)
    check(len(rows) == len(tape["ids"]) and rows and all(origin <= p < end for p in rows), "decoder positions")
    old = dict(zip(rows, tape["ids"]))
    paid, new = set(e["dirty"]), set(e["admission_rows"])
    sampled, writes = set(e["sampled"]), set(e["written_absolute"])
    changed = {int(p): v for p, v in tape["written"].items()}
    check(rows == sorted(work) and len(rows) == e["physical_q"] == e["decoder_Q"] == e["history_Q"]
          and e["layer_physical_q"] == [len(rows)] * 20 and e["encoder_F"] == e.get("encoder_Q", 0) == 0
          and e["decoder_F"] == e["generation_F"] == 1 and e["offer"] == q, "physical bill")
    check(debt == set(e["dirty_before"]) and paid == debt & work
          and (debt - paid) | writes == set(e["dirty_after"])
          and changed.keys() == writes and writes <= sampled <= work, "debt or writes")
    check(not (debt - paid and sampled), "sampled with mandatory debt outstanding")
    check(not new or (new <= sampled and all(old[p] == MASK for p in new)
                      and min(new) == tape["before"] == e["admission_start"]
                      and min(new) % B == 0 and sorted(new) == list(range(min(new), min(min(new)+B,end)))
                      and len(new) <= B and new.isdisjoint(paid)), "new block")
    plan = e["parent_plan_rows"]
    check(e["response_mode"] == "full_block" and e["nominal_offer"] == q and
          len(plan) == len(set(plan)) <= q and work == set(plan) | new and
          (not new or new & set(plan)) and
          e["over_offer_Q"] == max(0, len(rows)-q) and (len(rows) <= q or new), "overflow or mode")
    other = work - paid - new
    prefix = {p for p in other & set(e["prefix"]) if old[p] != MASK}
    edits = {p for p in writes if old[p] != MASK and changed[p] not in (MASK, old[p])}
    check(edits == set(e["edited"]), "T2T attribution")
    c = Counter(Q=len(rows), F=1, writeback=len(paid), new_block=len(new),
                other_mask=sum(old[p] == MASK for p in other),
                clean_decoded=sum(old[p] != MASK for p in other), t2t=len(edits),
                t2t_on_writeback=len(edits & paid), t2t_on_clean_decoded=len(edits & other),
                clean_decoded_sampled=sum(old[p] != MASK for p in other & sampled),
                selected_optional_prefix=len(prefix), sampled_optional_prefix=len(prefix & sampled),
                t2t_on_optional_prefix=len(prefix & edits),
                admission_F=bool(new), admission_Q=len(rows) if new else 0,
                admission_overage_F=bool(new and len(rows) > q),
                ordinary_overflow_F=bool(not new and len(rows) > q),
                over_offer_Q=e["over_offer_Q"])
    check(sum(c[k] for k in ROLES) == c["Q"], "row roles do not partition Q")
    return c, set(e["dirty_after"])


def report(root, manifest, cohort, native_dir=None):
    manifest_blob = manifest.read_bytes()
    check(sha256(manifest_blob).hexdigest() == MANIFEST_SHA, "changed manifest")
    meta = json.loads(manifest_blob)
    transport = (root/"bundle/bundle.sha256").is_file()
    close_path = root/"complete.json"
    close = json.loads(close_path.read_bytes()) if close_path.is_file() else None
    if transport:
        uuid, scores = a1_registration(root)
        q, executor, run_source = 8, A1_EXECUTOR, A1_BUNDLE
        native_dir = native_dir or root/"bundle/reference"
        check(close is not None and close["completed"] == 90
              and close["source_sha256"] == run_source
              and set(close["case_sha256"]) == {f"{i:02}.json" for i in range(90)}, "A1 final seal")
    else:
        registered = json.loads((root/"registration.json").read_bytes())
        campaign = (root.parent/"registration.json").read_bytes()
        executor = digest(root/"src"/EXECUTOR_FILE)
        check(registered["mode"] == "full_block" and registered["cap"] == CAP
              and sha256(campaign).hexdigest() == registered["parent_sha"]
              and json.loads(campaign)["executor_sha256"] == executor, "registration")
        q, run_source = registered["offer"], digest(root/"source.sha256")
    ids = {"original30": range(30), "new60": range(30, 90), "frozen90": range(90)}[cohort]
    by_id = {t["index"]: t for t in meta["samples"]}
    check(meta["samples_sha256"] == SAMPLES and set(by_id) == set(range(90))
          and q in (4, 8, 16, 32, 64), "manifest or budget")
    measures, total, checksums, received_checksums, failures, identities = {}, Counter(), {}, {}, [], set()
    for i in ids:
        seal = root/"cases"/f"{i:02}.json"
        if not seal.is_file():
            continue
        data = seal.read_bytes()
        r = json.loads(data)
        check(all(r[k] == by_id[i][k] for k in ("index", "task_id", "seed", "dataset"))
              and r["offer"] == q and r["mode"] == "full_block" and r["samples_sha256"] == SAMPLES
              and r["source_sha256"] == run_source, (i, "identity"))
        if r["status"] != "complete":
            failures.append(i)
            continue
        if transport:
            check((r["uuid"], r["checkpoint"], r["scorer"]) == (uuid, A1_PAYLOAD, SCORER)
                  and all(r[k] == scores[i][k] for k in ("N", "F", "Q", "correct")),
                  (i, "A1 checkpoint/scorer or scored seal mismatch"))
        received_checksums[i], checksums[i] = case_digests(
            data, r, close["case_sha256"][seal.name] if close is not None else None, transport=transport)
        identities.add((r["checkpoint"],r["scorer"]))
        pre = r["prefill"]
        check(pre["decoder_f"] == 1 and pre["decoder_q"] > 0 and pre["encoder_f"] == pre["encoder_q"] == 0,
              (i, "not a fresh prefill"))
        if native_dir:
            ref = (native_dir/f"{i:02}-native32.json").read_bytes()
            native = json.loads(ref)
            token_ids = native["prompt_ids"]
            check(sha256(ref).hexdigest() == r["ref_sha256"] and native["index"] == i
                  and native["generation_cap"] == CAP and len(token_ids) == pre["decoder_q"]
                  and sha256(json.dumps(token_ids,separators=(",",":")).encode()).hexdigest() == r["prompt_sha256"],
                  (i, "wrong native prompt"))
        raw = tape_path(root, r, i, transport=transport)
        receipt = sha256()
        c, debt, starts, peak = Counter(), set(), set(), 0
        ledger, phases, widths = Counter(), Counter(), Counter()
        writes = {}
        origin = pre["decoder_q"]
        with raw.open("rb") as stream:
            for j, line in enumerate(stream):
                receipt.update(line)
                t = json.loads(line)
                check(t["event"]["tick"] == j, (i, "tick"))
                billed, debt = inspect_tick(t, q, debt, origin, origin+CAP)
                c.update(billed)
                peak = max(peak,billed["Q"])
                if transport:
                    phases[t["event"]["phase"]] += 1
                    widths[str(billed["Q"])] += 1
                    writes.update((int(p), token) for p, token in t["written"].items())
                    ledger.update(F=1, Q=billed["Q"], encoder_F=0, encoder_Q=0,
                                  paid_rows=billed["writeback"], written_rows=len(t["written"]))
                if billed["admission_F"]:
                    start = t["event"]["admission_start"]
                    check(start not in starts, (i, "repeated admission"))
                    starts.add(start)
                    ledger.update(build_F=1, build_Q=billed["Q"], new_block_Q=billed["new_block"],
                                  over_offer_Q=billed["over_offer_Q"])
        check(receipt.hexdigest() == r["raw_sha256"] and not debt and c["F"] == r["F"] > 0
              and c["Q"] == r["Q"] and c["over_offer_Q"] == r["bills"].get("over_offer_Q", 0)
              and c["new_block"] == r["bills"].get("new_block_Q", 0)
              and c["admission_F"] == r["bills"].get("build_F", 0)
              and r["end_reason"] in ("cap","eos") and type(r["correct"]) is bool, (i, "unreconciled endpoint"))
        if transport:
            check(r["bills"] == dict(ledger) and r["phases"] == dict(phases) and r["widths"] == dict(widths)
                  and r["decoder_F"] == r["F"] and r["decoder_Q"] == r["Q"], (i, "A1 counts"))
            output_ids(r, writes)
        measures[i] = dict(N=r["N"], F=c["F"], Q=c["Q"], correct=r["correct"], terminal=r["end_reason"],
                           seed=r["seed"], task_id=r["task_id"], peak=peak, over=c["over_offer_Q"],
                           build_Q=c["new_block"], decoder_F=c["F"], encoder_F=0)
        total.update(c)
        total.update(prefill_F=pre["decoder_f"], prefill_Q=pre["decoder_q"])
    check(len(identities)<=1,"mixed checkpoint or scorer")
    missing = sorted(set(ids) - measures.keys() - set(failures))
    out = dict(cohort=cohort, q=q, family=FAMILY, model="llada21", manifest_sha256=MANIFEST_SHA,
               executor_sha256=executor, checkpoint=next(iter(identities), (None,None))[0],
               scorer=next(iter(identities), (None,None))[1], native_prompts_verified=native_dir is not None,
               verified=len(measures), missing=missing, sealed_failures=failures,
               runner_sources_sha256=run_source, seal_digests=checksums,
               received_seal_digests=received_checksums, complete=not (missing or failures))
    if out["complete"]:
        if cohort == "frozen90":
            check(close is not None and close["completed"] == 90 and close["source_sha256"] == run_source and
                  close["case_sha256"] == {f"{i:02}.json":v for i,v in checksums.items()}, "final seal")
        n = sum(m["N"] for m in measures.values())
        out.update(counts=dict(total), shares={k:total[k]/total["Q"] for k in ROLES},
                   correct=sum(m["correct"] for m in measures.values()), N=n,
                   TPF=mean(m["N"]/m["F"] for m in measures.values()), TE=n/total["Q"],
                   terminals=dict(Counter(m["terminal"] for m in measures.values())),
                   peak=max(m["peak"] for m in measures.values()), cases=measures)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("manifest", type=Path)
    p.add_argument("cohort", choices=("original30","new60","frozen90"))
    p.add_argument("--native-dir", type=Path, help="verify native prompt token IDs; required for freeze gate")
    a = p.parse_args()
    print(json.dumps(report(a.root, a.manifest, a.cohort, a.native_dir), indent=2))
