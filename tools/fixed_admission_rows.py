"""Audit row roles on fixed-offer LLaDA full-admission tapes, without model inference."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

B, MASK = 32, 156895
ROLES = ("writeback", "new_block", "other_mask", "clean_decoded")
SAMPLES = "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed"
MANIFEST_SHA = "2e67d12f5045d486d1676a4b9cd35718b0791ad1b2a3c81fdec86a27bd773e62"
FAMILY = "live-full-successor-v1"


def check(condition, note):
    if not condition:
        raise ValueError(note)


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
          and e["layer_physical_q"] == [len(rows)] * 20 and e["encoder_F"] == 0
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
                admission_F=bool(new), over_offer_Q=e["over_offer_Q"])
    check(sum(c[k] for k in ROLES) == c["Q"], "row roles do not partition Q")
    return c, set(e["dirty_after"])


def report(root, manifest, cohort, native_dir=None):
    manifest_blob = manifest.read_bytes()
    check(sha256(manifest_blob).hexdigest() == MANIFEST_SHA, "changed manifest")
    meta = json.loads(manifest_blob)
    registered = json.loads((root/"registration.json").read_bytes())
    campaign = (root.parent/"registration.json").read_bytes()
    executor = sha256((root/"src/experiments/t2t/serving_slots/llada_response_executor.py").read_bytes()).hexdigest()
    check(meta["samples_sha256"] == SAMPLES and registered["mode"] == "full_block" and registered["cap"] == 2048
          and sha256(campaign).hexdigest() == registered["parent_sha"]
          and json.loads(campaign)["executor_sha256"] == executor, "registration")
    q = registered["offer"]
    ids = {"original30": range(30), "new60": range(30, 90), "frozen90": range(90)}[cohort]
    by_id = {t["index"]: t for t in meta["samples"]}
    check(set(by_id) == set(range(90)) and q in (4, 8, 16, 32, 64), "manifest or budget")
    measures, total, checksums, failures, identities = {}, Counter(), {}, [], set()
    run_source = sha256((root/"source.sha256").read_bytes()).hexdigest()
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
        identities.add((r["checkpoint"],r["scorer"]))
        pre = r["prefill"]
        check(pre["decoder_f"] == 1 and pre["decoder_q"] > 0 and pre["encoder_f"] == pre["encoder_q"] == 0,
              (i, "not a fresh prefill"))
        if native_dir:
            ref = (native_dir/f"{i:02}-native32.json").read_bytes()
            native = json.loads(ref)
            token_ids = native["prompt_ids"]
            check(sha256(ref).hexdigest() == r["ref_sha256"] and native["index"] == i
                  and native["generation_cap"] == 2048 and len(token_ids) == pre["decoder_q"]
                  and sha256(json.dumps(token_ids,separators=(",",":")).encode()).hexdigest() == r["prompt_sha256"],
                  (i, "wrong native prompt"))
        raw = (root/r["raw"]).resolve()
        check(root.resolve() in raw.parents, (i, "raw path"))
        receipt = sha256()
        c, debt, starts, peak = Counter(), set(), set(), 0
        with raw.open("rb") as stream:
            for j, line in enumerate(stream):
                receipt.update(line)
                t = json.loads(line)
                check(t["event"]["tick"] == j, (i, "tick"))
                origin = r["prefill"]["decoder_q"]
                billed, debt = inspect_tick(t, q, debt, origin, origin+2048)
                c.update(billed)
                peak = max(peak,billed["Q"])
                if billed["admission_F"]:
                    start = t["event"]["admission_start"]
                    check(start not in starts, (i, "repeated admission"))
                    starts.add(start)
        check(receipt.hexdigest() == r["raw_sha256"] and not debt and c["F"] == r["F"] > 0
              and c["Q"] == r["Q"] and c["over_offer_Q"] == r["bills"].get("over_offer_Q", 0)
              and c["new_block"] == r["bills"].get("new_block_Q", 0)
              and c["admission_F"] == r["bills"].get("build_F", 0)
              and r["end_reason"] in ("cap","eos") and type(r["correct"]) is bool, (i, "unreconciled endpoint"))
        checksums[i] = sha256(data).hexdigest()
        measures[i] = dict(N=r["N"], F=c["F"], Q=c["Q"], correct=r["correct"], terminal=r["end_reason"],
                           seed=r["seed"], task_id=r["task_id"], peak=peak, over=c["over_offer_Q"],
                           build_Q=c["new_block"], decoder_F=c["F"], encoder_F=0)
        total.update(c)
    check(len(identities)<=1,"mixed checkpoint or scorer")
    missing = sorted(set(ids) - measures.keys() - set(failures))
    out = dict(cohort=cohort, q=q, family=FAMILY, model="llada21", manifest_sha256=MANIFEST_SHA,
               executor_sha256=executor, checkpoint=next(iter(identities), (None,None))[0],
               scorer=next(iter(identities), (None,None))[1], native_prompts_verified=native_dir is not None,
               verified=len(measures), missing=missing, sealed_failures=failures,
               runner_sources_sha256=run_source, seal_digests=checksums, complete=not (missing or failures))
    if out["complete"]:
        if cohort == "frozen90":
            close = json.loads((root/"complete.json").read_bytes())
            check(close["completed"] == 90 and close["source_sha256"] == run_source and
                  close["case_sha256"] == {f"{i:02}.json":v for i,v in checksums.items()}, "final seal")
        n = sum(m["N"] for m in measures.values())
        out.update(counts=dict(total), shares={k:total[k]/total["Q"] for k in ROLES},
                   correct=sum(m["correct"] for m in measures.values()), N=n,
                   TPF=mean(m["N"]/m["F"] for m in measures.values()), TE=n/total["Q"],
                   cases=measures)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("manifest", type=Path)
    p.add_argument("cohort", choices=("original30","new60","frozen90"))
    p.add_argument("--native-dir", type=Path, help="verify native prompt token IDs; required for freeze gate")
    a = p.parse_args()
    print(json.dumps(report(a.root, a.manifest, a.cohort, a.native_dir), indent=2))
