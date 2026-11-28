"""LLaDA Q8 successor fallback ablation; emit receipts, never write on A1."""
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

INDICES = (7, 13, 25, 33, 55, 71)
UUID = "GPU-861c9b5d-1e73-2550-f832-b73988bc64ff"
OLD_REG = "afbbb74f7b7e5d5246eda6809244da73c91371a341ac433e72b1d56a9d63736b"
OLD_RAW = "63d9cd573323e9c1a0fa4cad6b97a2d5713ee6ed8173f7ce311537e6aaa9a4b8"
SCORER = "a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692"
SAMPLES = "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed"
OLD = Path("/gs/work/llada21-ibs-q8-tail-overlap-a1-v1")
SRC = Path("/gs/code/formal90-7a38ffc/src")
TASKS = Path("/gs/code/samples.json")
CKPT = Path("/gs/models/LLaDA2.1-mini")
KEYS = ("index", "dataset", "task_id", "seed")
CONVENTION = "native gathered argmax softmax; score >= .9 cast to score dtype before force-top1"


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def emit(kind, **fields):
    print(json.dumps(dict(kind=kind, **fields), sort_keys=True, allow_nan=False), flush=True)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def old_rows(ours, sealed):
    check((sealed / "runner.exit").read_text().strip() == "0", "original exit is not zero")
    raw = (sealed / "overlap.ndjson").read_bytes()
    check(ours.sha(raw) == OLD_RAW, "original tape digest drift")
    meta = json.loads((sealed / "launch.meta.json").read_bytes())
    check(meta["registration_sha256"] == OLD_REG, "original registration drift")
    rows = [json.loads(line) for line in raw.splitlines()]
    end = rows.pop()
    check(end.get("complete") is True and end["rows"] == len(rows) == 18, "original run incomplete")
    by = {(r["index"], r["arm"]): r for r in rows}
    check(set(by) == {(i, a) for i in INDICES for a in ("overlap", "stock", "noop")},
          "incorrect sealed case set")
    for i in INDICES:
        for a in ("overlap", "stock", "noop"):
            r = by[i, a]
            check(r["gpu_uuid"] == UUID and r["registration_sha256"] == OLD_REG
                  and r["q"] == 8 and r["samples_sha256"] == SAMPLES, "wrong original arm")
        check(ours.pristine(by[i, "stock"]) == ours.pristine(by[i, "noop"]), "stock/no-op drift")
    return by


def prepare(helper, sealed):
    ours = import_file("old_runner", helper)
    by = old_rows(ours, sealed)
    print(json.dumps(dict(original_exit=0, raw_sha256=OLD_RAW, registration=OLD_REG, indices={
        str(i): dict(expected=ours.identity(ours.pristine(by[i, "overlap"])),
                     prompt=by[i, "overlap"]["prompt_ids_sha256"],
                     **{k: by[i, "overlap"][k] for k in KEYS}) for i in INDICES}),
        sort_keys=True))


def observer(native_dual, tape, mask_id, filtered):
    """Observe the scores fed to the pinned native selector, not a recomputation."""
    def dual(engine, logits, start, tail):
        unmask = engine.unmask
        original = unmask._apply_dynamic_from_scores
        captured = []
        def capture(scores, mask, *args, **kwargs):
            captured.append(scores.clone())
            return original(scores, mask, *args, **kwargs)
        with patch.object(unmask, "_apply_dynamic_from_scores", capture):
            proposed, take = native_dual(engine, logits, start, tail)
        scores, = captured
        config = unmask.dllm_config
        check(config.confidence_threshold == .9 and config.block_length == 32,
              "unexpected native threshold or block")
        loc = tail - (start + 32)
        check(loc.tolist() == list(range(len(tail))) and 0 < len(tail) < 8,
              "expected a contiguous paid partial successor")
        threshold = scores.new_tensor(config.confidence_threshold)
        paid_scores = scores[loc]
        check(bool(paid_scores.isfinite().all()), "non-finite native confidence")
        natural = paid_scores >= threshold
        ordinary = proposed != mask_id
        forced = scores.topk(1, dim=-1).indices[0]
        check(bool((take == (natural | (loc == forced))).all()), "native selection mismatch")
        has_qualifier = bool((natural & ordinary).any())
        veto = take & ordinary & (not has_qualifier)
        check(not bool((veto & natural).any()) and int(veto.sum()) <= 1, "not a forced-only M2T")
        after = take & ~veto if filtered else take
        positions = tail.tolist()
        tape.append(dict(
            start=start, positions=positions, dtype=str(scores.dtype),
            threshold_hex=float(threshold.item()).hex(), forced_local=int(forced.item()),
            has_ordinary_qualifier=has_qualifier,
            veto_rows=tail[veto].tolist(),
            rows=[dict(position=p, proposal=token, score_hex=float(score).hex(),
                       clears=above, ordinary=valid, native_take=base, actual_take=new,
                       material_if_promoted=base and valid)
                  for p, token, score, above, valid, base, new in
                  zip(positions, proposed.tolist(), paid_scores.tolist(), natural.tolist(),
                      ordinary.tolist(), take.tolist(), after.tolist(), strict=True)]))
        return proposed, after
    return dual


def reconcile(row, tape):
    cursor = 0
    for tick, event in enumerate(row["events"]):
        s = event.get("successor")
        if s is None:
            continue
        t = tape[cursor]
        cursor += 1
        check(s["tail_rows"] == t["positions"]
              and s["tail_proposed"] == [r["proposal"] for r in t["rows"]]
              and t["start"] == event["block"] * 32, "successor tape alignment")
        actual = [r["position"] for r in t["rows"] if r["actual_take"] and r["ordinary"]]
        check(actual == (s["tail_writes"] if s["promoted"] else s["discarded_writes"]),
              "measured write does not match observed native decision")
        t.update(tick=tick, promoted=s["promoted"],
                 base_material=[r["position"] for r in t["rows"]
                                if s["promoted"] and r["material_if_promoted"]])
    check(cursor == len(tape), "missing successor tape")


def first_effect(tape):
    return next((t for t in tape if set(t["base_material"]) & set(t["veto_rows"])), None)


def align(ours, base, filtered, paper, filter_tape):
    first = first_effect(paper)
    if first is None:
        check(ours.pristine(base) == ours.pristine(filtered), "unexposed case changed")
        return None
    j = first["tick"]
    check(j + 1 < len(base["events"]) and j + 1 < len(filtered["events"]),
          "missing next physical round")
    check(base["events"][:j] == filtered["events"][:j]
          and base["forward_inputs"][:j+1] == filtered["forward_inputs"][:j+1]
          and base["calls"][:j+1] == filtered["calls"][:j+1],
          "first fork is not on the observed joint call")
    a, b = base["events"][j], filtered["events"][j]
    cut = set(first["veto_rows"]) & set(first["base_material"])
    check(len(cut) == 1, "expected one material forced write")
    expected = dict(a, written_absolute=[p for p in a["written_absolute"] if p not in cut])
    expected["successor"] = dict(a["successor"], tail_writes=[p for p in a["successor"]["tail_writes"]
                                                           if p not in cut])
    check(expected == b, "first joint call changed more than the forced successor write")
    t = next(t for t in filter_tape if t["tick"] == j)
    check(t == dict(first, rows=[dict(x, actual_take=False) if x["position"] in cut else x
                                 for x in first["rows"]]), "confidence or candidate drift on first fork")
    check(cut <= set(base["events"][j+1]["dirty_before"])
          and not (cut & set(filtered["events"][j+1]["dirty_before"])), "successor debt mismatch")
    return first


def run(reference, deadline, runner_sha):
    ours = import_file("old_runner", OLD / "runner_a1.py")
    reg = json.loads((OLD / "registration.json").read_bytes())
    static = json.loads((OLD / "static_cases.json").read_bytes())
    args = SimpleNamespace(src=SRC, work=OLD, samples=TASKS, checkpoint=CKPT,
                           deadline=reg["deadline"], registration=OLD/"registration.json")
    check(ours.identity(reg) == OLD_REG and ours.sha(Path(__file__)) == runner_sha
          and reference["raw_sha256"] == OLD_RAW and reference["original_exit"] == 0
          and reference["registration"] == OLD_REG
          and socket.gethostname().split(".")[0] == "n1", "preflight identity mismatch")
    ours.check_registration(args, reg, static)
    sys.path[:0] = [str(OLD), str(SRC), "/gs/env/fixed-baseline-deps"]
    ours.preflight(args)
    ours.check_idle(deadline)
    from bdlm_serving.methods.optimus import fixed, upstream
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from experiments.t2t.serving_slots import ibs_prefix_pair as pair, shared_llada_worker as worker
    import torch
    import transformers
    check(str(torch.__version__) == "2.6.0+cu124" and transformers.__version__ == "4.57.1",
          "unexpected pinned runtime")
    upstream.control_directory = lambda name: OLD / "controls/Optimus"
    module = import_file("bdlm_serving.methods.t2t.close_overlap", OLD / "close_overlap.py")
    import bdlm_serving.methods.t2t as parent
    parent.close_overlap = module
    import fused
    native_dual = fused.dual_sample
    check(ours.sha(Path(fused.__file__)) == reg["sources"]["fused.py"], "fused code drift")
    tasks = load_tasks(TASKS)
    handle = worker.load(CKPT)
    mask_id = handle["mask_id"]
    emit("loaded", pid=os.getpid(), gpu_uuid=UUID, gpu_index=1, gpu=torch.cuda.get_device_name(),
         runner_sha256=runner_sha, original_raw_sha256=OLD_RAW, threshold_convention=CONVENTION,
         checkpoint_fingerprint=handle["fingerprint"], weights=handle["weights"], mask_id=mask_id)

    def arm(i, name, prompt):
        check(time.time() < deadline - 30, "bounded deadline")
        tape, votes = [], []
        dual = native_dual if name == "original" else observer(
            native_dual, tape, mask_id, name == "filtered")
        with patch.object(fused, "dual_sample", dual):
            patched_run = fused.install_run(fixed, arm="overlap", receipt=votes)
            with patch.object(fixed.FixedOptimus, "run", patched_run):
                row = pair.generate(handle["model"], handle["tok"], tasks[i], prompt, 2048, 0)
        row.update(q=8, block=32, cap=2048, samples_sha256=SAMPLES,
                   prompt_ids_sha256=reference["indices"][str(i)]["prompt"])
        check(row["max_q"] <= 8 and row["verified_layers"] == 20
              and not row["remaining_masks"], "unexpected physical bill or delivered MASK")
        if name != "original":
            reconcile(row, tape)
        print(f"SEAL {i:02d} {name} N/F/Q={row['output_tokens']}/"
              f"{row['generation_forwards']}/{row['generation_query_rows']}",
              file=sys.stderr, flush=True)
        return row, tape

    qualified, pending, complete = False, [], 0
    def finish(i, prompt, base, tape):
        nonlocal qualified, complete
        value, actual = arm(i, "filtered", prompt)
        first = align(ours, base, value, tape, actual)
        if not qualified:
            check(first is not None, "initial paper-tape gate did not expose a write")
            emit("qualified", index=i, first=first, masks_preserved=True,
                 same_joint_forward=True, next_step_debt_checked=True)
            qualified = True
        emit("arm", arm="filtered", receipt=value, paper_tape=actual, first_effect=first)
        complete += 1

    for i in INDICES:
        expected = reference["indices"][str(i)]
        prompt, _ = ours.local_prompt(handle["tok"], tasks[i], static["cases"][str(i)])
        check(all(tasks[i][k] == expected[k] for k in KEYS), "case/seed mismatch")
        baseline, _ = arm(i, "original", prompt)
        check(ours.identity(ours.pristine(baseline)) == expected["expected"],
              f"sealed same-A1 fused replay failed #{i:02d}")
        obs, tape = arm(i, "observer", prompt)
        check(ours.pristine(obs) == ours.pristine(baseline), f"observer changed fused #{i:02d}")
        emit("arm", arm="original", receipt=baseline, historical_exact=True)
        emit("arm", arm="observer", receipt=obs, paper_tape=tape, historical_exact=True)
        if qualified or first_effect(tape) is not None:
            finish(i, prompt, baseline, tape)
            for item in pending:
                finish(*item)
            pending.clear()
        else:
            pending.append((i, prompt, baseline, tape))
    emit("complete", status="frozen-six" if qualified else "no-go-no-material-write",
         complete=complete, observed=len(INDICES))


def report(sealed, helper, receipts, samples):
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen90 import load_tasks
    ours = import_file("old_runner", helper)
    old = old_rows(ours, sealed)
    check(ours.sha(samples) == SAMPLES and ours.sha(Path(scoring.__file__)) == SCORER,
          "scoring inputs changed")
    stream = [json.loads(x) for x in receipts.read_bytes().splitlines()]
    records = {(x["receipt"]["index"], x["arm"]): x for x in stream if x.get("kind") == "arm"}
    end = stream[-1]
    n = 18 if end.get("status") == "frozen-six" else 12
    check(end["kind"] == "complete" and end["observed"] == 6 and len(records) == n
          and sum(x.get("kind") == "arm" for x in stream) == n, "unsealed new tape")
    tasks = load_tasks(samples)
    cases, totals, hist = [], {}, Counter()
    for i in INDICES:
        case = dict(index=i)
        base = records[i, "original"]["receipt"]
        obsv = records[i, "observer"]
        check(ours.pristine(base) == ours.pristine(obsv["receipt"])
              == ours.pristine(old[i, "overlap"]), "fused replay disagrees on scoring host")
        if (i, "filtered") in records:
            filtered = records[i, "filtered"]
            check(align(ours, base, filtered["receipt"], obsv["paper_tape"],
                        filtered["paper_tape"]) == filtered["first_effect"], "first effect did not replay")
        for arm, r in [("stock", old[i, "stock"])] + [
            (a, records[i, a]["receipt"]) for a in ("original", "observer", "filtered")
            if (i, a) in records
        ]:
            check(all(tasks[i][k] == r[k] for k in KEYS), "incorrect scorer identity")
            f, q, nn = r["generation_forwards"], r["generation_query_rows"], r["output_tokens"]
            correct = r["status"] == "success" and bool(scoring.score(r["output_text"], tasks[i])["correct"])
            c = case[arm] = dict(N=nn, F=f, Q=q, correct=correct, status=r["status"],
                                 same_ids_as_stock=r["output_ids"] == old[i, "stock"]["output_ids"],
                                 same_ids_as_original=r["output_ids"] == records[i, "original"]["receipt"]["output_ids"])
            t = totals.setdefault(arm, Counter())
            t.update(N=nn, F=f, Q=q, correct=int(correct), tpf_sum=nn/f, count=1)
        t = records[i, "observer"]["paper_tape"]
        first = first_effect(t)
        case["first_material_tick"] = first and first["tick"]
        case["material_opportunities"] = sum(bool(set(x["veto_rows"]) & set(x["base_material"])) for x in t)
        hist.update(joints=len(t), cases_exposed=int(first is not None),
                    material_opportunities=case["material_opportunities"],
                    qualifying_joints=sum(x["has_ordinary_qualifier"] for x in t),
                    forced_no_qualifier_but_no_write=sum(not x["has_ordinary_qualifier"]
                                                        and not x["base_material"] for x in t))
        cases.append(case)
    for v in totals.values():
        v["macro_tpf"] = v.pop("tpf_sum") / v["count"]
        v["pooled_te"] = v["N"] / v["Q"]
    emit("summary", cases=cases, totals=totals, original_paper_tape=hist,
         decision=end["status"], qualification=next((x for x in stream if x.get("kind") == "qualified"), None),
         loaded=next(x for x in stream if x.get("kind") == "loaded"), raw_sha256=ours.sha(receipts))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd, args in (("prepare", ("helper", "sealed")), ("run", ("reference", "deadline", "runner_sha")),
                      ("report", ("sealed", "helper", "receipts", "samples"))):
        q = sub.add_parser(cmd)
        for name in args:
            q.add_argument(name, type=int if name == "deadline" else str if name == "runner_sha" else Path)
        q.set_defaults(func=globals()[cmd], keys=args)
    a = p.parse_args()
    a.func(*(getattr(a, k) if k != "reference" else json.loads(a.reference.read_bytes()) for k in a.keys))
