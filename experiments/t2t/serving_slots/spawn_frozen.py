"""Protected Q64 spawn-gate extension on the AllSlots GPU4 launcher.

Two causal arms only: ``eager-causal`` (the existing AllSlots ``fixed64``) and
``progress80-causal`` (``SpawnGatedKernel`` threshold 0.8).  The original30
lease runs first and is sealed by an audit-only qualification; the new60 lease
requires it.  Every generation forward is audited for the 64 offer and its
physical width; both leases are recorded as original30/new60/full90.  This
module never registers or queues itself; the owner does that after AllSlots90.

Owner entry points (never run by the reviewer):
  ../venv/bin/python -u -m experiments.t2t.serving_slots.spawn_frozen queue
  ../venv/bin/python -u -m experiments.t2t.serving_slots.spawn_frozen score
"""
from __future__ import annotations

from collections import Counter
import hashlib
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import time

from experiments.t2t.serving_slots import llada_all_slots90 as reg

STUDY = reg.DATA / "llada21-q64-spawn-frozen-v1"
PARENT = "prefix90-llada-all-slots90"
ARMS = {"eager-causal": (None, False), "progress80-causal": (0.8, False)}
GROUPS = (("original30", range(0, 30)), ("new60", range(30, 90)))
COHORTS = {"original30": range(0, 30), "new60": range(30, 90), "full90": range(90)}
TRACE = (64,)
FIXED64_GROUP = next(i for i, (arms, _) in enumerate(reg.GROUPS)
                     if arms == ("fixed64",))


def spec(i):
    name, indices = GROUPS[i]
    return dict(model="llada21", suite="frozen90", gpu=4, gpu_uuid=reg.UUID,
                gen_length=reg.CAP, samples_sha256=reg.SAMPLES_SHA,
                launcher_sha256=reg.LAUNCHER_SHA, group=i, cohort=name,
                indices=list(indices), fuse_growth=True,
                arms={arm: dict(trace=list(TRACE), opened=opened,
                                spawn_threshold=threshold, mask="causal")
                      for arm, (threshold, opened) in ARMS.items()})


def audit(row, arm):
    """Audit the 64 offer and physical width of every forward; track commits."""
    threshold, opened = ARMS[arm]
    reg.require(row["policy"] == arm and row["model"] == "llada21"
                and row["status"] == "success", "Wrong arm/model/status")
    reg.require(not opened and row["opened_edges"] == 0,
                "Causal arm must not open cross-block edges")
    prompt = len(row["prompt_ids"])
    prefill = prompt // 32 * 32
    events = row["events"]
    gen = events[int(prefill > 0):]
    reg.require(gen and row["prefill_rows"] == prefill,
                "Missing generation or wrong prefill")
    reg.require((row["first"] is None) == (row["opportunities"] == 0),
                "First fingerprint disagrees with opportunities")
    widths, stages, queries, admissions = Counter(), Counter(), Counter(), Counter()
    expected_block = prompt // 32
    for j, event in enumerate(events):
        pos, q, end = event["absolute_positions"], event["physical_q"], event["visible_kv"]
        reg.require(event["block"] == expected_block and pos == sorted(set(pos))
                    and len(pos) == event["actual_q"] == q == event["q_tokens"]
                    and 0 <= pos[0] <= pos[-1] < end and event["F"] == 1
                    and event["layer_physical_q"] == [q] * reg.LAYERS,
                    "Invalid physical width or block")
        if prefill and j == 0:
            reg.require(event["phase"] == "prefill" and pos == list(range(prefill))
                        and end == prefill and event.get("offer_tick") is None,
                        "Unexpected prompt forward")
            continue
        tick = j - int(prefill > 0)
        reg.require(event["phase"] != "prefill" and event["offer_tick"] == tick
                    and event["q_budget"] == event["nominal_q"] == TRACE[tick % len(TRACE)]
                    and event["previous_offer"] == (
                        TRACE[(tick - 1) % len(TRACE)] if tick else None),
                    "Wrong external offer clock")
        reg.require(event["overflow"] == max(0, q - 64)
                    and event["underfill"] == max(0, 64 - q),
                    "Unexplained physical overrun/underfill")
        sampled = ([] if event["phase"] == "commit"
                   else sorted(set(pos) - set(event["refresh_only_positions"])))
        reg.require(event["sampled_q"] == len(event["sampled_positions"]) == len(sampled)
                    and set(event["sampled_positions"]) == set(sampled),
                    "Wrong sampled/scatter partition")
        reg.require(set(event["refresh_only_positions"]) <= set(pos)
                    and set(event["refreshed_pending"]) <= set(pos),
                    "Bad pending rows")
        reg.require(event.get("tail_admission") in (None, "open", "defer"),
                    "Bad tail admission")
        if event.get("tail_admission") in ("open", "defer"):
            admissions[event["tail_admission"]] += 1
        widths[str(q)] += 1
        stages[event["stage"]] += 1
        queries[event["stage"]] += q
        if event.get("top_action") == "commit_head":
            expected_block += 1
    f = len(gen)
    q = sum(event["physical_q"] for event in gen)
    n = row["output_tokens"]
    ids, eos, special = row["output_ids"], row["eos_id"], row["special_ids"]
    reg.require(bool(ids) and len(ids) <= reg.CAP and eos not in ids[:-1]
                and row["eos"] == (ids[-1] == eos) and row["truncated"] != row["eos"]
                and (row["eos"] or len(ids) == reg.CAP)
                and n == sum(t not in special for t in ids),
                "Wrong output/EOS convention")
    reg.require(row["generation_forwards"] == f
                and row["generation_physical_padded_rows"] == q
                and row["generation_query_rows"] == q
                and row["tpf"] == n / f and row["token_efficiency"] == n / q,
                "Wrong packaged metrics")
    reg.require(row["q_histogram"] == widths
                and row["max_q"] == max(map(int, widths))
                and row["forwards_by_phase"] == stages
                and row["q_by_phase"] == queries,
                "Wrong physical histogram")
    return dict(index=row["index"], N=n, F=f, Q=q,
                opened=admissions["open"], deferred=admissions["defer"],
                opportunities=row["opportunities"],
                overflow=sum(event["overflow"] for event in gen))


def verify_worker(study, i, verify_bytes=False):
    out = reg.folder(study, i)
    _, indices = GROUPS[i]
    work = reg.read(out / "worker-complete.json")
    reg.require(reg.read(out / "run.json") == spec(i)
                and work["spec_sha256"] == reg.digest(spec(i))
                and work["files"].keys() == {
                    reg.filename(j, arm) for j in indices for arm in ARMS},
                "Incomplete or wrong lease manifest")
    if verify_bytes:
        for name, checksum in work["files"].items():
            reg.require(reg.sha(out / name) == checksum, f"Wrong request digest: {name}")
    return work


def return_receipt(study, i, previous):
    out = reg.folder(study, i)
    work = verify_worker(study, i, verify_bytes=True)
    receipt = reg.read(out / "protected-complete.json")
    reg.require(receipt["worker_sha256"] == reg.digest(work)
                and receipt["previous_sha256"] == previous
                and receipt["spec_sha256"] == reg.digest(spec(i))
                and receipt["launcher_exit"] == 0
                and receipt["holder"]["uuid"] == reg.UUID
                and receipt["holder"]["sole_holder_pid"] > 0,
                "Missing or mismatched protected return receipt")
    return reg.digest(receipt)


def paired(on, off):
    a, b = audit(on, "eager-causal"), audit(off, "progress80-causal")
    reg.require(all(on[k] == off[k] for k in
                    (*reg.KEYS, "prompt_ids", "special_ids", "eos_id")),
                "Different request identity or input")
    return a, b


def qualify_original30(study):
    out = reg.folder(study, 0)
    work = verify_worker(study, 0, verify_bytes=True)
    receipt = reg.read(out / "protected-complete.json")
    reg.require(receipt["worker_sha256"] == reg.digest(work)
                and receipt["spec_sha256"] == reg.digest(spec(0))
                and receipt["holder"]["uuid"] == reg.UUID,
                "Missing original30 protected receipt")
    results = {arm: [] for arm in ARMS}
    eager_sources = Counter()
    for index in GROUPS[0][1]:
        rows = {arm: reg.read(out / reg.filename(index, arm)) for arm in ARMS}
        paired(rows["eager-causal"], rows["progress80-causal"])
        eager_sources[rows["eager-causal"].get("eager_source", "unknown")] += 1
        for arm in ARMS:
            results[arm].append(audit(rows[arm], arm))
    summary = {arm: dict(
        requests=len(rows), N=sum(r["N"] for r in rows), F=sum(r["F"] for r in rows),
        Q=sum(r["Q"] for r in rows), opened=sum(r["opened"] for r in rows),
        deferred=sum(r["deferred"] for r in rows),
        opportunities=sum(r["opportunities"] for r in rows))
        for arm, rows in results.items()}
    reg.save(out / "qualified.json", dict(
        original30=summary, eager_sources=dict(eager_sources),
        spec_sha256=reg.digest(spec(0)),
        worker_sha256=reg.digest(work), receipt_sha256=reg.digest(receipt)))
    return summary


def summarize(rows):
    """Aggregate the two causal arms; rows are {arm: {index: row with correct}}."""
    summary = []
    for cohort, indices in COHORTS.items():
        for arm in ARMS:
            selected = [rows[arm][j] for j in indices]
            audits = [audit(row, arm) for row in selected]
            n = sum(a["N"] for a in audits)
            f = sum(a["F"] for a in audits)
            q = sum(a["Q"] for a in audits)
            summary.append(dict(
                cohort=cohort, arm=arm, requests=len(selected),
                correct=sum(row["correct"] for row in selected),
                N=n, F=f, physical_Q=q,
                macro_tpf=sum(a["N"] / a["F"] for a in audits) / len(audits),
                token_efficiency=n / q if q else None,
                opened=sum(a["opened"] for a in audits),
                deferred=sum(a["deferred"] for a in audits),
                overflow=sum(a["overflow"] for a in audits)))
        eager = [rows["eager-causal"][j] for j in indices]
        progress = [rows["progress80-causal"][j] for j in indices]
        summary.append(dict(
            cohort=cohort, arm="progress80-causal", baseline="eager-causal",
            requests=len(indices),
            rescued=sum(not a["correct"] and b["correct"]
                        for a, b in zip(eager, progress)),
            lost=sum(a["correct"] and not b["correct"]
                     for a, b in zip(eager, progress)),
            macro_tpf_delta=(sum(b["tpf"] for b in progress)
                             - sum(a["tpf"] for a in eager)) / len(indices),
            physical_Q_delta=(
                sum(b["generation_physical_padded_rows"] for b in progress)
                - sum(a["generation_physical_padded_rows"] for a in eager))))
    return summary


def reuse_eager(index, task, prompt):
    """Reuse the qualified AllSlots ``fixed64`` eager row when it matches."""
    lease = reg.folder(reg.STUDY, FIXED64_GROUP)
    if not lease.is_dir():
        return None
    path = lease / reg.filename(index, "fixed64")
    if not path.is_file():
        raise ValueError(f"Missing qualified fixed64 eager row for index {index}")
    row = reg.read(path)
    if (row.get("policy") != "fixed64"
            or any(row.get(k) != task[k] for k in reg.KEYS)
            or row.get("prompt_ids") != prompt):
        raise ValueError(f"Qualified fixed64 eager row does not match index {index}")
    sealed_files = reg.read(lease / "worker-complete.json")["files"]
    reg.require(reg.sha(path) == sealed_files.get(path.name), "Fixed64 reference digest mismatch")
    row["policy"] = "eager-causal"
    if "first" not in row and "first_input_sha256" in row:
        row["first"] = row["first_input_sha256"]
    row.setdefault("opened_edges",
                   sum(event.get("opened_edges", 0) for event in row["events"]))
    row.setdefault("opportunities",
                   sum(event.get("mask_window") is not None for event in row["events"]))
    # The eager AllSlots control logs the executed state transition, not the
    # later gate annotation. Only these two fused transitions create a tail.
    for event in row["events"]:
        if event["phase"] == "prefill":
            continue
        action = event["top_action"]
        reg.require(action and action != "split_growth", "Missing or non-fused fixed64 action")
        if action in ("pair_bootstrap", "fused_growth"):
            event["tail_admission"] = "open"
    audit(row, "eager-causal")
    row["eager_source"] = "fixed64-reuse"
    return row


def generate(model, tok, task, prompt, arm):
    """Reuse the pilot's generate; add audit-only identity/special fields."""
    from experiments.t2t.serving_slots.llada_spawn_pilot import generate as pilot_generate

    row = pilot_generate(model, tok, task, prompt, arm)
    eos = tok.eos_token_id
    mask = tok.convert_tokens_to_ids("<|mask|>")
    row.update(model="llada21", dataset=task["dataset"], eos_id=eos,
               special_ids=sorted(set(tok.all_special_ids) | {mask, eos}))
    return row


def run(args, prepare):
    """Only called from the verified launcher worker. No task judging here."""
    import torch
    import transformers.utils as utils
    from typing import TypedDict
    if not hasattr(utils, "LossKwargs"):
        class LossKwargs(TypedDict, total=False):
            num_items_in_batch: int
        utils.LossKwargs = LossKwargs
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.runtime.serial import install_llada_rectangular_mask
    from experiments.t2t.serving_slots.top_transition90 import resolve_reference, validate_prompts

    i = int(os.environ["SPAWN_GROUP"])
    reg.require(0 <= i < len(GROUPS) and prepare is not None
                and os.environ["CUDA_VISIBLE_DEVICES"] == reg.UUID
                and args.output == reg.folder(STUDY, i)
                and args.samples == reg.DATA / "samples.json"
                and reg.sha(args.samples) == reg.SAMPLES_SHA,
                "Not the protected Q64 spawn worker")
    tasks = load_tasks(args.samples)
    tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True,
                                        local_files_only=True)
    reference = resolve_reference(args)
    validate_prompts(tok, tasks, reference)
    for item in reg.read(reg.REPO / "benchmarks/checkpoints/llada21.json"):
        path = args.checkpoint / item["name"]
        reg.require(path.stat().st_size == item["bytes"]
                    and ("sha256" not in item or reg.sha(path) == item["sha256"]),
                    "Wrong checkpoint")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    install_llada_rectangular_mask(model, allow_selected_cache=True)
    for parameter in model.parameters():
        parameter.data = parameter.data.clone()
    prepare(model)
    model.to("cuda")
    out = args.output
    out.mkdir(exist_ok=False)
    reg.save(out / "run.json", spec(i))
    files = {}
    _, indices = GROUPS[i]
    for index in indices:
        task = tasks[index]
        prompt = reg.read(reference / f"{index:02d}-native32.json")["prompt_ids"]
        eager = reuse_eager(index, task, prompt)
        if eager is None:
            eager = generate(model, tok, task, prompt, "eager-causal")
            eager["eager_source"] = "fresh"
        audit(eager, "eager-causal")
        progress = generate(model, tok, task, prompt, "progress80-causal")
        audit(progress, "progress80-causal")
        for arm, row in (("eager-causal", eager), ("progress80-causal", progress)):
            name = reg.filename(index, arm)
            reg.save(out / name, row)
            files[name] = reg.sha(out / name)
        print(f"lease {i:02d} complete request {index:02d}: {','.join(ARMS)}", flush=True)
    reg.save(out / "worker-complete.json",
             dict(spec_sha256=reg.digest(spec(i)), files=files))


def source_hashes():
    slots = reg.REPO / "experiments/t2t/serving_slots"
    paths = [slots / n for n in (
        "spawn_frozen.py", "llada_spawn.py", "llada_spawn_pilot.py",
        "llada_quote.py",
        "llada_all_slots90.py", "llada_all_slots.py", "llada_top_transition.py",
        "mask_pair_llada.py")]
    paths += [reg.REPO / "bdlm_serving/methods/t2t" / n
              for n in ("llada_dual.py", "llada_sparse.py")]
    return {p.name: reg.sha(p) for p in paths}


def predecessor():
    meta = reg.predecessor(reg.PRED)
    registration = reg.read(reg.STUDY / "registered.json")
    previous = reg.digest(registration)
    for i in range(len(reg.GROUPS)):
        previous = reg.return_receipt(reg.STUDY, i, previous)
    reg.require(reg.read(reg.STUDY / "complete.json") == dict(
        groups=len(reg.GROUPS), requests=90, policies=list(reg.ARMS), final_receipt_sha256=previous),
        "AllSlots predecessor did not finish all registered groups")
    return dict(session=PARENT, run=reg.STUDY.name, samples_sha256=reg.SAMPLES_SHA,
                policies=list(reg.ARMS), previous_sha256=previous,
                qualified_sha256=reg.sha(reg.STUDY / "registered.json"),
                all_slots90_meta=meta)


def protected(argv):
    i = int(os.environ["SPAWN_GROUP"])
    reg.require(argv in (["llada21", "4"], ["llada21", "4", "worker"])
                and 0 <= i < len(GROUPS) and not sys.flags.optimize,
                "Wrong protected invocation")
    registration = reg.read(STUDY / "registered.json")
    reg.require(registration["sources"] == source_hashes()
                and registration["predecessor"] == predecessor(),
                "Registration inputs changed")
    previous = reg.digest(registration)
    if i:
        qualified = reg.read(reg.folder(STUDY, 0) / "qualified.json")
        reg.require(qualified["spec_sha256"] == reg.digest(spec(0)),
                    "Missing original30 qualification")
        previous = return_receipt(STUDY, 0, previous)
    launcher = reg.ROOT / "launch_prefix90.py"
    source = launcher.read_bytes()
    key = b"from experiments.t2t.serving_slots.llada_reference import run"
    reg.require(hashlib.sha256(source).hexdigest() == reg.LAUNCHER_SHA
                and source.count(key) == 1, "Protected launcher changed")
    os.environ.update(PREFIX_MODE="llrefs", TRACE_OUT=str(reg.folder(STUDY, i)),
                      FROZEN90_SAMPLES=str(reg.DATA / "samples.json"),
                      LL_BASELINE_POLICIES="q64_spawn")
    replacement = b"from experiments.t2t.serving_slots.spawn_frozen import run"
    exec(compile(source.replace(key, replacement), str(launcher), "exec"),
         {"__name__": "__main__",
          "__file__": str(Path(__file__).resolve())})
    reg.require(len(argv) == 2, "Worker cannot certify holder return")
    returned = reg.holder()
    work = verify_worker(STUDY, i, verify_bytes=True)
    reg.save(reg.folder(STUDY, i) / "protected-complete.json",
             dict(worker_sha256=reg.digest(work), spec_sha256=reg.digest(spec(i)),
                  previous_sha256=previous, launcher_exit=0, holder=returned))
    if i == 0:
        qualify_original30(STUDY)


def queue():
    """Owner-only: wait for AllSlots90, verify it, then run both leases."""
    reg.require(not STUDY.exists()
                and reg.sha(reg.DATA / "samples.json") == reg.SAMPLES_SHA
                and reg.sha(reg.ROOT / "launch_prefix90.py") == reg.LAUNCHER_SHA,
                "Existing run or changed pinned inputs")
    while True:
        probe = subprocess.run(["tmux", "has-session", "-t", "=" + PARENT],
                               capture_output=True)
        if probe.returncode == 1 and (b"can't find session:" in probe.stderr
                                      or b"no server running" in probe.stderr):
            break
        probe.check_returncode()
        time.sleep(8)
    receipt = dict(predecessor=predecessor(), holder=reg.holder(),
                   sources=source_hashes())
    STUDY.mkdir(exist_ok=False)
    reg.save(STUDY / "registered.json", receipt)
    previous = reg.digest(receipt)
    for i in range(len(GROUPS)):
        with (STUDY / f"lease-{i:02d}.log").open("x") as log:
            subprocess.run([sys.executable, "-u", "-m",
                            "experiments.t2t.serving_slots.spawn_frozen",
                            "llada21", "4"], cwd=reg.REPO,
                           env=dict(os.environ, SPAWN_GROUP=str(i)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        previous = return_receipt(STUDY, i, previous)
    reg.save(STUDY / "complete.json", dict(
        groups=len(GROUPS), requests=90, policies=list(ARMS),
        final_receipt_sha256=previous))


def score(study):
    """Model-free: judge sealed artifacts, audit every row, record all cohorts."""
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.evaluation.scoring import score as judge

    samples = reg.DATA / "samples.json"
    reg.require(reg.sha(samples) == reg.SAMPLES_SHA, "Wrong sample digest")
    tasks = {task["index"]: task for task in load_tasks(samples)}
    registration = reg.read(study / "registered.json")
    previous = reg.digest(registration)
    for i in range(len(GROUPS)):
        previous = return_receipt(study, i, previous)
    reg.require(reg.read(study / "complete.json") == dict(
        groups=len(GROUPS), requests=90, policies=list(ARMS), final_receipt_sha256=previous),
        "No completed study")
    qualified = reg.read(reg.folder(study, 0) / "qualified.json")
    reg.require(qualified["spec_sha256"] == reg.digest(spec(0)),
                "Missing original30 qualification")
    rows = {arm: {} for arm in ARMS}
    for i, (_, indices) in enumerate(GROUPS):
        out = reg.folder(study, i)
        for index in indices:
            for arm in ARMS:
                row = reg.read(out / reg.filename(index, arm))
                reg.require(all(row[k] == tasks[index][k] for k in reg.KEYS), "Frozen task identity mismatch")
                row["correct"] = bool(judge(row["output_text"], tasks[index])["correct"])
                rows[arm][index] = row
    for arm in ARMS:
        for row in rows[arm].values():
            audit(row, arm)
    summary = summarize(rows)
    eager_sources = Counter(
        row.get("eager_source", "unknown") for row in rows["eager-causal"].values())
    reg.save(study / "summary.json", dict(
        study=str(study), arms=list(ARMS), cohorts=list(COHORTS),
        summary=summary, eager_sources=dict(eager_sources),
        qualified_sha256=reg.sha(reg.folder(study, 0) / "qualified.json"),
        final_receipt_sha256=previous))
    return summary


if __name__ == "__main__":
    if sys.argv[1:2] == ["score"]:
        score(STUDY)
    elif sys.argv[1:2] == ["queue"]:
        queue()
    else:
        protected(sys.argv[1:])
