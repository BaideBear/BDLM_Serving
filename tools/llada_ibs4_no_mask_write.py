"""Bounded paired readout of native T2T writes on measured no-MASK IBS4 calls."""
import argparse
from collections import Counter
from hashlib import sha256
import inspect
import json
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

ROOT = Path("/data1/project_data/gx-erqw-20261123")
DEPLOY = ROOT / "runtime/formal90-7a38ffc"
DATA = ROOT / "runtime/t2t-native-1125/prefix90"
OUT = ROOT / "results/llada-ibs4-no-mask-paid-write-h4-v1"
REG_SHA = "4f99d4a590bd5c9ef2311c456679d5983b59e00b2cd90488332da662edbdd940"
KEYS = ("index", "dataset", "task_id", "seed")


def check(ok, why):
    if not ok:
        raise ValueError(why)


def dump(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(x):
    return sha256(dump(x).encode()).hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def save(p, obj):
    with p.open("x") as f:
        f.write(dump(obj) + "\n")


def traced_run(fixed, reg, arm, tape):
    """Install one process-local callback before the pinned native canvas assignment."""
    check(arm in ("keep", "block"), "unknown arm")
    check(sha256(Path(fixed.__file__).read_bytes()).hexdigest() == reg["fixed_source_sha256"],
          "unregistered IBS source")
    src = dedent(inspect.getsource(fixed.FixedOptimus.run))
    anchor = "                x[0, positions[edited]] = proposed[edited]"
    check(src.count(anchor) == 1, "ambiguous hook site")

    def observe(edited, masked, positions, dirty, before, selected, proposed, confidence):
        pos, masks = positions.tolist(), masked.tolist()
        native = positions[edited].tolist()
        eligible = sorted(set(pos) & dirty) if not any(masks) else []
        targets = sorted(set(native) & set(eligible))
        tape.append(dict(rows=pos, masks=masks, dirty=sorted(dirty), old=before[0, selected].tolist(),
                         draft=proposed.tolist(), confidence=confidence.tolist(),
                         native=native, eligible=eligible, targets=targets))
        if arm == "block" and targets:
            return edited & ~positions.new_tensor([p in targets for p in pos], dtype=edited.dtype)
        return edited

    insert = ("                edited = observe(edited, masked, positions, dirty, before, selected,"
              " proposed, confidence)\n")
    ns = dict(vars(fixed), observe=observe)
    exec(compile(src.replace(anchor, insert + anchor), "<pinned-ibs4-write-observer>", "exec"), ns)
    return ns["run"]


def summarize(row, arm, task):
    from bdlm_serving.evaluation.scoring import score
    from experiments.t2t.serving_slots import ibs_prefix_pair as pair
    physical = pair.audit(row, 0)
    check(row["verified_layers"] == 20, "missing layer widths")
    events = [e for e in row["events"] if e["phase"] != "prefill"]
    feed = [v for v, e in zip(row["forward_inputs"], row["events"], strict=True)
            if e["phase"] != "prefill"]
    debt, count = set(), Counter()
    for e, v, a in zip(events, feed, row["observer"], strict=True):
        pos = a["rows"]
        targets = set(a["targets"])
        expected = set(a["native"]) - (targets if arm == "block" else set())
        check(e["absolute_positions"] == pos and e["dirty_before"] == a["dirty"] == sorted(debt)
              and e["selected_masked"] == a["masks"] and v == a["old"], "input/geometry mismatch")
        masks = {p for p, m in zip(pos, a["masks"], strict=True) if m}
        check(set(a["eligible"]) == (set() if masks else debt & set(pos)) and
              targets == set(a["native"]) & set(a["eligible"]), "eligibility mismatch")
        wanted = {p for p, o, d, c in zip(pos, a["old"], a["draft"], a["confidence"], strict=True)
                  if p not in masks and p >= len(row["prompt_ids"]) and o != d and c > .5}
        check(wanted == set(a["native"]) and set(e["edited_positions"]) == expected, "wrong edit")
        check(set(pos) >= debt and expected <= set(e["written_absolute"]), "unpaid or unrecorded write")
        debt = set(e["written_absolute"])
        count.update(no_mask_F=int(not masks), eligible_F=int(bool(a["eligible"])),
                     eligible_Q=len(a["eligible"]), target_F=int(bool(targets)), target_Q=len(targets),
                     actual_edits=len(expected), target_writes=len(targets) if arm == "keep" else 0)
    failed = row["status"] == "generation_failed"
    check(not debt or failed, "unpaid terminal debt")
    return dict(**{k: physical[k] for k in ("N", "F", "Q")}, **dict(count),
                prefill_Q=row["prefill_rows"], max_Q=row["max_q"], layers=row["verified_layers"],
                correct=not failed and bool(score(row["output_text"], task)["correct"]),
                failure=row["request_failure"], terminal_debt=len(debt), eos=row["eos"],
                ids_sha256=digest(row["output_ids"]), events_sha256=digest(row["events"]))


def paired(a, b):
    for k in (*KEYS, "prompt_ids", "generation_cap", "prefill_rows"):
        check(a[k] == b[k], f"between-arm identity: {k}")
    history = [r["events"] for r in (a, b)]
    obs = [r["observer"] for r in (a, b)]
    ticks = [next((i for i, e in enumerate(t) if e["targets"]), None) for t in obs]
    check(ticks[0] == ticks[1], "first native edit opportunity differs")
    stop = ticks[0]
    if stop is None:
        check(history[0] == history[1] and obs[0] == obs[1] and
              all(a[k] == b[k] for k in ("output_ids", "forward_inputs", "status", "request_failure")),
              "unexposed trajectories differ")
        return None
    j = int(a["events"][0]["phase"] == "prefill") + stop
    left, right = (t[j] for t in history)
    check(history[0][:j] == history[1][:j] and obs[0][:stop+1] == obs[1][:stop+1] and
          a["forward_inputs"][:j+1] == b["forward_inputs"][:j+1], "pre-write trajectories differ")
    for k in ("block", "step", "offer_tick", "q_tokens", "candidate_local", "selected_local",
              "absolute_positions", "visible_kv", "cached_prefix_before", "dirty_before"):
        check(left[k] == right[k], f"first decision differed on {k}")
    e = obs[0][stop]
    check(set(left["written_absolute"]) - set(e["targets"]) == set(right["written_absolute"]),
          "first actual write not isolated")
    return dict(tick=stop, block=left["block"], step=left["step"], rows=e["rows"], dirty=e["dirty"],
                visible_kv=left["visible_kv"], same_preceding_calls=j,
                kept_elsewhere=right["edited_positions"],
                blocked=[dict(row=p, old=o, native=d, confidence=c) for p, o, d, c in
                         zip(e["rows"], e["old"], e["draft"], e["confidence"], strict=True)
                         if p in e["targets"]])


def main():
    from experiments.t2t.serving_slots import ibs_prefix_pair as pair, shared_llada_worker as w
    from experiments.t2t.serving_slots import shared_fixed_shard as shard, llada_reference as ref
    from bdlm_serving.methods.optimus import fixed, upstream
    from llada_q4_first_fork import controls
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("check", "run", "report"))
    mode = p.parse_args().mode
    check(sha256((OUT / "registration.json").read_bytes()).hexdigest() == REG_SHA, "registration")
    reg = read(OUT / "registration.json")
    base = read(DEPLOY / "registration.json")
    check(shard.identity(base) == reg["base"], "release")
    shard.check_sources(base)
    for path, key in ((DATA/"samples.json", "samples_sha256"),
                      (DATA/"fixed-family-reference-case-ledger-v1.json", "reference_ledger_sha256")):
        check(sha256(path.read_bytes()).hexdigest() == reg[key], key)
    controls("Optimus")
    tasks = shard.load_tasks(DATA / "samples.json")
    cases = shard.shard_cases(read(shard.MANIFEST)["samples"], reg["board"], "formal90")
    cells = {}
    for t in sorted(cases, key=lambda t: t["index"]):
        if t["index"] not in (1, 67):
            cells.setdefault((t["index"] >= 30, t["dataset"]), t["index"])
    check(sorted(cells.values()) == reg["cases"], "unregistered cases")
    traced_run(fixed, reg, "block", [])
    paths = {(i, arm): OUT/f"{i:02d}-{arm}.json" for i in reg["cases"] for arm in ("keep", "block")}
    if mode == "check":
        print("validated frozen selection, code and write point")
        return
    if mode == "run":
        check(not any(path.exists() for path in paths.values()), "refusing replay over evidence")
        shard.check_board(reg["board"])
        h = w.load(Path("/data0/ysy/models/LLaDA2.1-mini"))
    ledger = read(DATA/"fixed-family-reference-case-ledger-v1.json")["arms"]["llada21:optcap_ibs_4"]["items"]
    with patch.object(pair, "Q", 4), patch.dict(pair.LABELS, {0: "optcap_ibs_4"}), \
         patch.object(upstream, "control_directory", controls):
        for ordinal, i in enumerate(reg["cases"]):
            task = tasks[i]
            hist = read(DATA/f"llada21-optimus-hardcap-low/{i:02d}-optcap_ibs_4.json")
            check(all(task[k] == ledger[i][k] == hist[k] for k in KEYS), "sample identity")
            arms, summaries = {}, {}
            order = ("block", "keep") if ordinal % 2 else ("keep", "block")
            for arm in order:
                if mode == "run":
                    tape = []
                    prompt, _ = w.canonical_prompt(h["tok"], task, hist["prompt_ids"])
                    with patch.object(fixed.FixedOptimus, "run", traced_run(fixed, reg, arm, tape)):
                        r = pair.generate(h["model"], h["tok"], task, prompt, 2048, 0)
                    r.update(observer=tape, study_arm=arm)
                    save(paths[i, arm], r)
                else:
                    r = read(paths[i, arm])
                check(r["study_arm"] == arm and all(task[k] == r[k] for k in KEYS), "arm identity")
                arms[arm] = r
                s = summaries[arm] = summarize(r, arm, task)
                if arm == "keep":
                    ref.verify_row(hist, r)
                    check(hist["events"] == r["events"] and hist["extra"] == r["extra"] and
                          all(s[k] == ledger[i][k] for k in ("N", "F", "Q", "correct")),
                          "keep failed exact historical replay")
            print(dump(dict(**{k: task[k] for k in KEYS}, registration=REG_SHA, arm_order=order,
                            input_sha256=digest(hist["prompt_ids"]), **summaries,
                            first=paired(arms["keep"], arms["block"]))), flush=True)


if __name__ == "__main__":
    main()
