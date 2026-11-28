"""A1 Q8 paired rerun of the frozen Q16 paid-agreement rule."""
import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import socket
from statistics import mean
import sys
import time
from unittest.mock import patch

Q, CAP = 8, 2048
DEV = (7, 13, 25, 33, 55, 71)
REST = tuple(i for i in range(90) if i not in DEV)
ARMS = ("stock", "tracked", "apply")
ID = ("index", "dataset", "task_id", "seed")
UUID = "GPU-861c9b5d-1e73-2550-f832-b73988bc64ff"
SAMPLES = "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed"
OLD = "63d9cd573323e9c1a0fa4cad6b97a2d5713ee6ed8173f7ce311537e6aaa9a4b8"
SCORER = "a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692"
PINS = {
    "paid_agreement": "f4ae0deaf51708c168cf1c3453043e535ce5c7318cd47d886288d8698c96f155",
    "llada_ibs_eos_stop": "70f58f50cf3ba3fa8515b7b81a92d92ea4b97fa964e3516686ccd5064627889b",
    "llada_ibs_eos_eval": "dff08c993abc1d5c4fdd7e0963b46116bb315e1c7c719ee373db0f7440a71737",
    "llada_ibs_eos_q16_agreement": "f66f253604d859c07f8a38817b821e630a2b8f859f5d1d9d45a7240406e6da1f",
    "llada_ibs_eos_q16_eval": "a3e2f93b8bb3e928217a8e00cbfdcf5917ed434f6d8f1bcabba8e107bf93762b",
    "llada_ibs_eos_raw_audit": "7d0388628a0b13dffcc355c6561d98cab7986750d5a6f4814e25c8601c73d5a1",
}
BASE_PINS = {
    "bdlm_serving/methods/optimus/fixed.py": "b95003b5f8867bd6ef7d5126c9814e08fb6d03bab57f67642618028d8a1960d4",
    "bdlm_serving/methods/optimus/upstream.py": "f21a4704697cace53b19dfbbbf94bbabc877241b9c979e15b0d40e9a2ea0d223",
    "bdlm_serving/runtime/batch.py": "f9a5363bffdd10026a0ab29fba5505da9c838cc13fc6449a469da91894b23a03",
    "bdlm_serving/runtime/serial.py": "6f92dcee3651e8248deb62f46c54b5f72f94b6dad02953f00712a35d13144fe8",
    "experiments/t2t/serving_slots/ibs_prefix_pair.py": "987012cf11deb9c27de832db77f0c918fc727f53da308f2e629f974b05704ab8",
    "experiments/t2t/serving_slots/shared_llada_worker.py": "4519b696c3155cd603c3b0ca5f5f0e953c5a47b964eb8211b81cdd53446f1471",
    "bdlm_serving/evaluation/scoring.py": SCORER,
}
CONTROL_PINS = {
    "sequence.py": "99c80047d517113579e04fd57443471a759a738dc30b206ba1342e0d4f9b3da2",
    "unmasking.py": "bbc3c3328e5497aa08aed6e04150ab0383f95dc06b4ff570a69adaed4a769746",
}
FROZEN = Path(__file__).resolve().parent / "frozen-q16"
SRC = Path("/gs/code/formal90-7a38ffc/src")
CONTROLS = Path("/gs/work/llada21-ibs-q8-tail-overlap-a1-v1/controls/Optimus")
CHECKPOINT = Path("/gs/models/LLaDA2.1-mini")


def check(ok, why):
    if not ok:
        raise ValueError(why)


def h(data):
    return sha256(data).hexdigest()


def blob(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def emit(kind, **value):
    print(blob(dict(kind=kind, **value)).decode(), flush=True)


def pinned_module(name, group, *, change_offer=False):
    path = FROZEN / (name + ".py")
    source = path.read_bytes()
    if name in PINS:
        check(h(source) == PINS[name], "Frozen Q16 source changed: " + name)
    canonical = group + "." + name
    spec = importlib.util.spec_from_file_location(canonical, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[canonical] = mod
    if change_offer:
        check(name == "llada_ibs_eos_q16_agreement", "Wrong adapter")
        for a, b in ((b"Q = 16\n", b"Q = 8\n"), (b"self.chunk == 16", b"self.chunk == 8"),
                     (b"self.caps == (16,)", b"self.caps == (8,)"),
                     (b"Expected strict IBS Q16", b"Expected strict IBS Q8")):
            check(source.count(a) == 1, "Offer substitution is not unique")
            source = source.replace(a, b)
    exec(compile(source, str(path), "exec"), mod.__dict__)
    return mod


def helpers():
    import bdlm_serving.methods.t2t
    pinned_module("paid_agreement", "bdlm_serving.methods.t2t")
    pinned_module("llada_ibs_eos_stop", "tools")
    pinned_module("llada_ibs_eos_eval", "tools")
    hook = pinned_module("llada_ibs_eos_q16_agreement", "tools", change_offer=True)
    audit = pinned_module("llada_ibs_eos_q16_eval", "tools")
    raw = pinned_module("llada_ibs_eos_raw_audit", "tools")
    check(audit.Q == hook.Q == Q and audit.SIX == DEV and hook.ARMS == ARMS, "Q8 overlay mismatch")
    return hook, audit, raw


def old_stock(sealed):
    check((sealed/"runner.exit").read_text().strip() == "0", "Unsealed A1 strict reference")
    data = (sealed/"overlap.ndjson").read_bytes()
    check(h(data) == OLD, "Original reference hash changed")
    receipts = [json.loads(x) for x in data.splitlines()]
    check(receipts[-1]["complete"] and len(receipts) == 19, "Reference incomplete")
    out = {r["index"]: r for r in receipts[:-1] if r["arm"] == "stock"}
    check(tuple(sorted(out)) == DEV and all(r["gpu_uuid"] == UUID and r["q"] == Q
                                                for r in out.values()), "Wrong A1 endpoint")
    return out


def prepare(sealed):
    old = old_stock(sealed)
    print(blob({str(i): dict(prompt_sha=h(blob(old[i]["prompt_ids"])),
                             identity={k: old[i][k] for k in ID}) for i in DEV}).decode())


def run(stage, reference, deadline):
    deps = "/gs/env/fixed-baseline-deps"
    sys.path[:] = [str(SRC), deps] + [p for p in sys.path
                                     if p not in (str(SRC), deps) and not p.startswith("/gs/history-repo")]
    hook, audit, raw = helpers()
    check(socket.gethostname().split(".")[0] == "n1" and os.environ["CUDA_VISIBLE_DEVICES"] == UUID,
          "Only n1 physical A1 may run")
    check(all(h((SRC/k).read_bytes()) == v for k, v in BASE_PINS.items()), "A1 runner source drift")
    check(all(h((CONTROLS/k).read_bytes()) == v for k, v in CONTROL_PINS.items()), "Optimus source drift")
    taskfile = Path("/gs/code/samples.json")
    check(h(taskfile.read_bytes()) == SAMPLES, "Frozen prompts drift")
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.methods.optimus import fixed, upstream
    from experiments.t2t.serving_slots import ibs_prefix_pair as pair, shared_llada_worker as worker
    import torch
    import transformers
    check(str(torch.__version__) == "2.6.0+cu124" and transformers.__version__ == "4.57.1"
          and torch.cuda.device_count() == 1, "Wrong native environment")
    for arm in ARMS:
        hook.install_run(fixed, arm, {}, set())
    model = worker.load(CHECKPOINT)
    eos, mask, tok = model["eos_id"], model["mask_id"], model["tok"]
    check((eos, mask) == (raw.EOS, raw.MASK), "Unexpected special IDs")
    excluded = set(tok.all_special_ids) | {eos, mask} | {
        i for i, t in tok.added_tokens_decoder.items() if t.special}
    excluded |= {t["id"] for t in json.loads((CHECKPOINT/"tokenizer.json").read_bytes())
                 ["added_tokens"] if t["special"]}
    tasks = load_tasks(taskfile)
    check([t["index"] for t in tasks] == list(range(90)), "Wrong frozen order")
    panel = DEV if stage == "six" else REST
    emit("registered", stage=stage, indices=panel, pid=os.getpid(), gpu=UUID, q=Q,
         policy=h(blob(PINS)), checkpoint=model["fingerprint"], excluded=sorted(excluded),
         code=h(Path(__file__).read_bytes()), samples=SAMPLES)
    with patch.object(pair, "Q", Q), patch.dict(pair.LABELS, {0: "optcap_ibs_8"}), \
            patch.object(upstream, "control_directory", lambda n: CONTROLS):
        for j, i in enumerate(panel):
            check(time.time() < deadline - 30, "Bounded stage timed out")
            t = tasks[i]
            text = tok.apply_chat_template([{"role": "user", "content": t["prompt"]}],
                                           tokenize=False, add_generation_prompt=True)
            prompt = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
            if stage == "six":
                want = reference[str(i)]
                check(want["prompt_sha"] == h(blob(prompt)) and
                      want["identity"] == {k: t[k] for k in ID}, "Same-A1 input/seed drift")
            rows, tapes = {}, {}
            order = ARMS[j%3:] + ARMS[:j%3]
            for arm in order:
                tape = tapes[arm] = dict(certificate=None, writes=[], ticks=[])
                torch.manual_seed(t["seed"])
                with patch.object(fixed.FixedOptimus, "run", hook.install_run(fixed, arm, tape, excluded)):
                    rows[arm] = pair.generate(model["model"], tok, t, prompt, CAP, 0)
                print("case", i, arm, rows[arm]["generation_forwards"], file=sys.stderr, flush=True)
            bills, fork = audit_pair(rows, tapes, audit, raw, excluded)
            emit("pair", index=i, order=order, arms=rows, tapes=tapes, bills=bills, fork=fork)
    emit("complete", stage=stage, count=len(panel), gpu=UUID)


def audit_pair(rows, tapes, audit, raw, excluded):
    check(rows["stock"] == rows["tracked"] and all(
        tapes["stock"][k] == tapes["tracked"][k] for k in ("certificate", "writes")),
        "Observer changed paid stock")
    bills = {}
    for arm in ARMS:
        tape = tapes[arm]
        cert = raw.replay(rows[arm], tape, q=Q)
        if cert is not None:
            cert["debt_after"] = []
        check(cert == tape["certificate"], "Independent canvas/debt replay failed")
        bills[arm] = audit.audit(rows[arm], tape, arm, eos=raw.EOS, mask=raw.MASK, excluded=excluded)
        for t in tape["ticks"]:
            t["applied"] = tuple(t["applied"])
    return bills, audit.first_fork(rows["tracked"], rows["apply"], tapes["tracked"], tapes["apply"])


def aggregate(cases):
    result = {}
    for arm in ARMS:
        a = [c["arms"][arm] for c in cases]
        result[arm] = {k: sum(x[k] for x in a) for k in ("N", "F", "Q", "correct", "extra", "paid_D", "paid_Q")}
        out = result[arm]
        out.update(TPF=mean(x["N"]/x["F"] for x in a), TE=out["N"]/out["Q"],
                   failures=sum(bool(x["failure"]) for x in a))
    return result


def report(stage, receipts, samples, sealed):
    _, audit, raw = helpers()
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen90 import load_tasks
    check(h(samples.read_bytes()) == SAMPLES and h(Path(scoring.__file__).read_bytes()) == SCORER,
          "Frozen scorer inputs changed")
    tasks = load_tasks(samples)
    lines = [json.loads(s) for s in receipts.read_bytes().splitlines()]
    reg, *pairs, end = lines
    panel = DEV if stage == "six" else REST
    check(reg["kind"] == "registered" and reg["gpu"] == UUID and reg["indices"] == list(panel)
          and reg["policy"] == h(blob(PINS)) and reg["q"] == Q and reg["samples"] == SAMPLES
          and end == dict(kind="complete", stage=stage, count=len(panel), gpu=UUID)
          and tuple(p["index"] for p in pairs) == panel, "Wrong or incomplete receipt")
    cases, native = [], {}
    stock_ref = old_stock(sealed) if stage == "six" else {}
    for p in pairs:
        i, rows, tapes = p["index"], p["arms"], p["tapes"]
        check(p["kind"] == "pair" and all(all(rows[a][k] == tasks[i][k] for k in ID) for a in ARMS)
              and all(rows[a]["prompt_ids"] == rows["stock"]["prompt_ids"] for a in ARMS), "Identity drift")
        bills, fork = audit_pair(rows, tapes, audit, raw, set(reg["excluded"]))
        check(bills == p["bills"] and fork == p["fork"], "Raw score or fork drift")
        for arm in ARMS:
            r = rows[arm]
            bills[arm]["correct"] = int(r["status"] == "success" and bool(
                scoring.score(r["output_text"], tasks[i])["correct"]))
        if i in stock_ref:
            r, s, cert = stock_ref[i], rows["stock"], tapes["stock"]["certificate"]
            n = cert["event"]+1 if cert else len(s["events"])
            native[i] = dict(equal_ids=s["output_ids"] == r["output_ids"],
                             same_prefix=all(s[k][:n] == r[k][:n] for k in
                                             ("events", "calls", "forward_inputs")),
                             N=r["output_tokens"], F=r["generation_forwards"], Q=r["generation_query_rows"])
        cases.append(dict(index=i, arms=bills, fork=fork))
    totals = aggregate(cases)
    s, a = totals["stock"], totals["apply"]
    gate = dict(quality=a["correct"] >= s["correct"], TPF=a["TPF"] > s["TPF"],
                F=a["F"] < s["F"], TE=a["TE"] >= s["TE"], audit=True,
                no_new_failures=not any(c["arms"]["apply"]["failure"] and not c["arms"]["stock"]["failure"]
                                        for c in cases))
    emit("summary", stage=stage, gate=gate, expand=stage == "six" and all(gate.values()), cases=cases,
         totals=totals, native_endpoint_separate=native, raw_sha256=h(receipts.read_bytes()),
         registration={k: v for k, v in reg.items() if k != "excluded"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest="mode", required=True)
    a = modes.add_parser("prepare")
    a.add_argument("sealed", type=Path)
    a = modes.add_parser("run")
    a.add_argument("stage", choices=("six", "remaining"))
    a.add_argument("reference", type=Path)
    a.add_argument("deadline", type=int)
    a = modes.add_parser("report")
    a.add_argument("stage", choices=("six", "remaining"))
    for field in ("receipts", "samples", "sealed"):
        a.add_argument(field, type=Path)
    a = p.parse_args()
    if a.mode == "prepare":
        prepare(a.sealed)
    elif a.mode == "run":
        run(a.stage, json.loads(a.reference.read_bytes()), a.deadline)
    else:
        report(a.stage, a.receipts, a.samples, a.sealed)
