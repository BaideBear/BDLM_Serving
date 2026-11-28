"""Three-arm same-H0 strict IBS Q4 stopping control."""
import argparse
from collections import Counter
from contextlib import nullcontext
from hashlib import sha256
import json
import os
from pathlib import Path
from time import time
from unittest.mock import patch

from tools.llada_ibs_eos_stop import B, Q, install_run, require

BASE = Path("/data1/project_data/gx-erqw-20261123")
DATA = BASE / "runtime/t2t-native-1125/prefix90"
CONTROLS = BASE / "sources/optimus-controls"
CKPT = Path("/data0/ysy/models/LLaDA2.1-mini")
UUID = "GPU-ceadb3cd-e63f-1446-a2d5-a291144f8efb"
SIX = (0, 10, 20, 30, 40, 50)
ORDER = SIX + tuple(i for i in range(90) if i not in SIX)
ARMS = ("stock", "instrumented", "certificate")
KEYS = ("index", "dataset", "task_id", "seed")
CAP = 2048
LABEL = "optcap_ibs_4"


def read(path):
    return json.loads(path.read_bytes())


def compact(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(x):
    return sha256(compact(x).encode()).hexdigest()


def file_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix(".writing")
    tmp.write_text(compact(value)+"\n")
    tmp.replace(path)


def qualify(stock, observed, ended, tape, end_tape, eos, mask):
    """Reconcile the runtime certificate against an independently measured stock prefix."""
    require(stock == observed, "Stock and observation-only run differ")
    for k in (*KEYS, "prompt_ids", "generation_cap", "prefill_rows", "verified_layers"):
        require(stock[k] == ended[k], f"Pair identity drift: {k}")
    first = tape["certificate"]
    other = end_tape["certificate"]
    if first is None:
        require(ended == stock and tape == end_tape, "No-certificate arm changed execution")
        return None
    require(other == dict(first, debt_after=[]), "Certificate or terminal debt differs")
    n = first["event"] + 1
    for k in ("events", "calls", "forward_inputs"):
        require(ended[k][:n] == stock[k][:n], f"Ordinary trace differs before stop: {k}")
    writes = end_tape["writes"]
    require(writes == tape["writes"][:len(writes)] and len(writes) == n -
            sum(e["phase"] == "prefill" for e in stock["events"][:n]), "Native writes differ")
    e = stock["events"][n-1]
    pp = len(stock["prompt_ids"])
    eos_pos, due = first["eos"], first["debt_before"]
    start = e["block"] * B
    old_debt = set(e["dirty_before"])
    require(first["block"] == e["block"] and first["step"] == e["step"] and
            first["ready"] == e["cached_prefix_after"] and old_debt <= set(e["absolute_positions"]) and
            due == e["written_absolute"] and all(v >= start+first["ready"] for v in due) and
            max(pp,start) <= eos_pos < min(start+first["ready"], pp+CAP),
            "Wrong ordinary write/debt/frozen boundary")
    ids = ended["output_ids"]
    require(ended["status"] == "success" and ended["eos"] and ended["request_failure"] is None
            and ids and eos not in ids[:-1] and ids[-1] == eos and mask not in ids
            and len(ids) == eos_pos - pp + 1, "Not a complete certified prefix")
    if stock["status"] == "success":
        require(ids == stock["output_ids"], "Certified IDs differ from successful stock")
    remaining = ended["events"][n:]
    require(len(remaining) == bool(due), "Wrong number of real debt calls")
    if due:
        pay, = remaining
        require(pay["phase"] == "certificate_debt" and pay["q_tokens"] == len(due) and
                pay["absolute_positions"] == pay["dirty_before"] == due and
                pay["physical_cap"] == Q and pay["physical_forwards"] == 1 and
                not (pay["written_absolute"] or pay["sampled_absolute"] or pay["edited_positions"]),
                "Wrong physical debt payment or sampled during terminal payment")
        c = ended["calls"][-1]
        require(c["rows"] == c["physical_rows"] == len(due) and c["batch"] == 1
                and c["kv_rows"] == pay["visible_kv"] == due[-1]+1
                and list(zip(due, ended["forward_inputs"][-1])) == [tuple(x) for x in writes[-1]],
                "Payment did not project exactly the owed token inputs")
        if n < len(stock["events"]):
            stock_next = dict(zip(stock["events"][n]["absolute_positions"], stock["forward_inputs"][n], strict=True))
            require(all(stock_next.get(p) == t for p,t in writes[-1]), "Stock did not pay same write")
    after = stock["events"][n:]
    require(all(x["phase"] == "optimus" and x["block"] == first["block"] for x in after),
            "Stock crossed the certified block")
    return dict(**other, paid_D=int(bool(due)), paid_Q=len(due), ordinary_after_D=len(after),
                ordinary_after_Q=sum(x["q_tokens"] for x in after))


def metrics(row, task, judge):
    gen = [e for e in row["events"] if e["phase"] != "prefill"]
    require(row["verified_layers"] == 20 and row["max_q"] <= Q and
            len(gen) == row["generation_forwards"] and
            sum(e["q_tokens"] for e in gen) == row["generation_query_rows"] ==
            row["generation_physical_padded_rows"], "Missing charged execution")
    for e, c in zip(row["events"], row["calls"], strict=True):
        require(c["batch"] == e["physical_forwards"] == 1 and
                c["rows"] == c["physical_rows"] == e["q_tokens"] == len(e["absolute_positions"]),
                "Physical layer/accounting width mismatch")
    failed = row["status"] == "generation_failed"
    require(failed == bool(row["request_failure"]) and (not failed or not row["output_ids"]),
            "Failure ledger lost")
    return dict(N=row["output_tokens"], F=row["generation_forwards"], Q=row["generation_query_rows"],
                failure=row["request_failure"], eos=row["eos"],
                correct=not failed and bool(judge(row["output_text"], task)["correct"]),
                ids_sha=digest(row["output_ids"]))


def summary(rows, reg):
    cohorts = {}
    for label in ("all", "stock_success", "stock_failure_certified", "stock_failure_retained"):
        group = rows if label == "all" else [x for x in rows if x["cohort"] == label]
        bill = {arm: {k:sum(x["arms"][arm][k] for x in group) for k in ("N", "F", "Q", "correct", "eos")}
                for arm in ARMS}
        for arm in ARMS:
            bill[arm]["failures"] = sum(bool(x["arms"][arm]["failure"]) for x in group)
        certs = [x["certificate"] for x in group if x["certificate"] is not None]
        bill_delta = {k:bill["stock"][k]-bill["certificate"][k] for k in ("F","Q")}
        require(bill_delta["F"] == sum(c["ordinary_after_D"]-c["paid_D"] for c in certs) and
                bill_delta["Q"] == sum(c["ordinary_after_Q"]-c["paid_Q"] for c in certs),
                "Net bill cannot be explained by measured suffix vs payment")
        cohorts[label] = dict(n=len(group), arms=bill, saved=bill_delta, certificates=len(certs),
            with_later_D=sum(c["ordinary_after_D"]>0 for c in certs),
            ordinary_after_D=sum(c["ordinary_after_D"] for c in certs),
            ordinary_after_Q=sum(c["ordinary_after_Q"] for c in certs),
            paid_D=sum(c["paid_D"] for c in certs), paid_Q=sum(c["paid_Q"] for c in certs))
    return dict(reg=reg, indices=[x["index"] for x in rows], cohorts=cohorts)


def read_timeout_prefix(prior, panel_sha):
    """Continue only a time-limited run with a complete, hash-verified prefix."""
    from tools import llada_ibs_eos_stop as endpoint
    require(prior.parent == BASE/"results" and prior.name.startswith("llada-ibs-certified-eos-h0-v")
            and (prior/"returned.identity").is_file() and not (prior/"complete.json").exists(),
            "Not a restored incomplete H0 run")
    source = BASE/"runtime"/prior.name
    log = (source/"supervisor.log").read_text().splitlines()
    legacy_timeout = any("guard failure rc=142 " in x and 'wait "$worker"' in x for x in log)
    marked_timeout = (prior/"time-limit").is_file() and any(
        "guard failure rc=143 " in x and 'wait "$worker"' in x for x in log)
    if marked_timeout:
        expected = (prior/"worker.identity").read_text().split()
        actual = (prior/"time-limit").read_text().split()
        require(len(actual) == 4 and actual[:3] == expected, "Timeout marker identity changed")
    require(legacy_timeout or marked_timeout, "Only the worker's time limit permits continuation")
    require(file_sha(source/"tools/llada_ibs_eos_stop.py") == file_sha(Path(endpoint.__file__)),
            "Endpoint code changed across resume")
    old = read(prior/"registration.json")
    old_reg = digest(old)
    require(old["panel_sha"] == panel_sha and old["order"] == list(ORDER)
            and old["source"] == file_sha(source/"source.sha256") and old["uuid"] == UUID,
            "Original registration changed")
    rows = [json.loads(line) for line in (prior/"receipts.ndjson").read_text().splitlines()]
    require(len(SIX) <= len(rows) < len(ORDER) and
            [r["index"] for r in rows] == list(ORDER[:len(rows)]), "Invalid completed prefix")
    require(read(prior/"qualified-six.json") == summary(rows[:len(SIX)], old_reg),
            "Missing or changed six-case gate")
    for r in rows:
        require(r["reg"] == old_reg and file_sha(prior/"pairs"/f"{r['index']:02}.json") == r["raw_sha"],
                "Previous raw receipt not intact")
    return rows, dict(path=str(prior), registration=old_reg, sealed_cases=len(rows),
                      receipts_sha=file_sha(prior/"receipts.ndjson"))


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("mode", choices=("check", "run"))
    arg.add_argument("--panel", type=Path, required=True)
    arg.add_argument("--out", type=Path)
    arg.add_argument("--resume", type=Path)
    a = arg.parse_args()
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.methods.optimus import fixed, upstream
    from experiments.t2t.serving_slots import ibs_prefix_pair as pair, llada_reference as ref
    from experiments.t2t.serving_slots import shared_llada_worker as worker

    panel = read(a.panel)
    require(panel["six"] == list(SIX) and panel["arms"] == list(ARMS) and panel["q"] == Q
            and panel["cap"] == CAP and panel["uuid"] == UUID, "Panel changed")
    require(file_sha(DATA/"samples.json") == panel["samples_sha256"] and
            file_sha(DATA/"fixed-family-reference-case-ledger-v1.json") == panel["ledger_sha256"],
            "Changed frozen inputs")
    def controls(name):
        require(name == "Optimus" and all(file_sha(CONTROLS/k) == v for k,v in panel["opt_controls"].items()),
                "Changed frozen Optimus controls")
        return CONTROLS
    controls("Optimus")
    install_run(fixed, "certificate", {})
    earlier, predecessor = read_timeout_prefix(a.resume, file_sha(a.panel)) if a.resume else ([], None)
    if a.mode == "check":
        print("Pinned hook source, upstream and registered inputs verified; no inference")
        return
    require(a.out is not None and os.environ["CUDA_VISIBLE_DEVICES"] == UUID, "Expected H0-only run")
    remaining = int(os.environ["BDLM_EOS_END"]) - int(time()) - 45
    require(remaining >= 120, "Insufficient time in exclusive H0 window")
    import torch
    import transformers
    require(torch.cuda.device_count() == 1 and
            str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-") == UUID[4:],
            "Wrong physical board")
    h = worker.load(CKPT)
    model,tok = h["model"],h["tok"]
    require(model.device.type == "cuda", "No CPU inference")
    tasks = load_tasks(DATA/"samples.json")
    ledger = read(DATA/"fixed-family-reference-case-ledger-v1.json")["arms"]["llada21:"+LABEL]["items"]
    out = a.out
    (out/"pairs").mkdir(mode=0o700)
    prov = dict(panel=panel, panel_sha=file_sha(a.panel), order=ORDER, source=os.environ["BDLM_EOS_SOURCE"],
                pid=os.getpid(), uid=os.getuid(), start=time(), uuid=UUID, boot=Path(
                    "/proc/sys/kernel/random/boot_id").read_text().strip(), ticks=int(
                    Path("/proc/self/stat").read_text().split(") ")[1].split()[19]),
                torch=str(torch.__version__), transformers=transformers.__version__,
                checkpoint_fingerprint=h["fingerprint"], weights=h["weights"], scorer=h["scorer"])
    reg = digest(prov)
    if predecessor:
        prov.update(resume=predecessor)
        reg = digest(prov)
        save(out/"qualified-six.json", read(a.resume/"qualified-six.json"))
    save(out/"registration.json", prov)
    rows = list(earlier)
    with patch.object(pair, "Q", Q), patch.dict(pair.LABELS, {0:LABEL}), patch.object(upstream, "control_directory", controls):
        for ordinal, i in enumerate(ORDER[len(earlier):], len(earlier)):
            task, old = tasks[i], read(DATA/f"llada21-optimus-hardcap-low/{i:02d}-{LABEL}.json")
            require(all(task[k] == old[k] == ledger[i][k] for k in KEYS), "Case identity")
            prompt, _ = worker.canonical_prompt(tok, task, old["prompt_ids"])
            arms, tapes = {}, {}
            order = ARMS[ordinal%3:] + ARMS[:ordinal%3]
            for arm in order:
                torch.manual_seed(task["seed"])
                tape = tapes[arm] = dict(writes=[], certificate=None)
                hook = (nullcontext() if arm == "stock" else
                        patch.object(fixed.FixedOptimus, "run", install_run(fixed, arm, tape)))
                with hook:
                    arms[arm] = pair.generate(model, tok, task, prompt, CAP, 0)
            path = out/"pairs"/f"{i:02}.json"
            save(path, dict(arms=arms,tapes=tapes,reg=reg,order=order))
            stock = arms["stock"]
            ref.verify_row(old,stock)
            require(old["events"] == stock["events"] and old["extra"] == stock["extra"], "Historical tick drift")
            pair.audit(stock, 0)
            m = {key: metrics(val,task,scoring.score) for key,val in arms.items()}
            require(all(m["stock"][k] == ledger[i][k] for k in ("N","F","Q","correct")), "Historical bill drift")
            cert = qualify(*(arms[k] for k in ARMS), tapes["instrumented"], tapes["certificate"],
                           h["eos_id"], h["mask_id"])
            cohort = ("stock_success" if not m["stock"]["failure"] else
                      "stock_failure_certified" if cert is not None else "stock_failure_retained")
            row = dict(**{k:task[k] for k in KEYS}, reg=reg, cohort=cohort, arms=m, certificate=cert,
                       input_sha=digest(prompt), raw_sha=file_sha(path), arm_order=order)
            print(compact(row), flush=True)
            rows.append(row)
            if len(rows)==len(SIX):
                gate = summary(rows,reg)
                require(gate["cohorts"]["all"]["certificates"] > 0, "Six-case gate did not exercise stop")
                save(out/"qualified-six.json",gate)
            if len(rows)%10==0:
                save(out/"progress.json",summary(rows,reg))
    save(out/"complete.json",summary(rows,reg))


if __name__ == "__main__":
    main()
