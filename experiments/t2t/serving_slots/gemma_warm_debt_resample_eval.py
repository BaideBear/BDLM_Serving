"""Gemma warm resolved-debt resample paired evaluation against verified-EOS."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import time
from unittest.mock import patch

from .gemma_fixed_dual import B, FULL
from .gemma_joint_verify_eval import ORDER, check
from .gemma_verified_eos_eval import run
from . import gemma_warm_debt_resample as factor
from .layout_sweep import write


SCHEMA = "gemma-warm-debt-resample-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def available(uuid):
    import torch
    check(torch.cuda.device_count() == 1, "Expected exactly one visible GPU")
    check(os.environ.get("CUDA_VISIBLE_DEVICES") == uuid, "Wrong CUDA_VISIBLE_DEVICES")
    rows = subprocess.check_output(
        ["nvidia-smi", "-i", uuid, "--query-compute-apps=pid,gpu_uuid",
         "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    check(all(row.split(", ")[1] == uuid for row in rows), "Foreign GPU process")


def replay(control, ref, task):
    for k in ("output_ids", "tape", "input_sha256", "hit", "encoder_digest",
              "decoder_F", "decoder_Q", "encoder_F", "encoder_Q", "F",
              "route_F", "route_Q", "N", "correct"):
        check(control[k] == ref[k], f"control replay failed: {k}")


def first_difference(control_req, treatment_req, control, treatment):
    for tick, (a, b) in enumerate(zip(control_req.events, treatment_req.events)):
        if a == b:
            continue
        check(a["route"] == "drain" and b["route"] == "warm_debt_resample",
              "first change is not the warm resolved-debt retry")
        pre = control_req.pre_states[tick]
        check(pre["ordinal"] == a["ordinal"] and pre["phase"] == "sample" and not pre["tail"],
              "wrong pre-state")
        resolved = set(b["positions"]) - {control_req.origin + B * pre["ordinal"] + i
                                      for i in FULL - pre["accepted"]}
        check(a["positions"] == b["positions"] and a["offer"] == b["offer"] and a["q"] == b["q"],
              "changed width or positions")
        check(a["dirty_before_absolute"] == b["dirty_before_absolute"], "changed debt")
        check(set(b["sampled_absolute"]) == set(b["positions"]) & resolved,
              "not exactly the resolved dirty rows")
        check(set(b["written_absolute"]) <= set(b["sampled_absolute"]), "unmeasured write")
        check(a["written_absolute"] == [], "control drain wrote")
        check(set(b["dirty_after_absolute"]) ==
              (set(a["dirty_before_absolute"]) - set(b["positions"])) | set(b["written_absolute"]),
              "dirty ledger")
        pa, pb = a["parts"][0], b["parts"][0]
        check(pa["phase"] == "drain" and pb["phase"] == "sample", "wrong phase")
        check(pa["age_after"] == pa["age_before"] and pb["age_after"] == pa["age_after"] + 1,
              "sampler clock")
        return dict(tick=tick, added=sorted(set(b["sampled_absolute"])),
                    admitted=list(pb["admitted"]), edited=list(pb["edited"]),
                    revoked=list(pb["revoked"]), input_changed=list(pb["input_changed"]))
    check(len(control_req.events) == len(treatment_req.events)
          and control["output_ids"] == treatment["output_ids"]
          and control["F"] == treatment["F"], "no exposure but results differ")
    return None


def cohort(rows, indices):
    present = [i for i in indices if i in rows]
    missing = [i for i in indices if i not in rows]
    part = [rows[i] for i in present]
    out = dict(expected=len(indices), present=len(present), missing=missing,
               complete=not missing, exposed=sum(bool(r["first"]) for r in part))
    for arm in ("control", "treatment"):
        if part:
            out[arm] = dict(
                correct=sum(r[arm]["correct"] for r in part),
                N=sum(r[arm]["N"] for r in part),
                F=sum(r[arm]["F"] for r in part),
                decoder_Q=sum(r[arm]["decoder_Q"] for r in part),
                encoder_Q=sum(r[arm]["encoder_Q"] for r in part),
                macro_TPF=mean(r[arm]["N"] / r[arm]["F"] for r in part),
                N_per_history_Q=mean(r[arm]["N"] / r[arm]["decoder_Q"] for r in part),
                pooled_N_per_Q=sum(r[arm]["N"] for r in part) / sum(r[arm]["decoder_Q"] for r in part),
            )
    for k in ("warm_debt_F", "warm_F", "new_admits", "new_edits", "new_revoked",
              "new_input_changed", "treatment_drain_F"):
        out[k] = sum(r[k] for r in part)
    return out


def summarize(rows, q, gpu, sources, samples_sha, scorer_sha):
    cohorts = {name: cohort(rows, indices) for name, indices in (
        ("pilot3", ORDER[:3]), ("original30", ORDER[:30]),
        ("new60", ORDER[30:]), ("all90", ORDER))}
    return dict(schema=SCHEMA, q=q, gpu=gpu, sources=sources,
                samples_sha256=samples_sha, scorer_sha256=scorer_sha,
                order=ORDER, cohorts=cohorts)


def write_csv(path, summary):
    columns = ("cohort", "arm", "requests", "correct", "N", "F", "decoder_Q",
               "encoder_Q", "macro_TPF", "N_per_history_Q", "pooled_N_per_Q",
               "warm_debt_F", "warm_F", "new_admits", "new_edits", "new_revoked",
               "new_input_changed")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, c in summary["cohorts"].items():
            for arm in ("control", "treatment"):
                if arm not in c:
                    continue
                writer.writerow(dict(cohort=name, arm=arm, requests=c["present"],
                                     **c[arm],
                                     **{k: c[k] for k in ("warm_debt_F", "warm_F", "new_admits",
                                                         "new_edits", "new_revoked",
                                                         "new_input_changed")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("out", "verified", "samples", "checkpoint"):
        ap.add_argument("--" + k, type=Path, required=True)
    ap.add_argument("--deadline", type=int, required=True)
    ap.add_argument("--q", type=int, choices=(32, 64), required=True)
    ap.add_argument("--limit", type=int, choices=(3, 90), default=3)
    ap.add_argument("--auto-full90", action="store_true")
    a = ap.parse_args()
    available(os.environ["CUDA_VISIBLE_DEVICES"])
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from .mask_pair_gemma import SAMPLES_SHA
    check(sha(a.samples) == SAMPLES_SHA and transformers.__version__ == "5.12.0"
          and a.checkpoint.name == "diffusiongemma-cd40195", "Wrong data or runtime")
    tasks = load_tasks(a.samples)
    a.out.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(a.checkpoint, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        a.checkpoint, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", experts_implementation="grouped_mm",
        device_map={"": "cuda:0"}).eval()
    special = set(proc.tokenizer.all_special_ids) | set(model.generation_config.eos_token_id)
    scorer = sha(Path(scoring.__file__))
    sources = {n: sha(Path(__file__).with_name(n + ".py")) for n in (
        "gemma_warm_debt_resample", "gemma_warm_debt_resample_eval",
        "gemma_verified_eos", "gemma_verified_eos_eval", "gemma_fixed_dual",
        "gemma_strict_sampler", "gemma_strict_request", "gemma_unified_audit",
        "gemma_strict_rows")}
    rows, exposures, limit = {}, 0, a.limit
    with torch.inference_mode():
        for j, i in enumerate(ORDER):
            if j >= limit:
                break
            check(time.time() < a.deadline, "Time limit")
            dst = a.out / f"C{a.q}-pair-{i:02}.json"
            if dst.exists():
                r = json.loads(dst.read_bytes())
                src = a.verified / f"C{a.q}-pair-{i:02}.json"
                check(r["source"] == sources and r["reference_sha256"] == sha(src)
                      and r["index"] == i and r["q"] == a.q, "Cannot resume mixed run")
                rows[i] = r
                exposures += bool(r["first"])
                continue
            task = tasks[i]
            src = a.verified / f"C{a.q}-pair-{i:02}.json"
            rec = json.loads(src.read_bytes())
            check(rec["q"] == a.q and rec["schema"] == "verified-eos-pair-v1"
                  and rec["samples_sha256"] == SAMPLES_SHA and rec["index"] == i,
                  "Wrong verified-EOS reference")
            check(all(rec[k] == task[k] for k in ("index", "seed", "dataset", "task_id")),
                  "Wrong verified-EOS identity")
            ref = rec["on"]
            data = proc.apply_chat_template(
                [{"role": "user", "content": task["prompt"]}], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
            order = ("control", "treatment") if j % 2 == 0 else ("treatment", "control")
            arms = {}
            for arm in order:
                cls = (factor.RecordingControlRequest if arm == "control"
                       else factor.WarmDebtResampleRequest)
                factor.REQUESTS.clear()
                with patch("experiments.t2t.serving_slots.gemma_verified_eos.VerifiedEOSRequest", cls):
                    row = run(model, data, task, a.q, True)
                req = factor.REQUESTS[-1]
                ids = row["output_ids"]
                text = proc.tokenizer.decode(ids, skip_special_tokens=True)
                row.update(N=sum(x not in special for x in ids),
                           correct=bool(scoring.score(text, task)["correct"]))
                arms[arm] = (row, req)
            control, control_req = arms["control"]
            treatment, treatment_req = arms["treatment"]
            replay(control, ref, task)
            first = first_difference(control_req, treatment_req, control, treatment)
            retry_events = [e for e in treatment_req.events if e["route"] == "warm_debt_resample"]
            check(len({e["ordinal"] for e in retry_events}) == len(retry_events),
                  "more than one warm-debt retry per physical head")
            r = {k: task[k] for k in ("index", "seed", "dataset", "task_id")}
            r.update(schema=SCHEMA, q=a.q, gpu=os.environ["CUDA_VISIBLE_DEVICES"],
                     source=sources, samples_sha256=SAMPLES_SHA, scorer_sha256=scorer,
                     reference_sha256=sha(src), control=control, treatment=treatment,
                     first=first, order=order,
                     warm_debt_F=len(retry_events),
                     warm_F=sum(e["route"] == "drain" for e in control_req.events),
                     new_admits=sum(len(p["admitted"]) for e in retry_events for p in e["parts"]),
                     new_edits=sum(len(p["edited"]) for e in retry_events for p in e["parts"]),
                     new_revoked=sum(len(p["revoked"]) for e in retry_events for p in e["parts"]),
                     new_input_changed=sum(len(p["input_changed"]) for e in retry_events for p in e["parts"]),
                     treatment_drain_F=sum(e["route"] == "drain" for e in treatment_req.events))
            write(dst, r)
            rows[i] = r
            exposures += bool(first)
            print("PAIR", a.q, i, bool(first), r["warm_debt_F"], r["warm_F"],
                  r["new_admits"], flush=True)
            if j + 1 == 3 and a.limit == 3:
                if exposures == 0:
                    print("NO WARM-DEBT EXPOSURE", a.q, flush=True)
                    break
                if a.auto_full90 and time.time() < a.deadline:
                    limit = 90
    summary = summarize(rows, a.q, os.environ["CUDA_VISIBLE_DEVICES"], sources,
                        SAMPLES_SHA, scorer)
    write(a.out / "summary.json", summary)
    write_csv(a.out / "summary.csv", summary)


if __name__ == "__main__":
    main()
