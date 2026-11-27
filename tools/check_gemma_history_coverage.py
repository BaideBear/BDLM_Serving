"""Extend GPU diagnostic coverage on a fixed panel, never change a scored cohort."""
import json
from pathlib import Path

from tools import run_history_gemma as impl

# Same fixed panel, cap, and seeds for every sparse tier; retain all examples.
ORIGINAL = (0, 10, 20)
PANEL = ORIGINAL + (1, 11, 21, 2, 12, 22)
TIERS = (32, 64, 128)


def compatible(gate):
    if not (gate["protocol"] == impl.DISPATCH and not gate["ready_for_cohort"]
            and not gate["split_parity_pass"] and gate["source_sha256"] == impl.provenance(True)
            and [r["index"] for r in gate["endpoints"]] == list(ORIGINAL)
            and all(r["pass_"] for r in gate["endpoints"])
            and [r["index"] for r in gate["operator"]] == list(ORIGINAL)
            and all(r["cases"] == 9 for r in gate["operator"])):
        raise ValueError("Need verified full endpoints, packed operator, and an incomplete sparse gate")


def run(checkpoint, samples, prior, out, budgets, prepare):
    import torch
    import transformers
    import triton
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.runtime.diffusiongemma import STUDY, check_snapshot

    eligible = impl.read(prior)
    compatible(eligible)
    if (impl.sha(samples) != eligible["samples_sha256"] or eligible["software"] !=
            dict(torch=str(torch.__version__), transformers=transformers.__version__, triton=triton.__version__)):
        raise ValueError("Gate and validation use different input or libraries")
    plan = impl.read(STUDY / "config.json")
    tasks = load_tasks(samples)
    out.mkdir(parents=True, exist_ok=False)
    print("Checking pinned checkpoint bytes", flush=True)
    check_snapshot(checkpoint)
    proc = transformers.AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = transformers.DiffusionGemmaForBlockDiffusion.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16, attn_implementation=plan["attention"],
        experts_implementation=plan["experts"], device_map={"": "cpu"}).eval()
    prepare(model)
    model.to("cuda:0")
    if torch.cuda.get_device_name(0) != eligible["gpu"] or model.config.canvas_length != impl.WIDTH:
        raise ValueError("Gate and validation use different hardware or canvas")
    with torch.inference_mode():
        for c in budgets:
            for i in PANEL:
                archived = prior.parent / f"{i:02d}-history{c}.json"
                if i in ORIGINAL and archived.is_file():
                    row = impl.read(archived)
                else:
                    row = impl.request(model, proc, tasks[i], c, 512, audit=True, dispatch=True)
                if (row["nominal_history_q"] != c or row["cap"] != 512 or
                        any(row[k] != tasks[i][k] for k in ("index", "seed", "task_id", "dataset"))):
                    raise ValueError("Unexpected validation identity")
                audit = impl.audit_pilot(row)
                impl.write(out / f"{i:02d}-C{c}.json", audit)
                print(c, i, audit["edits_without_read"], flush=True)
    impl.write(out / "complete.json", dict(prior=impl.sha(prior), panel=list(PANEL), budgets=budgets,
                                             checker_sha256=impl.sha(Path(__file__))))


def merge(prior, shards, out):
    gate = impl.read(prior)
    compatible(gate)
    trials = {}
    for shard in shards:
        receipt = impl.read(shard / "complete.json")
        if receipt["prior"] != impl.sha(prior) or receipt["panel"] != list(PANEL) or (
                receipt["checker_sha256"] != impl.sha(Path(__file__))):
            raise ValueError("Mismatched extension evidence")
        for c in receipt["budgets"]:
            for i in PANEL:
                r = impl.read(shard / f"{i:02d}-C{c}.json")
                if (c, i) in trials or (r["budget"], r["index"]) != (c, i) or not r["sampler_debt_audit"]:
                    raise ValueError("Missing or duplicated observation")
                trials[c, i] = r
    if set(trials) != {(c, i) for c in TIERS for i in PANEL}:
        raise ValueError("Incomplete fixed coverage panel")
    passed = all(any(trials[c, i]["versions"] > 1 for i in PANEL)
                 and any(trials[c, i]["edits_without_read"] for i in PANEL) for c in TIERS)
    out.mkdir(parents=True, exist_ok=False)
    impl.write(out / "dispatch-gate.json", dict(gate, pass_=passed, ready_for_cohort=passed,
        original_gate_ready=False, original_gate_sha256=impl.sha(prior),
        qualification="fixed_extended_coverage_v1", validation_panel=list(PANEL),
        checker_sha256=impl.sha(Path(__file__)),
        pilots=[trials[c, i] for c in TIERS for i in ORIGINAL],
        extension_pilots=[trials[c, i] for c in TIERS for i in PANEL if i not in ORIGINAL]))
    if not passed:
        raise RuntimeError("Extended panel still lacks encoder invalidation or warm off-Q edits")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("shards", type=Path, nargs="+")
    a = p.parse_args()
    merge(a.prior, a.shards, a.out)
