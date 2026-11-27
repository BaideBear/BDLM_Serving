"""Summarize complete dispatch-v1 cohorts; never grade or count pilot rows."""
import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

from bdlm_serving.evaluation.frozen import sha
from bdlm_serving.evaluation.frozen90 import load_tasks


def read(p):
    return json.loads(p.read_bytes())


def summarize(samples, gate_path, budget, shards):
    tasks = load_tasks(samples)
    gate = read(gate_path)
    gate_hash = sha(gate_path)
    dataset_hash = sha(samples)
    if not (gate["protocol"] == "dispatch-v1" and gate["ready_for_cohort"] is True
            and gate["split_parity_pass"] is False and gate["samples_sha256"] == dataset_hash):
        raise ValueError("Unqualified implementation or mismatched data")
    backend = "triton_block_history" if budget < 256 else "native_sdpa_saturating_dispatch"
    indexed = {}
    for shard in shards:
        receipt = read(shard / "complete.json")
        if (receipt["source_sha256"] != gate["source_sha256"] or receipt["protocol"] != gate["protocol"]
                or receipt["samples_sha256"] != dataset_hash or receipt["gate_sha256"] != gate_hash
                or receipt["software"] != gate["software"] or receipt["gpu"] != gate["gpu"]
                or receipt["revision"] != gate["revision"] or receipt["budgets"] != [budget]
                or receipt["cap"] != 2048 or receipt["mode"] != "cohort"):
            raise ValueError(f"Incompatible completion receipt: {shard}")
        for i in receipt["indices"]:
            r = read(shard / f"{i:02d}-history{budget}.json")
            if (not 0 <= i < 90 or i in indexed or r["attention_backend"] != backend
                    or r["nominal_history_q"] != budget or r["cap"] != receipt["cap"]
                    or r["protocol"] != receipt["protocol"] or r["status"] != "success"
                    or type(r["score"]["correct"]) is not bool
                    or any(r[k] != tasks[i][k] for k in ("index", "seed", "task_id", "dataset"))):
                raise ValueError(f"Duplicate or mismatched task or score: {shard} {i}")
            pre, *calls = r["events"]
            if (pre != ["encoder", r["prefill_rows"]] or any(k not in ("decoder","encoder") or n != 256
                                                             for k, n in calls)
                    or len(calls) != r["generation_forwards"]
                    or sum(n for _, n in calls) != r["generation_physical_padded_rows"]
                    or sum(x["history_q"] for x in r["trace"]) != r["history_query_rows"]
                    or sum(k == "decoder" for k, _ in calls) != len(r["trace"])):
                raise ValueError(f"Work counters do not match actual calls: {shard} {i}")
            indexed[i] = r
    if set(indexed) != set(range(90)):
        raise ValueError("Expected exactly 90 unique tasks")
    for name, ids in (("original30", range(30)), ("added60", range(30, 90)), ("all90", range(90))):
        rows = [indexed[i] for i in ids]
        def total(key):
            return sum(r[key] for r in rows)
        n = total("output_tokens")
        q = total("generation_physical_padded_rows")
        yield dict(C=budget, cohort=name, requests=len(rows), correct=sum(r["score"]["correct"] for r in rows),
                   macro_tpf=fmean(r["output_tokens"] / r["generation_forwards"] for r in rows),
                   token_efficiency=n/q, N=n, F=total("generation_forwards"), physical_Q=q,
                   history_Q=total("history_query_rows"), decoder_Q=total("decoder_query_rows"),
                   reencode_Q=total("reencode_query_rows"), truncated=total("truncated"),
                   close_excess=total("close_history_excess"), backend=backend,
                   device=gate["gpu"], gate_sha256=gate_hash)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--budget", type=int, choices=(32,64,128,256,512), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("shards", type=Path, nargs="+")
    args = p.parse_args()
    rows = list(summarize(args.samples, args.gate, args.budget, args.shards))
    with args.output.open("x", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
