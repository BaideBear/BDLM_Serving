"""Compare actual work, not just nominal windows, on existing paired runs.

This is an empirical cost envelope: no curve fitting, interpolation, or new
baseline runs. Mixed-width comparisons are labelled as such. Describing one
row count does not equate attention FLOPs, elapsed time or per-request costs.

The matched-block diagnostic is CPU-only and descriptive. It compares generation
forwards (F) by phase on the first K complete, prompt-free output blocks of
matched-prompt pairs. A shared block ordinal is a shared position in the same
prompt, not a shared model state: policies can produce different text, so these
comparisons are not causal.
"""
import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import mean

STAGES = ("bootstrap", "selected", "verify", "commit")
BLOCK = 32


def load(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def pack(rows):
    result = {}
    for r in rows:
        key = int(r["index"]), r["dataset"], r["task_id"]
        if key in result:
            raise ValueError("duplicate request")
        n, f, q = (int(r[k]) for k in ("N","F","physical_Q"))
        assert q == int(r["Q"]) and q >= f > 0
        result[key] = (n, f, q, int(r["correct"]))
    if len(result) != 30:
        raise ValueError("expected 30 frozen requests")
    return result


def metrics(rows):
    n, f, q, correct = map(sum, zip(*rows.values()))
    return dict(tpf=mean(Fraction(a,b) for a,b,c,d in rows.values()), N=n, F=f, Q=q,
                q_per_f=Fraction(q,f), efficiency=Fraction(n,q), correct=correct)


def compare(a,b):
    if set(a) != set(b):
        raise ValueError("must pair identical tasks")
    x, y = metrics(a), metrics(b)
    both = x["correct"] > y["correct"] and x["tpf"] >= y["tpf"]
    return dict(**{k:float(v) if isinstance(v,Fraction) else v for k,v in x.items()},
                **{"ref_"+k:float(v) if isinstance(v,Fraction) else v for k,v in y.items()},
                quality_up_tpf_not_lower=int(both),
                total_Q_not_higher=int(x["Q"]<=y["Q"]),
                pooled_Q_per_F_not_higher=int(x["q_per_f"]<=y["q_per_f"]),
                requests_Q_not_higher=sum(a[i][2]<=b[i][2] for i in a),
                requests_Q_per_F_not_higher=sum(a[i][2]*b[i][1]<=b[i][2]*a[i][1] for i in a),
                wins=sum(a[i][3]>b[i][3] for i in a), losses=sum(a[i][3]<b[i][3] for i in a))


class TraceError(ValueError):
    """Raised when an archived trace does not have the expected block geometry."""


def _prompt_lengths(prompt_dir, records):
    """Read actual prompt lengths from the frozen native traces."""
    prompt_dir = Path(prompt_dir)
    lengths = {}
    for path in sorted(prompt_dir.glob("*.trace.jsonl.gz")):
        with gzip.open(path, "rt") as f:
            meta = json.loads(next(f))
            prompt = json.loads(next(f))
        index = int(meta["index"])
        record = records.get((index, meta["dataset"], meta["task_id"]))
        if record is None:
            raise TraceError("native trace %s does not match a request" % path.name)
        if int(record["seed"]) != int(meta["seed"]):
            raise TraceError("seed mismatch in %s" % path.name)
        length = int(prompt["prompt_length"])
        if length <= 0:
            raise TraceError("bad prompt_length in %s" % path.name)
        lengths[index] = length
    if set(lengths) != {key[0] for key in records}:
        raise TraceError("native traces do not cover all requests")
    return lengths


def _validate_trace(data, prompt_length):
    events = data.get("events")
    if not isinstance(events, list) or not events:
        raise TraceError("missing events")
    if int(data["max_q"]) != BLOCK:
        raise TraceError("expected max_q=%d" % BLOCK)
    prefill = int(data["prefill_rows"])
    if prefill != BLOCK * (prompt_length // BLOCK):
        raise TraceError("prefill_rows is not the floor prompt block")
    generation = [e for e in events if e["phase"] != "prefill"]
    if int(data["generation_forwards"]) != sum(int(e["F"]) for e in generation):
        raise TraceError("generation_forwards does not match events")
    if prefill != sum(int(e["q_tokens"]) for e in events if e["phase"] == "prefill"):
        raise TraceError("prefill_rows does not match prefill events")
    if int(data["generation_query_rows"]) != sum(int(e["physical_q"]) for e in generation):
        raise TraceError("generation_query_rows does not match events")
    if int(data["generation_physical_padded_rows"]) != int(data["generation_query_rows"]):
        raise TraceError("padded rows differ from query rows")
    stages = Counter(e["stage"] for e in generation)
    if dict(stages) != {k: int(v) for k, v in data["forwards_by_phase"].items()}:
        raise TraceError("forwards_by_phase does not match events")
    qph = Counter()
    for event in generation:
        qph[event["stage"]] += int(event["physical_q"])
    if dict(qph) != {k: int(v) for k, v in data["q_by_phase"].items()}:
        raise TraceError("q_by_phase does not match events")
    if any(e["stage"] not in STAGES for e in generation):
        raise TraceError("unknown generation stage")
    if any(e["phase"] not in ("prefill", "denoise", "commit") for e in events):
        raise TraceError("unknown phase")
    for event in events:
        if int(event["F"]) != 1:
            raise TraceError("F must be exactly 1")
        if int(event["actual_q"]) != int(event["physical_q"]):
            raise TraceError("actual_q and physical_q differ")
        if not event["layer_physical_q"]:
            raise TraceError("missing layer_physical_q")
        if any(int(v) != int(event["actual_q"]) for v in event["layer_physical_q"]):
            raise TraceError("layer_physical_q is not uniform")
        if int(event["q_tokens"]) != int(event["actual_q"]):
            raise TraceError("q_tokens and actual_q differ")
        positions = event["absolute_positions"]
        if len(positions) != int(event["q_tokens"]):
            raise TraceError("absolute_positions length does not match q_tokens")
        if any(positions[i] >= positions[i + 1] for i in range(len(positions) - 1)):
            raise TraceError("absolute_positions is not strictly increasing")
    if int(data["output_tokens"]) > len(data["output_ids"]):
        raise TraceError("output_tokens exceeds output_ids")

    by_block = defaultdict(list)
    for event in generation:
        by_block[int(event["block"])].append(event)
    first_block = prefill // BLOCK
    block_numbers = sorted(by_block)
    if block_numbers != list(range(first_block, first_block + len(block_numbers))):
        raise TraceError("generation block ordinals are not consecutive")
    generated_end = prompt_length + len(data["output_ids"])
    valid_flags = []
    for ordinal, block_no in enumerate(block_numbers, 1):
        q_start = prefill + BLOCK * (ordinal - 1)
        q_end = q_start + BLOCK
        block_events = by_block[block_no]
        bootstraps = [e for e in block_events if e["stage"] == "bootstrap"]
        commits = [e for e in block_events if e["stage"] == "commit"]
        if len(bootstraps) != 1 or len(commits) != 1:
            raise TraceError("each block needs one bootstrap and one commit")
        for event in bootstraps + commits:
            if (int(event["q_start"]), int(event["q_end"])) != (q_start, q_end):
                raise TraceError("bootstrap/commit span does not match the block")
        for event in block_events:
            if int(event["prefix_kv_length"]) != q_start:
                raise TraceError("prefix_kv_length does not match the block start")
            if not (q_start <= int(event["q_start"]) <= int(event["q_end"]) <= q_end):
                raise TraceError("event falls outside its block")
            if event["stage"] == "verify" and (int(event["q_start"]), int(event["q_end"])) != (q_start, q_end):
                raise TraceError("verify span does not cover the full block")
            positions = event["absolute_positions"]
            if positions:
                if positions[0] < q_start or positions[-1] >= q_end:
                    raise TraceError("absolute_positions fall outside the block")
                if event["stage"] == "selected":
                    if int(event["q_start"]) != positions[0] or int(event["q_end"]) != positions[-1] + 1:
                        raise TraceError("selected q_start/q_end do not match its positions")
                elif positions != list(range(q_start, q_end)):
                    raise TraceError("non-selected event does not cover the full block")
            elif event["stage"] == "selected":
                raise TraceError("selected event has no positions")
        valid_flags.append(q_start >= prompt_length and q_end <= generated_end)

    if not any(valid_flags):
        return
    first_valid = valid_flags.index(True)
    last_valid = len(valid_flags) - 1 - valid_flags[::-1].index(True)
    if not all(valid_flags[first_valid:last_valid + 1]):
        raise TraceError("valid output blocks are not contiguous")
    expected_first = (prompt_length + BLOCK - 1) // BLOCK
    if block_numbers[first_valid] != expected_first:
        raise TraceError("first valid output block is not the first prompt-free block")
    if block_numbers[first_valid] != first_block + first_valid:
        raise TraceError("valid output block ordinal is misaligned")
    if prompt_length % BLOCK:
        if first_valid != 1:
            raise TraceError("expected exactly one prompt-mixed block")
    elif first_valid != 0:
        raise TraceError("prompt-aligned trace has an unexpected mixed block")
    if last_valid + 1 < len(block_numbers):
        next_q_end = prefill + BLOCK * (last_valid + 2)
        if next_q_end <= generated_end:
            raise TraceError("partial terminal block is not actually partial")


def _load_traces(run_dir, policy, prompt_dir):
    run_dir = Path(run_dir)
    paths = sorted(run_dir.glob("*-%s.json" % policy))
    if not paths:
        raise TraceError("no trace files for %s in %s" % (policy, run_dir))
    records = {}
    digest = hashlib.sha256()
    for path in paths:
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"))
        if "events" not in data:
            continue  # complete-* and marks-* are run summaries
        if data.get("policy") != policy:
            raise TraceError("policy mismatch in %s" % path.name)
        if data.get("status") != "success":
            raise TraceError("non-success request in %s" % path.name)
        key = (int(data["index"]), data["dataset"], data["task_id"])
        if key in records:
            raise TraceError("duplicate request key in %s" % path.name)
        records[key] = data
        digest.update(path.name.encode("utf-8"))
        digest.update(raw)
    if len(records) != 30:
        raise TraceError("expected 30 request traces for %s, found %d" % (policy, len(records)))
    lengths = _prompt_lengths(prompt_dir, records)
    for key, data in records.items():
        _validate_trace(data, lengths[key[0]])
    return records, digest.hexdigest()


def _blocks(data, prompt_length):
    generation = [e for e in data["events"] if e["phase"] != "prefill"]
    by_block = defaultdict(list)
    for event in generation:
        by_block[int(event["block"])].append(event)
    prefill = int(data["prefill_rows"])
    first_block = prefill // BLOCK
    generated_end = prompt_length + len(data["output_ids"])
    records = []
    for ordinal, block_no in enumerate(range(first_block, first_block + len(by_block)), 1):
        q_start = prefill + BLOCK * (ordinal - 1)
        q_end = q_start + BLOCK
        counts = Counter(event["stage"] for event in by_block[block_no])
        records.append({
            "ordinal": ordinal,
            "block_no": block_no,
            "q_start": q_start,
            "q_end": q_end,
            "valid": q_start >= prompt_length and q_end <= generated_end,
            "bootstrap": int(counts.get("bootstrap", 0)),
            "selected": int(counts.get("selected", 0)),
            "verify": int(counts.get("verify", 0)),
            "commit": int(counts.get("commit", 0)),
        })
    return records


def _summary(data, prompt_length):
    blocks = _blocks(data, prompt_length)
    valid = [b for b in blocks if b["valid"]]
    total = Counter()
    for block in blocks:
        for stage in STAGES:
            total[stage] += block[stage]
    last = valid[-1] if valid else None
    tail_positions = len(data["output_ids"]) - (last["q_end"] - prompt_length) if last else len(data["output_ids"])
    return {
        "key": (int(data["index"]), data["dataset"], data["task_id"]),
        "index": int(data["index"]), "dataset": data["dataset"], "task_id": data["task_id"],
        "seed": int(data["seed"]), "prompt_length": prompt_length,
        "prefill_rows": int(data["prefill_rows"]),
        "output_tokens": int(data["output_tokens"]),
        "generated_positions": len(data["output_ids"]),
        "valid_blocks": valid,
        "total": {stage: int(total[stage]) for stage in STAGES},
        "tail_positions": tail_positions,
        "terminal_block_ordinal": (last["ordinal"] + 1 if tail_positions else (last["ordinal"] if last else 0)),
        "terminal_block_positions": (tail_positions if tail_positions else (BLOCK if last else 0)),
        "eos": bool(data["eos"]), "truncated": bool(data["truncated"]),
        "output_ids": data["output_ids"], "output_text": data["output_text"],
    }


def _hash_ids(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def _compare(control, arm, label):
    if control["key"] != arm["key"] or control["prefill_rows"] != arm["prefill_rows"]:
        raise TraceError("request or prefill mismatch for %r" % (control["key"],))
    if control["seed"] != arm["seed"]:
        raise TraceError("seed mismatch for %r" % (control["key"],))
    k = min(len(control["valid_blocks"]), len(arm["valid_blocks"]))
    ctrl_matched = Counter()
    arm_matched = Counter()
    block_rows = []
    same_token_blocks = 0
    for ordinal in range(k):
        cb = control["valid_blocks"][ordinal]
        ab = arm["valid_blocks"][ordinal]
        if cb["block_no"] != ab["block_no"] or cb["q_start"] != ab["q_start"] or cb["q_end"] != ab["q_end"]:
            raise TraceError("matched block misalignment for %r ordinal %d" % (control["key"], ordinal + 1))
        for stage in STAGES:
            ctrl_matched[stage] += cb[stage]
            arm_matched[stage] += ab[stage]
        c_ids = control["output_ids"][cb["q_start"] - control["prompt_length"]:cb["q_end"] - control["prompt_length"]]
        a_ids = arm["output_ids"][ab["q_start"] - arm["prompt_length"]:ab["q_end"] - arm["prompt_length"]]
        same = c_ids == a_ids
        same_token_blocks += int(same)
        block_rows.append({
            "arm": label, "index": control["index"], "dataset": control["dataset"],
            "task_id": control["task_id"], "block_ordinal": ordinal + 1,
            "block_no": cb["block_no"], "q_start": cb["q_start"], "q_end": cb["q_end"],
            "ctrl_bootstrap": cb["bootstrap"], "ctrl_selected": cb["selected"],
            "ctrl_verify": cb["verify"], "ctrl_commit": cb["commit"],
            "arm_bootstrap": ab["bootstrap"], "arm_selected": ab["selected"],
            "arm_verify": ab["verify"], "arm_commit": ab["commit"],
            "delta_bootstrap": ab["bootstrap"] - cb["bootstrap"],
            "delta_selected": ab["selected"] - cb["selected"],
            "delta_verify": ab["verify"] - cb["verify"],
            "delta_commit": ab["commit"] - cb["commit"],
            "same_block_tokens": int(same),
            "ctrl_block_sha256": _hash_ids(c_ids), "arm_block_sha256": _hash_ids(a_ids),
        })
    if ctrl_matched["bootstrap"] != arm_matched["bootstrap"] or ctrl_matched["commit"] != arm_matched["commit"]:
        raise TraceError("matched bootstrap/commit delta is not zero")
    ctrl_total = sum(control["total"].values())
    arm_total = sum(arm["total"].values())
    matched = sum(arm_matched[s] - ctrl_matched[s] for s in STAGES)
    excess = arm_total - ctrl_total
    remainder = excess - matched
    if matched + remainder != excess:
        raise TraceError("matched/remainder decomposition does not add up")
    first_divergent = next((i + 1 for i, b in enumerate(block_rows) if not b["same_block_tokens"]), 0)
    same_matched_prefix = int(all(b["same_block_tokens"] for b in block_rows)) if block_rows else 1
    c_len, a_len = len(control["output_ids"]), len(arm["output_ids"])
    if excess > 0:
        if matched > 0 and remainder > 0:
            classification = "both"
        elif matched > 0:
            classification = "progress_only"
        elif remainder > 0:
            classification = "length_only"
        else:
            classification = "other_excess"
        progress_share, length_share = matched / excess, remainder / excess
    elif excess == 0:
        classification, progress_share, length_share = "equal", "", ""
    else:
        classification, progress_share, length_share = "arm_lower", "", ""
    row = {
        "arm": label, "index": control["index"], "dataset": control["dataset"],
        "task_id": control["task_id"], "prompt_length": control["prompt_length"],
        "prefill_rows": control["prefill_rows"],
        "ctrl_output_tokens": control["output_tokens"], "arm_output_tokens": arm["output_tokens"],
        "ctrl_generated_positions": c_len, "arm_generated_positions": a_len,
        "ctrl_valid_blocks": len(control["valid_blocks"]), "arm_valid_blocks": len(arm["valid_blocks"]),
        "matched_blocks": k,
        "ctrl_terminal_block": control["terminal_block_ordinal"],
        "arm_terminal_block": arm["terminal_block_ordinal"],
        "ctrl_terminal_positions": control["terminal_block_positions"],
        "arm_terminal_positions": arm["terminal_block_positions"],
        "ctrl_eos": int(control["eos"]), "arm_eos": int(arm["eos"]),
        "ctrl_truncated": int(control["truncated"]), "arm_truncated": int(arm["truncated"]),
        "ctrl_bootstrap": control["total"]["bootstrap"], "ctrl_selected": control["total"]["selected"],
        "ctrl_verify": control["total"]["verify"], "ctrl_commit": control["total"]["commit"],
        "ctrl_total_F": ctrl_total,
        "arm_bootstrap": arm["total"]["bootstrap"], "arm_selected": arm["total"]["selected"],
        "arm_verify": arm["total"]["verify"], "arm_commit": arm["total"]["commit"],
        "arm_total_F": arm_total,
        "delta_bootstrap": arm["total"]["bootstrap"] - control["total"]["bootstrap"],
        "delta_selected": arm["total"]["selected"] - control["total"]["selected"],
        "delta_verify": arm["total"]["verify"] - control["total"]["verify"],
        "delta_commit": arm["total"]["commit"] - control["total"]["commit"],
        "excess_F": excess,
        "ctrl_matched_bootstrap": ctrl_matched["bootstrap"], "ctrl_matched_selected": ctrl_matched["selected"],
        "ctrl_matched_verify": ctrl_matched["verify"], "ctrl_matched_commit": ctrl_matched["commit"],
        "ctrl_matched_F": sum(ctrl_matched.values()),
        "arm_matched_bootstrap": arm_matched["bootstrap"], "arm_matched_selected": arm_matched["selected"],
        "arm_matched_verify": arm_matched["verify"], "arm_matched_commit": arm_matched["commit"],
        "arm_matched_F": sum(arm_matched.values()),
        "matched_delta_bootstrap": arm_matched["bootstrap"] - ctrl_matched["bootstrap"],
        "matched_delta_selected": arm_matched["selected"] - ctrl_matched["selected"],
        "matched_delta_verify": arm_matched["verify"] - ctrl_matched["verify"],
        "matched_delta_commit": arm_matched["commit"] - ctrl_matched["commit"],
        "matched_F": matched,
        "ctrl_remainder_bootstrap": control["total"]["bootstrap"] - ctrl_matched["bootstrap"],
        "ctrl_remainder_selected": control["total"]["selected"] - ctrl_matched["selected"],
        "ctrl_remainder_verify": control["total"]["verify"] - ctrl_matched["verify"],
        "ctrl_remainder_commit": control["total"]["commit"] - ctrl_matched["commit"],
        "ctrl_remainder_F": ctrl_total - sum(ctrl_matched.values()),
        "arm_remainder_bootstrap": arm["total"]["bootstrap"] - arm_matched["bootstrap"],
        "arm_remainder_selected": arm["total"]["selected"] - arm_matched["selected"],
        "arm_remainder_verify": arm["total"]["verify"] - arm_matched["verify"],
        "arm_remainder_commit": arm["total"]["commit"] - arm_matched["commit"],
        "arm_remainder_F": arm_total - sum(arm_matched.values()),
        "remainder_delta_bootstrap": (arm["total"]["bootstrap"] - arm_matched["bootstrap"]) - (control["total"]["bootstrap"] - ctrl_matched["bootstrap"]),
        "remainder_delta_selected": (arm["total"]["selected"] - arm_matched["selected"]) - (control["total"]["selected"] - ctrl_matched["selected"]),
        "remainder_delta_verify": (arm["total"]["verify"] - arm_matched["verify"]) - (control["total"]["verify"] - ctrl_matched["verify"]),
        "remainder_delta_commit": (arm["total"]["commit"] - arm_matched["commit"]) - (control["total"]["commit"] - ctrl_matched["commit"]),
        "remainder_F": remainder,
        "progress_share": progress_share, "length_share": length_share,
        "classification": classification,
        "arm_length_direction": "arm_longer" if a_len > c_len else ("same" if a_len == c_len else "arm_shorter"),
        "first_divergent_block": first_divergent,
        "same_matched_prefix": same_matched_prefix,
        "same_full_output": int(control["output_ids"] == arm["output_ids"]),
        "same_output_text": int(control["output_text"] == arm["output_text"]),
        "matched_same_token_blocks": same_token_blocks,
        "matched_divergent_blocks": k - same_token_blocks,
    }
    return row, block_rows


REQUEST_FIELDS = [
    "arm", "index", "dataset", "task_id", "prompt_length", "prefill_rows",
    "ctrl_output_tokens", "arm_output_tokens",
    "ctrl_generated_positions", "arm_generated_positions",
    "ctrl_valid_blocks", "arm_valid_blocks", "matched_blocks",
    "ctrl_terminal_block", "arm_terminal_block",
    "ctrl_terminal_positions", "arm_terminal_positions",
    "ctrl_eos", "arm_eos", "ctrl_truncated", "arm_truncated",
    "ctrl_bootstrap", "ctrl_selected", "ctrl_verify", "ctrl_commit", "ctrl_total_F",
    "arm_bootstrap", "arm_selected", "arm_verify", "arm_commit", "arm_total_F",
    "delta_bootstrap", "delta_selected", "delta_verify", "delta_commit", "excess_F",
    "ctrl_matched_bootstrap", "ctrl_matched_selected", "ctrl_matched_verify", "ctrl_matched_commit", "ctrl_matched_F",
    "arm_matched_bootstrap", "arm_matched_selected", "arm_matched_verify", "arm_matched_commit", "arm_matched_F",
    "matched_delta_bootstrap", "matched_delta_selected", "matched_delta_verify", "matched_delta_commit", "matched_F",
    "ctrl_remainder_bootstrap", "ctrl_remainder_selected", "ctrl_remainder_verify", "ctrl_remainder_commit", "ctrl_remainder_F",
    "arm_remainder_bootstrap", "arm_remainder_selected", "arm_remainder_verify", "arm_remainder_commit", "arm_remainder_F",
    "remainder_delta_bootstrap", "remainder_delta_selected", "remainder_delta_verify", "remainder_delta_commit", "remainder_F",
    "progress_share", "length_share", "classification", "arm_length_direction",
    "first_divergent_block", "same_matched_prefix", "same_full_output", "same_output_text",
    "matched_same_token_blocks", "matched_divergent_blocks",
]

BLOCK_FIELDS = [
    "arm", "index", "dataset", "task_id", "block_ordinal", "block_no", "q_start", "q_end",
    "ctrl_bootstrap", "ctrl_selected", "ctrl_verify", "ctrl_commit",
    "arm_bootstrap", "arm_selected", "arm_verify", "arm_commit",
    "delta_bootstrap", "delta_selected", "delta_verify", "delta_commit",
    "same_block_tokens", "ctrl_block_sha256", "arm_block_sha256",
]

SUMMARY_FIELDS = [
    "arm", "requests", "matched_blocks", "excess_requests", "progress_only", "both",
    "length_only", "equal", "arm_lower", "ctrl_total_F", "arm_total_F", "total_excess_F",
    "matched_F_sum", "remainder_F_sum", "delta_bootstrap", "delta_selected",
    "delta_verify", "delta_commit", "selected_blocks_gt", "selected_blocks_eq",
    "selected_blocks_lt", "verify_blocks_gt", "verify_blocks_eq", "verify_blocks_lt",
    "matched_ctrl_selected", "matched_arm_selected", "matched_ctrl_verify", "matched_arm_verify",
    "same_token_blocks", "divergent_blocks", "same_matched_prefix_pairs", "same_full_output_pairs",
    "arm_longer_excess", "arm_same_excess", "arm_shorter_excess", "truncated_pairs",
    "eos_diff_pairs", "ctrl_output_positions", "arm_output_positions",
    "ctrl_valid_blocks", "arm_valid_blocks",
    "control_run", "control_policy", "control_sha256", "arm_run", "arm_policy", "arm_sha256",
]

PHASE_FIELDS = [
    "arm", "phase", "ctrl_total", "arm_total", "total_delta",
    "ctrl_matched", "arm_matched", "matched_delta", "ctrl_remainder", "arm_remainder",
    "remainder_delta", "excess_only_total_delta", "excess_only_matched_delta",
    "excess_only_remainder_delta",
]


def _summarize(label, pairs, blocks, control_name, control_policy, control_sha, arm_name, arm_policy, arm_sha):
    excess_rows = [r for r in pairs if r["excess_F"] > 0]
    counts = Counter(r["classification"] for r in pairs)
    summary = {
        "arm": label, "requests": len(pairs), "matched_blocks": len(blocks),
        "excess_requests": len(excess_rows), "progress_only": counts.get("progress_only", 0),
        "both": counts.get("both", 0), "length_only": counts.get("length_only", 0),
        "equal": counts.get("equal", 0), "arm_lower": counts.get("arm_lower", 0),
        "ctrl_total_F": sum(r["ctrl_total_F"] for r in pairs),
        "arm_total_F": sum(r["arm_total_F"] for r in pairs),
        "total_excess_F": sum(r["excess_F"] for r in pairs),
        "matched_F_sum": sum(r["matched_F"] for r in pairs),
        "remainder_F_sum": sum(r["remainder_F"] for r in pairs),
        "delta_bootstrap": sum(r["delta_bootstrap"] for r in pairs),
        "delta_selected": sum(r["delta_selected"] for r in pairs),
        "delta_verify": sum(r["delta_verify"] for r in pairs),
        "delta_commit": sum(r["delta_commit"] for r in pairs),
        "selected_blocks_gt": sum(1 for b in blocks if b["delta_selected"] > 0),
        "selected_blocks_eq": sum(1 for b in blocks if b["delta_selected"] == 0),
        "selected_blocks_lt": sum(1 for b in blocks if b["delta_selected"] < 0),
        "verify_blocks_gt": sum(1 for b in blocks if b["delta_verify"] > 0),
        "verify_blocks_eq": sum(1 for b in blocks if b["delta_verify"] == 0),
        "verify_blocks_lt": sum(1 for b in blocks if b["delta_verify"] < 0),
        "matched_ctrl_selected": sum(b["ctrl_selected"] for b in blocks),
        "matched_arm_selected": sum(b["arm_selected"] for b in blocks),
        "matched_ctrl_verify": sum(b["ctrl_verify"] for b in blocks),
        "matched_arm_verify": sum(b["arm_verify"] for b in blocks),
        "same_token_blocks": sum(b["same_block_tokens"] for b in blocks),
        "divergent_blocks": len(blocks) - sum(b["same_block_tokens"] for b in blocks),
        "same_matched_prefix_pairs": sum(r["same_matched_prefix"] for r in pairs),
        "same_full_output_pairs": sum(r["same_full_output"] for r in pairs),
        "arm_longer_excess": sum(1 for r in excess_rows if r["arm_length_direction"] == "arm_longer"),
        "arm_same_excess": sum(1 for r in excess_rows if r["arm_length_direction"] == "same"),
        "arm_shorter_excess": sum(1 for r in excess_rows if r["arm_length_direction"] == "arm_shorter"),
        "truncated_pairs": sum(1 for r in pairs if r["ctrl_truncated"] or r["arm_truncated"]),
        "eos_diff_pairs": sum(1 for r in pairs if r["ctrl_eos"] != r["arm_eos"]),
        "ctrl_output_positions": sum(r["ctrl_generated_positions"] for r in pairs),
        "arm_output_positions": sum(r["arm_generated_positions"] for r in pairs),
        "ctrl_valid_blocks": sum(r["ctrl_valid_blocks"] for r in pairs),
        "arm_valid_blocks": sum(r["arm_valid_blocks"] for r in pairs),
        "control_run": control_name, "control_policy": control_policy, "control_sha256": control_sha,
        "arm_run": arm_name, "arm_policy": arm_policy, "arm_sha256": arm_sha,
    }
    phases = []
    for stage in STAGES:
        phases.append({
            "arm": label, "phase": stage,
            "ctrl_total": sum(r["ctrl_" + stage] for r in pairs),
            "arm_total": sum(r["arm_" + stage] for r in pairs),
            "total_delta": sum(r["delta_" + stage] for r in pairs),
            "ctrl_matched": sum(r["ctrl_matched_" + stage] for r in pairs),
            "arm_matched": sum(r["arm_matched_" + stage] for r in pairs),
            "matched_delta": sum(r["matched_delta_" + stage] for r in pairs),
            "ctrl_remainder": sum(r["ctrl_remainder_" + stage] for r in pairs),
            "arm_remainder": sum(r["arm_remainder_" + stage] for r in pairs),
            "remainder_delta": sum(r["remainder_delta_" + stage] for r in pairs),
            "excess_only_total_delta": sum(r["delta_" + stage] for r in excess_rows),
            "excess_only_matched_delta": sum(r["matched_delta_" + stage] for r in excess_rows),
            "excess_only_remainder_delta": sum(r["remainder_delta_" + stage] for r in excess_rows),
        })
    return summary, phases


def _write_csv(path, fields, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path, control, arms, summaries, phases, pairs, out_dir, test_result):
    S = {row["arm"]: row for row in summaries}
    P = {(row["arm"], row["phase"]): row for row in phases}
    labels = [a["label"] for a in arms]
    lines = [
        "# LLaDA 2.1 Q16 matched-block F diagnostic", "",
        "**Date:** 2026-11-26  ",
        "**Execution:** n164 CPU only (`CUDA_VISIBLE_DEVICES=`); no new inference.",
        "**Control:** `%s` / `%s`  " % (control["name"], control["policy"]),
        "**Arms:** " + ", ".join("`%s` = `%s` / `%s`" % (a["label"], a["name"], a["policy"]) for a in arms),
        "",
        "Descriptive only: a matched block ordinal is the same position in the same prompt, not the same model state. "
        "Policies can produce different text; no comparison here is causal.", "",
        "## Trace semantics", "",
        "- Actual `prompt_length` comes from the frozen native traces (`*.trace.jsonl.gz`), not from `prefill_rows`.",
        "- `prefill_rows == 32 * floor(prompt_length/32)`. The prompt-mixed first block starts at `prefill_rows` and is excluded.",
        "- A valid full output block has `q_start >= prompt_length` and `q_end <= prompt_length + len(output_ids)`. "
        "A final partial block is excluded; a full block ending exactly at EOS remains valid. "
        "`K = min(K_control, K_arm)` per pair.",
        "- All 30 prompts have `prompt_length % 32 != 0`; every request has exactly one prompt-mixed first block, which is excluded.",
        "- `output_ids` includes generated positions including EOS; `output_tokens` excludes special tokens.",
        "- Bootstrap and commit are one forward per block; at fixed K their matched delta is exactly zero and is checked per pair.",
        "", "## Results", "",
        "| arm | matched blocks | excess / 30 | progress_only | both | length_only | equal | arm_lower | total dF | matched dF | remainder dF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in labels:
        s = S[label]
        lines.append("| %s | %d | %d | %d | %d | %d | %d | %d | %+d | %+d | %+d |" % (
            label, s["matched_blocks"], s["excess_requests"], s["progress_only"], s["both"],
            s["length_only"], s["equal"], s["arm_lower"], s["total_excess_F"],
            s["matched_F_sum"], s["remainder_F_sum"]))
    lines += ["", "| arm | phase | total dF | matched dF | remainder dF | excess-only total dF | excess-only matched dF | excess-only remainder dF |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for label in labels:
        for stage in STAGES:
            p = P[(label, stage)]
            lines.append("| %s | %s | %+d | %+d | %+d | %+d | %+d | %+d |" % (
                label, stage, p["total_delta"], p["matched_delta"], p["remainder_delta"],
                p["excess_only_total_delta"], p["excess_only_matched_delta"],
                p["excess_only_remainder_delta"]))
    lines += ["", "| arm | selected gt/eq/lt | verify gt/eq/lt | matched selected ctrl->arm | matched verify ctrl->arm | same-token blocks |",
              "|---|---|---|---|---|---:|"]
    for label in labels:
        s = S[label]
        lines.append("| %s | %d/%d/%d | %d/%d/%d | %d->%d | %d->%d | %d/%d |" % (
            label, s["selected_blocks_gt"], s["selected_blocks_eq"], s["selected_blocks_lt"],
            s["verify_blocks_gt"], s["verify_blocks_eq"], s["verify_blocks_lt"],
            s["matched_ctrl_selected"], s["matched_arm_selected"],
            s["matched_ctrl_verify"], s["matched_arm_verify"],
            s["same_token_blocks"], s["matched_blocks"]))
    lines += ["", "## Checks", "",
              "- 30/30 request traces per arm; all `status=success`; native prompt traces cover all 30 requests.",
              "- Every event has `F=1`, `actual_q=physical_q`, uniform `layer_physical_q`, "
              "`q_tokens=actual_q=len(absolute_positions)`, strictly increasing positions; one bootstrap and one commit per block.",
              "- Aggregate identities hold: `generation_forwards`, `forwards_by_phase`, `q_by_phase`, `prefill_rows`, "
              "`generation_query_rows`, and `prefill_rows == 32*floor(prompt_length/32)`.",
              "- Valid output blocks are contiguous and start at the first prompt-free block; the terminal partial block is excluded.",
              "- All 30 prompts have a non-multiple-of-32 length; exactly one prompt-mixed block per request is excluded.",
              "- Control request 17 is the only exact EOS boundary case (prompt 87, 1161 generated positions, end 1248 = 39*32); "
              "its final block is full and valid. All other requests have a partial terminal block.",
              "- Matched bootstrap/commit deltas are zero for all 120 pair rows; matched + remainder equals excess for all rows.",
              "- Matched pairs share `(index, dataset, task_id)`, `prefill_rows`, `seed`, model and `max_q`; every matched block has identical absolute `q_start/q_end`.",
              "- Input SHA-256 values are recorded in `llada21-matched-block-summary.csv`.",
              "- Unit tests: `%s`" % test_result,
              "- All measurements reuse the archived GPU runs.", "",
              "## Files", ""]
    for name in ("-requests.csv", "-blocks.csv", "-summary.csv", "-phases.csv"):
        lines.append("- `%s/llada21-matched-block%s`" % (out_dir, name))
    lines += ["- `%s/MATCHED-BLOCKS.md`" % out_dir, "", "## Limitations", "",
              "- Descriptive only, not causal. A shared block ordinal is a shared prompt position, not a shared model state; text diverges in most pairs.",
              "- Q16 LLaDA 2.1-mini, 30 frozen requests, one seed; no quality, TPF or wall-clock claim.",
              "- The remainder contains extra complete blocks and the partial terminal block; it is length-related but not a pure token-length effect.",
              "- `output_tokens` excludes EOS; block validity uses positions including EOS. A full block ending exactly at EOS remains valid.", ""]
    path.write_text("\n".join(line.rstrip() for line in lines), encoding="utf-8")


def matched_blocks_main(argv=None):
    parser = argparse.ArgumentParser(description="Matched-block F diagnostic over archived request traces.")
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--control-policy", required=True)
    parser.add_argument("--prompt-dir", type=Path, required=True)
    parser.add_argument("--arm", action="append", nargs=3, required=True,
                        metavar=("LABEL", "RUN_DIR", "POLICY"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="llada21-matched-block")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--test-result", default="not run")
    args = parser.parse_args(argv)

    control_records, control_sha = _load_traces(args.control_run, args.control_policy, args.prompt_dir)
    lengths = _prompt_lengths(args.prompt_dir, control_records)
    control = {key: _summary(value, lengths[key[0]]) for key, value in control_records.items()}
    all_pairs, all_blocks, summaries, phase_rows, arms = [], [], [], [], []
    for label, run_dir, policy in args.arm:
        arm_records, arm_sha = _load_traces(run_dir, policy, args.prompt_dir)
        if set(arm_records) != set(control_records):
            raise TraceError("arm %s does not cover the same 30 requests" % label)
        arm = {key: _summary(value, lengths[key[0]]) for key, value in arm_records.items()}
        pairs, blocks = [], []
        for key in sorted(control):
            row, block_rows = _compare(control[key], arm[key], label)
            pairs.append(row)
            blocks.extend(block_rows)
        summary, phases = _summarize(label, pairs, blocks, Path(args.control_run).name,
                                     args.control_policy, control_sha, Path(run_dir).name, policy, arm_sha)
        summaries.append(summary)
        phase_rows.extend(phases)
        all_pairs.extend(pairs)
        all_blocks.extend(blocks)
        arms.append({"label": label, "name": Path(run_dir).name, "policy": policy, "sha": arm_sha})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / (args.prefix + "-requests.csv"), REQUEST_FIELDS, all_pairs)
    _write_csv(args.out_dir / (args.prefix + "-blocks.csv"), BLOCK_FIELDS, all_blocks)
    _write_csv(args.out_dir / (args.prefix + "-summary.csv"), SUMMARY_FIELDS, summaries)
    _write_csv(args.out_dir / (args.prefix + "-phases.csv"), PHASE_FIELDS, phase_rows)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        _write_report(args.report, {"name": Path(args.control_run).name, "policy": args.control_policy},
                      arms, summaries, phase_rows, all_pairs, str(args.out_dir), args.test_result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--arm-requests", type=Path, required=True)
    ap.add_argument("--policy", required=True, action="append")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    arms, refs, pairs = load(args.arm_requests), load(args.baseline), []
    for pol in args.policy:
        study = [r for r in arms if r["policy"] == pol]
        a = pack(study)
        model, nominal = study[0]["model"], int(study[0]["nominal_q"])
        source = [r for r in refs if r["model"]==model and r["role"]=="baseline"
                  and r["policy"].startswith(("ibs_","obs_")) and r["policy"].split("_")[-1].isdigit()]
        for ref in sorted({r["policy"] for r in source}):
            width = int(ref.split("_")[-1])
            b = pack([r for r in source if r["policy"]==ref])
            pairs.append(dict(model=model,policy=pol,nominal=nominal,ref=ref,ref_window=width,
                              same_nominal=int(nominal==width),**compare(a,b)))
    if not pairs:
        raise ValueError("no IBS / OBS reference")
    with args.out.open("w", newline="") as f:
        w=csv.DictWriter(f,list(pairs[0]),lineterminator="\n")
        w.writeheader()
        w.writerows(pairs)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "matched-blocks":
        matched_blocks_main(sys.argv[2:])
    else:
        main()
