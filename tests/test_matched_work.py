"""CPU-only tests for the matched-block diagnostic in tools/work_envelope.py."""
import csv
import gzip
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tools.work_envelope import BLOCK, STAGES, TraceError, _compare, _summary, _validate_trace, matched_blocks_main


def make_trace(index=0, prompt_length=58, output_ids=None, blocks=None, eos=True,
               truncated=False, policy="test", seed=7):
    prefill = BLOCK * (prompt_length // BLOCK)
    if output_ids is None:
        output_ids = list(range(1000, 1000 + BLOCK * len(blocks)))
    if blocks is None:
        blocks = [(1, 1)] * ((len(output_ids) + BLOCK - 1) // BLOCK)
    events = [dict(phase="prefill", block=prefill // BLOCK, q_start=0, q_end=prefill,
                   q_tokens=prefill, prefix_kv_length=0, store_kv=True, context_version=0,
                   stage="prefill", absolute_positions=list(range(prefill)), actual_q=prefill,
                   physical_q=prefill, layer_physical_q=[prefill] * 4, F=1)]
    for ordinal, (selected, verify) in enumerate(blocks, 1):
        block_no = prefill // BLOCK + ordinal - 1
        q_start = prefill + BLOCK * (ordinal - 1)
        q_end = q_start + BLOCK
        events.append(dict(phase="denoise", block=block_no, q_start=q_start, q_end=q_end,
                           q_tokens=BLOCK, prefix_kv_length=q_start, store_kv=False,
                           context_version=0, stage="bootstrap",
                           absolute_positions=list(range(q_start, q_end)), actual_q=BLOCK,
                           physical_q=BLOCK, layer_physical_q=[BLOCK] * 4, F=1))
        for _ in range(selected):
            events.append(dict(phase="denoise", block=block_no, q_start=q_start,
                               q_end=q_start + 16, q_tokens=16, prefix_kv_length=q_start,
                               store_kv=False, context_version=0, stage="selected",
                               absolute_positions=list(range(q_start, q_start + 16)),
                               actual_q=16, physical_q=16, layer_physical_q=[16] * 4, F=1))
        for _ in range(verify):
            events.append(dict(phase="denoise", block=block_no, q_start=q_start, q_end=q_end,
                               q_tokens=BLOCK, prefix_kv_length=q_start, store_kv=False,
                               context_version=0, stage="verify",
                               absolute_positions=list(range(q_start, q_end)), actual_q=BLOCK,
                               physical_q=BLOCK, layer_physical_q=[BLOCK] * 4, F=1))
        events.append(dict(phase="commit", block=block_no, q_start=q_start, q_end=q_end,
                           q_tokens=BLOCK, prefix_kv_length=q_start, store_kv=True,
                           context_version=0, stage="commit",
                           absolute_positions=list(range(q_start, q_end)), actual_q=BLOCK,
                           physical_q=BLOCK, layer_physical_q=[BLOCK] * 4, F=1))
    generation = [e for e in events if e["phase"] != "prefill"]
    forwards = len(generation)
    queries = sum(e["physical_q"] for e in generation)
    stages = Counter(e["stage"] for e in generation)
    qph = Counter()
    for event in generation:
        qph[event["stage"]] += event["physical_q"]
    n = len(output_ids) - 1 if eos else len(output_ids)
    return dict(status="success", model="llada21", policy=policy,
                task_id="task-%02d" % index, index=index, dataset="gsm8k", seed=seed,
                output_ids=list(output_ids), output_text="text-%02d" % index,
                output_tokens=n, generation_forwards=forwards,
                generation_query_rows=queries, generation_physical_padded_rows=queries,
                prefill_rows=prefill, max_q=BLOCK,
                q_histogram=dict(Counter(str(e["q_tokens"]) for e in generation)),
                tpf=n / forwards if forwards else 0.0, token_efficiency=0.0,
                eos=eos, truncated=truncated, forwards_by_phase=dict(stages),
                q_by_phase=dict(qph), events=events, t2t_edits=0)


class MatchedBlockTest(unittest.TestCase):
    def test_valid_blocks_exclude_prompt_mixed_and_terminal(self):
        trace = make_trace(prompt_length=58, output_ids=list(range(100, 392)),
                           blocks=[(1, 1)] * 10, eos=False)
        _validate_trace(trace, 58)
        summary = _summary(trace, 58)
        self.assertEqual(len(summary["valid_blocks"]), 8)
        self.assertEqual(summary["valid_blocks"][0]["block_no"], 2)
        self.assertEqual(summary["valid_blocks"][-1]["block_no"], 9)
        self.assertEqual(summary["tail_positions"], 30)
        self.assertEqual(summary["terminal_block_ordinal"], 10)
        self.assertEqual(summary["terminal_block_positions"], 30)

    def test_prompt_aligned_has_no_mixed_block(self):
        trace = make_trace(prompt_length=64, output_ids=list(range(100, 164)),
                           blocks=[(1, 1)] * 2, eos=False)
        _validate_trace(trace, 64)
        summary = _summary(trace, 64)
        self.assertEqual([b["block_no"] for b in summary["valid_blocks"]], [2, 3])
        self.assertEqual(summary["tail_positions"], 0)
        self.assertEqual(summary["terminal_block_ordinal"], 2)
        self.assertEqual(summary["terminal_block_positions"], 32)

    def test_prefill_not_floor_is_rejected(self):
        trace = make_trace(prompt_length=58, output_ids=list(range(100, 164)),
                           blocks=[(1, 1)] * 2)
        trace["prefill_rows"] = 64
        with self.assertRaisesRegex(TraceError, "floor prompt block"):
            _validate_trace(trace, 58)

    def test_pair_decomposition_and_text_divergence(self):
        control = _summary(make_trace(prompt_length=58, output_ids=list(range(100, 196)),
                                      blocks=[(1, 1), (2, 1), (1, 2), (1, 1)], eos=False), 58)
        arm_ids = list(range(100, 202))
        arm_ids[40] = 999
        arm = _summary(make_trace(prompt_length=58, output_ids=arm_ids,
                                  blocks=[(1, 1), (3, 1), (2, 2), (2, 1)], eos=False), 58)
        row, blocks = _compare(control, arm, "arm")
        self.assertEqual(row["matched_blocks"], 2)
        self.assertEqual(row["matched_F"], 2)
        self.assertEqual(row["remainder_F"], 1)
        self.assertEqual(row["excess_F"], 3)
        self.assertEqual(row["classification"], "both")
        self.assertEqual(row["first_divergent_block"], 2)
        self.assertEqual(row["same_matched_prefix"], 0)
        self.assertEqual(row["same_full_output"], 0)
        self.assertEqual(row["matched_same_token_blocks"], 1)
        self.assertEqual(blocks[0]["delta_selected"], 1)
        self.assertEqual(blocks[1]["same_block_tokens"], 0)

    def test_same_text_can_have_different_block_work(self):
        ids = list(range(100, 196))
        control = _summary(make_trace(prompt_length=58, output_ids=ids,
                                      blocks=[(1, 1), (2, 1), (1, 2), (1, 1)], eos=False), 58)
        arm = _summary(make_trace(prompt_length=58, output_ids=ids,
                                  blocks=[(1, 1), (1, 1), (2, 2), (1, 1)], eos=False), 58)
        row, blocks = _compare(control, arm, "arm")
        self.assertEqual(row["excess_F"], 0)
        self.assertEqual(row["matched_F"], 0)
        self.assertEqual(row["remainder_F"], 0)
        self.assertEqual(row["classification"], "equal")
        self.assertEqual(row["same_full_output"], 1)
        self.assertEqual(row["same_matched_prefix"], 1)
        self.assertEqual(blocks[0]["delta_selected"], -1)
        self.assertEqual(blocks[1]["delta_selected"], 1)

    def test_cli_writes_csv_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_dir = root / "native"
            prompt_dir.mkdir()
            control_dir = root / "control"
            control_dir.mkdir()
            labels = ["fixed", "gate", "geometry", "slack_geometry"]
            arm_dirs = {label: root / label for label in labels}
            for directory in arm_dirs.values():
                directory.mkdir()
            for i in range(30):
                with gzip.open(prompt_dir / ("%02d.trace.jsonl.gz" % i), "wt") as f:
                    f.write(json.dumps({"schema": 1, "index": i, "task_id": "task-%02d" % i,
                                        "dataset": "gsm8k", "seed": i}) + "\n")
                    f.write(json.dumps({"prompt_length": 58, "mask_id": 156895}) + "\n")
                ids = list(range(1000 + i * 100, 1000 + i * 100 + 96))
                trace = make_trace(index=i, prompt_length=58, output_ids=ids,
                                   blocks=[(1, 1), (1, 1), (1, 1), (1, 1)], seed=i)
                (control_dir / ("%02d-test.json" % i)).write_text(json.dumps(trace), encoding="utf-8")
                for label in labels:
                    trace = make_trace(index=i, prompt_length=58, output_ids=ids,
                                       blocks=[(1, 1), (3, 1), (2, 2), (1, 1)], seed=i)
                    (arm_dirs[label] / ("%02d-test.json" % i)).write_text(json.dumps(trace), encoding="utf-8")
            out_dir = root / "out"
            report = root / "report.md"
            argv = ["--control-run", str(control_dir), "--control-policy", "test",
                    "--prompt-dir", str(prompt_dir)]
            for label in labels:
                argv += ["--arm", label, str(arm_dirs[label]), "test"]
            argv += ["--out-dir", str(out_dir), "--report", str(report), "--test-result", "6 passed"]
            matched_blocks_main(argv)
            with (out_dir / "llada21-matched-block-requests.csv").open(newline="") as f:
                requests = list(csv.DictReader(f))
            with (out_dir / "llada21-matched-block-blocks.csv").open(newline="") as f:
                blocks = list(csv.DictReader(f))
            with (out_dir / "llada21-matched-block-phases.csv").open(newline="") as f:
                phases = list(csv.DictReader(f))
            self.assertEqual(len(requests), 120)
            self.assertEqual(len(blocks), 240)
            self.assertEqual(len(phases), 16)
            self.assertEqual({r["arm"] for r in requests}, set(labels))
            self.assertTrue(all(int(r["matched_delta"]) == 0 for r in phases
                                if r["phase"] in ("bootstrap", "commit")))
            text = report.read_text(encoding="utf-8")
            self.assertIn("matched-block", text)
            self.assertNotIn("/data1/", text)


if __name__ == "__main__":
    unittest.main()
