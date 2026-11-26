from pathlib import Path
import tempfile
import unittest

from tools.prefix_native import (
    copy_verified, split_added, verify_gemma, verify_llada,
)


def task(index=0, seed=17000):
    return dict(index=index, task_id=f"task/{index}", dataset="gsm8k", seed=seed)


def llada_reference():
    calls = [
        dict(batch_call_id=0, rows=32, physical_rows=32, batch=1, kv_rows=32),
        dict(batch_call_id=1, rows=32, physical_rows=32, batch=1, kv_rows=64),
    ]
    events = [
        dict(phase="prefill", q_tokens=32),
        dict(phase="denoise", q_tokens=32),
    ]
    return dict(
        status="success", model="llada21", policy="native32",
        task_id="task/0", index=0, dataset="gsm8k", seed=17000,
        phase="formal", generation_cap=2048, prompt_ids=[1, 2],
        output_ids=[10, 11, 12], output_text="abc", events=events, calls=calls,
        extra={"native_serial": True}, output_tokens=3, generation_forwards=1,
        generation_query_rows=32, generation_physical_padded_rows=32,
        prefill_rows=32, max_q=32, q_histogram={"32": 1},
        tpf=3.0, token_efficiency=3 / 32,
        physical_token_efficiency=3 / 32, eos=False, truncated=True,
        remaining_masks=0, e2e_shared_gpu_wall_ms=1.0, t2t_edits=0,
        request_failure=None,
    )


def gemma_reference():
    return dict(
        index=0, dataset="gsm8k", task_id="task/0", seed=17000,
        output_tokens=3, generation_forwards=2, generation_query_rows=512,
        generation_physical_padded_rows=512, tpf=1.5,
        token_efficiency=3 / 512, max_q=256, prefill_rows=48,
        decoder_forwards=2, decoder_query_rows=512, reencode_forwards=0,
        reencode_query_rows=0, decoder_steps_per_canvas=[2],
        native_nonpad=3, native_tpf=1.5, eos=False, truncated=True,
        model="diffusiongemma", policy="native256", status="success",
        output_text="abc", output_ids=[10, 11, 12],
        events=[["encoder", 48], ["decoder", 256], ["decoder", 256]],
    )


class PrefixNativeTest(unittest.TestCase):
    def test_llada_accepts_native_reference(self):
        self.assertEqual(verify_llada(task(), llada_reference())["policy"], "native32")

    def test_llada_rejects_seed_change(self):
        ref = llada_reference()
        ref["seed"] += 1
        with self.assertRaises(ValueError):
            verify_llada(task(), ref)

    def test_llada_rejects_wrong_query_histogram(self):
        ref = llada_reference()
        ref["q_histogram"] = {"16": 2}
        with self.assertRaises(ValueError):
            verify_llada(task(), ref)

    def test_llada_rejects_sparse_policy(self):
        ref = llada_reference()
        ref["policy"] = "t2t_q32_f0_m32_targeted"
        with self.assertRaises(ValueError):
            verify_llada(task(), ref)

    def test_llada_rejects_wrong_forward_count(self):
        ref = llada_reference()
        ref["generation_forwards"] = 2
        with self.assertRaises(ValueError):
            verify_llada(task(), ref)

    def test_llada_rejects_output_token_mismatch(self):
        ref = llada_reference()
        ref["output_tokens"] = 4
        with self.assertRaises(ValueError):
            verify_llada(task(), ref, special={99})

    def test_gemma_accepts_native_reference(self):
        self.assertEqual(verify_gemma(task(), gemma_reference())["policy"], "native256")

    def test_gemma_rejects_wrong_events(self):
        ref = gemma_reference()
        ref["events"] = [["encoder", 48], ["decoder", 256]]
        with self.assertRaises(ValueError):
            verify_gemma(task(), ref)

    def test_gemma_rejects_bad_step_sum(self):
        ref = gemma_reference()
        ref["decoder_steps_per_canvas"] = [1]
        with self.assertRaises(ValueError):
            verify_gemma(task(), ref)

    def test_split_added_preserves_30_89(self):
        tasks = [dict(index=i) for i in range(90)]
        self.assertEqual([t["index"] for t in split_added(tasks)], list(range(30, 90)))

    def test_copy_verified_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source.json"
            destination = Path(root) / "destination.json"
            source.write_bytes(b'{"a": 1}\n')
            digest = copy_verified(source, destination)
            self.assertEqual(source.read_bytes(), destination.read_bytes())
            self.assertEqual(len(digest), 64)
            with self.assertRaises(FileExistsError):
                copy_verified(source, destination)


if __name__ == "__main__":
    unittest.main()
