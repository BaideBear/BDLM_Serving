"""CPU-only tests for same-task repeat seeds and native reference acquisition."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bdlm_serving.evaluation import frozen
from bdlm_serving.runtime import gemma_native_reference


def _samples(tmp_path):
    tasks = [
        {"index": 0, "source_index": 167, "dataset": "gsm8k",
         "task_id": "gsm8k/167", "seed": 17000, "prompt": "prompt-0",
         "reference": "answer-0", "source": {"question": "q0"},
         "prompt_sha256": "sha-0"},
        {"index": 1, "source_index": 42, "dataset": "math500",
         "task_id": "math500/42", "seed": 17100, "prompt": "prompt-1",
         "reference": "answer-1", "source": {"problem": "q1"},
         "prompt_sha256": "sha-1"},
    ]
    payload = {"selection_seed": 20261124, "samples": tasks,
               "source_sha256": {}, "scorer_version": "test"}
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps(payload))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples_sha256": frozen.sha(samples)}))
    return samples, manifest


def test_repeat_seed_offset_is_predeclared():
    assert frozen.REPEAT_SEED_OFFSET == 1_000_003


def test_load_tasks_default_preserves_archived_seeds(tmp_path):
    samples, manifest = _samples(tmp_path)
    before = samples.read_bytes()
    with patch.object(frozen, "MANIFEST", manifest):
        tasks = frozen.load_tasks(samples)
    assert [task["seed"] for task in tasks] == [17000, 17100]
    assert samples.read_bytes() == before


def test_load_tasks_offset_changes_only_seed_and_is_deterministic(tmp_path):
    samples, manifest = _samples(tmp_path)
    with patch.object(frozen, "MANIFEST", manifest):
        base = frozen.load_tasks(samples)
        first = frozen.load_tasks(samples, seed_offset=frozen.REPEAT_SEED_OFFSET)
        second = frozen.load_tasks(samples, seed_offset=frozen.REPEAT_SEED_OFFSET)
    assert first == second
    assert len(first) == len(base) == 2
    for old, new in zip(base, first):
        assert new["seed"] == old["seed"] + 1_000_003
        assert {k: v for k, v in new.items() if k != "seed"} == {
            k: v for k, v in old.items() if k != "seed"}


@pytest.mark.parametrize("bad", [True, 1.5, "3", None])
def test_load_tasks_rejects_non_integer_offsets(tmp_path, bad):
    with pytest.raises(TypeError):
        frozen.load_tasks(tmp_path / "missing.json", seed_offset=bad)


def test_load_tasks_rejects_negative_offset(tmp_path):
    with pytest.raises(ValueError):
        frozen.load_tasks(tmp_path / "missing.json", seed_offset=-1)


def test_acquire_native_reference_uses_fixed_offset_and_prepare_hook(tmp_path):
    seen = {}
    sentinel_rows = [{"policy": "native256", "index": 0, "seed": 1017003}]

    def prepare(model):
        return model

    def fake_run_native(args, before_cuda=None, trace_request=None, seed_offset=0):
        seen.update(args=args, before_cuda=before_cuda,
                    trace_request=trace_request, seed_offset=seed_offset)
        return sentinel_rows

    with patch.object(gemma_native_reference, "run_native", fake_run_native):
        rows = gemma_native_reference.acquire_native_reference(
            tmp_path / "checkpoint", tmp_path / "samples.json", tmp_path / "out",
            prepare=prepare)

    assert rows == sentinel_rows
    assert seen["seed_offset"] == frozen.REPEAT_SEED_OFFSET == 1_000_003
    assert seen["before_cuda"] is prepare
    assert seen["trace_request"] is None
    assert isinstance(seen["args"], SimpleNamespace)
    assert seen["args"].checkpoint == Path(tmp_path / "checkpoint")
    assert seen["args"].samples == Path(tmp_path / "samples.json")
    assert seen["args"].output == Path(tmp_path / "out")


def test_acquire_native_reference_cli_forwards_paths(tmp_path):
    seen = {}

    def fake_acquire(checkpoint, samples, output):
        seen.update(checkpoint=checkpoint, samples=samples, output=output)
        return [{"index": 0}]

    with patch.object(gemma_native_reference, "acquire_native_reference", fake_acquire):
        rows = gemma_native_reference.main([
            "--checkpoint", str(tmp_path / "checkpoint"),
            "--samples", str(tmp_path / "samples.json"),
            "--output", str(tmp_path / "out"),
        ])

    assert rows == [{"index": 0}]
    assert seen == {"checkpoint": tmp_path / "checkpoint",
                    "samples": tmp_path / "samples.json",
                    "output": tmp_path / "out"}
