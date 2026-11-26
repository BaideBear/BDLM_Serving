import json
from types import SimpleNamespace

import pytest

from bdlm_serving.evaluation import frozen90
from bdlm_serving.methods.t2t import online
from tools import run_t2t


def test_frozen90_dispatch_counts_provenance_and_q64_prefix_alias(tmp_path, monkeypatch):
    samples = tmp_path / "samples.json"
    samples.write_text("fixture")
    args = SimpleNamespace(model="llada21", suite="frozen90", samples=samples,
        output=tmp_path / "out", reference=tmp_path,
        arm=["64:0:62:hard_prefix_frontier_right"])
    tasks = [dict(index=i, dataset="sample", task_id=str(i), seed=i) for i in range(2)]
    dispatched = []

    def load(path):
        dispatched.append(path)
        return tasks

    def run(task, arm):
        return dict(task, status="ok", output_ids=[0 if arm is None else 1], output_tokens=1,
                    generation_forwards=1, generation_query_rows=1, generation_physical_padded_rows=1,
                    max_q=1, tpf=1, token_efficiency=1)

    def load_backend(*a, **kw):
        def call(task, arm):
            if arm is not None:
                assert (arm.budget, arm.edit, arm.selection) == (32, 2, "hard_prefix_frontier_right")
            return run(task, arm)
        return call

    for task in tasks:
        (tmp_path / f"{task['index']:02d}-native32.json").write_text(json.dumps(run(task, None)))
    monkeypatch.setattr(frozen90, "load_tasks", load)
    monkeypatch.setattr(online, "load_backend", load_backend)
    run_t2t.main(args)
    assert dispatched == [samples]
    policy = "t2t_q64_f0_m62_hard_prefix_frontier_right"
    run = json.loads((args.output / "run.json").read_text())
    assert run["samples_sha256"] == run_t2t.sha(samples) and run["suite"] == "frozen90"
    assert json.loads((args.output / "qualified.json").read_text())["matched"] == len(tasks)
    assert json.loads((args.output / f"complete-{policy}.json").read_text())["requests"] == len(tasks)
    row = json.loads((args.output / f"01-{policy}.json").read_text())
    assert row["single_block_prefix_capacity"] == dict(nominal_q=64, executable_width=32, quota=2)
    assert "single_block_saturation" not in row


def test_frozen90_rejects_seed_offset_before_reading_samples():
    with pytest.raises(ValueError, match="requires --seed-offset 0"):
        run_t2t.main(SimpleNamespace(model="gemma", suite="frozen90", seed_offset=1))


def test_frozen30_without_args_suite_preserves_seed_offset(tmp_path, monkeypatch):
    samples = tmp_path / "samples.json"
    samples.write_text("fixture")
    args = SimpleNamespace(model="gemma", samples=samples, arm=None, seed_offset=13)

    def load(path, *, seed_offset):
        assert (path, seed_offset) == (samples, 13)
        return ["mock task"]

    def run(args, prepare, tasks, arms, extra, block, offset, ranker, admission_fit):
        assert tasks == ["mock task"] and block == 256 and offset == 13
        assert extra == dict(suite="frozen30", samples_sha256=run_t2t.sha(samples))

    monkeypatch.setattr(run_t2t, "load_tasks", load)
    monkeypatch.setattr(run_t2t, "_run_matrix", run)
    run_t2t.main(args)
