import hashlib
import json
from pathlib import Path
import types

import pytest

from experiments.t2t.serving_slots import gemma_fixed_caps as caps


def task(index=0):
    return dict(index=index, task_id=f"task-{index}", seed=index, dataset="gsm8k")


def make_trace(policy, *, binding=False):
    cap = caps.cap_of(policy)
    trace = []
    for chunk in range(256 // cap):
        base = chunk * cap
        positions = list(range(base, base + cap))
        trace.append(dict(block=0, step=len(trace), q=cap, accepted=cap, ready=base,
                          cross=0, visible_end=base + cap, tick=len(trace), offer=cap,
                          proposed_q=cap + (1 if binding and chunk == 0 else 0),
                          positions=positions, admitted=positions))
        trace.append(dict(block=0, step=len(trace), q=cap, accepted=0,
                          ready=base + cap, cross=0, visible_end=base + cap,
                          tick=len(trace), offer=cap, proposed_q=cap,
                          positions=positions, admitted=[]))
    return trace


def make_row(policy, case, *, binding=False):
    trace = make_trace(policy, binding=binding)
    steps = len(trace)
    cap = caps.cap_of(policy)
    return dict(index=case["index"], task_id=case["task_id"], seed=case["seed"],
                dataset=case["dataset"], model="gemma", policy=policy,
                status="success", output_ids=[1, 2, 3], output_text="x",
                output_tokens=3,
                events=[["encoder", 48]] + [["decoder", cap]] * steps,
                trace=trace, generation_forwards=steps,
                generation_query_rows=cap * steps,
                generation_physical_padded_rows=cap * steps,
                max_q=cap, max_decoder_q=cap, tpf=3 / steps,
                token_efficiency=3 / (cap * steps), eos=False, truncated=True)


def uncapped_row(policy, case, *, binding=False):
    row = make_row(policy, case, binding=binding)
    cap = caps.cap_of(policy)
    row["policy"] = f"{policy.split('_')[0]}_{cap}"
    row["trace"] = [{key: value for key, value in step.items()
                     if key not in ("tick", "offer", "proposed_q", "positions",
                                    "admitted")} for step in row["trace"]]
    if binding:
        row["trace"][0].update(q=cap + 1, accepted=cap + 1)
    return row


def put(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def synthetic_run(tmp_path):
    data = tmp_path / "prefix90"
    folder = data / caps.OUTPUT
    tasks = [task(0), task(1)]
    for policy in caps.POLICIES:
        uncapped, arm = caps.UNCAPPED[policy]
        for case in tasks:
            index = case["index"]
            put(folder / f"{index:02d}-{policy}.json", make_row(policy, case))
            put(data / uncapped / f"{index:02d}-{arm}.json",
                uncapped_row(policy, case))
        put(folder / f"complete-{policy}.json",
            dict(policy=policy, requests=len(tasks)))
    put(folder / "run.json", dict(
        model="gemma", suite="frozen90", samples_sha256=caps.SAMPLES_SHA,
        hard_q_cap=True, cap_clock="all_decoder_forwards_request_local",
        cap_excludes="encoder", obs_extent="B+max(schedule)",
        actual_token_edits="M2T only", capped_plans=[[32], [64]],
        policies=list(caps.POLICIES)))
    put(folder / "qualified-logits.json", dict(
        modes=["ibs", "obs"], chunk=256, first_step_logits="bit_identical",
        index=0))
    return data, folder, tasks


def test_fixed_matrix_and_semantics():
    assert caps.PLANS == ("32", "64")
    assert caps.POLICIES == ("ibs_cap_32", "obs_cap_32", "ibs_cap_64",
                             "obs_cap_64")
    assert [caps.cap_of(policy) for policy in caps.POLICIES] == [32, 32, 64, 64]
    assert caps.OUTPUT == "gemma-fixed-caps-q32q64-v1"
    assert caps.MODE == "optimus"


def test_protected_source_replaces_only_the_optimus_import(tmp_path, monkeypatch):
    source = (b"x = 1\nif MODE == \"optimus\":\n    " + caps.IMPORT +
              b"\n    " + caps.CALL + b"\n")
    launcher = tmp_path / "launch_prefix90.py"
    launcher.write_bytes(source)
    with pytest.raises(ValueError, match="reviewed"):
        caps.protected_source(launcher)
    monkeypatch.setattr(caps, "LAUNCHER_SHA", hashlib.sha256(source).hexdigest())
    calls = []
    monkeypatch.setattr(caps, "run", lambda args, before_cuda=None: calls.append(args))
    args = object()
    exec(caps.protected_source(launcher),
         dict(args=args, prepare=None, MODE="optimus"))
    assert calls == [args]


def test_protected_source_rejects_a_changed_call_site(tmp_path, monkeypatch):
    source = b"from bdlm_serving.runtime.diffusiongemma_optimus import run\n" \
             b"run(args, before_cuda=prepare)\nrun(args, before_cuda=prepare)\n"
    launcher = tmp_path / "launch_prefix90.py"
    launcher.write_bytes(source)
    monkeypatch.setattr(caps, "LAUNCHER_SHA", hashlib.sha256(source).hexdigest())
    with pytest.raises(ValueError, match="call site"):
        caps.protected_source(launcher)


def test_run_adapter_registers_the_two_budgets(tmp_path, monkeypatch):
    seen = {}

    def engine_run(args, before_cuda=None):
        seen["args"] = args
        seen["before_cuda"] = before_cuda

    monkeypatch.setattr("bdlm_serving.runtime.diffusiongemma_optimus.run",
                        engine_run)
    monkeypatch.setattr(caps, "verify_run", lambda *args: None)
    monkeypatch.setattr(caps, "sha", lambda path: caps.SAMPLES_SHA)
    (tmp_path / "samples.json").write_text(json.dumps(dict(samples=[])))
    args = types.SimpleNamespace(
        suite="frozen90", samples=tmp_path / "samples.json",
        output=tmp_path / "out", modes=["ibs", "obs"], chunks=[32, 128],
        reference30_root=None)
    caps.run(args, 123)
    assert args.capped_plans == ["32", "64"]
    assert args.uncapped90 == caps.DATA
    assert args.chunks == [32, 64]
    assert seen["before_cuda"] == 123


def test_audit_accepts_nonbinding_and_binding_rows():
    case = task()
    row = make_row("ibs_cap_32", case)
    caps.audit_row(row, uncapped_row("ibs_cap_32", case), 32)
    binding = make_row("ibs_cap_64", case, binding=True)
    caps.audit_row(binding, uncapped_row("ibs_cap_64", case, binding=True), 64)


def test_audit_rejects_a_decoder_over_the_cap():
    case = task()
    row = make_row("ibs_cap_32", case)
    row["trace"][0].update(q=33, positions=list(range(33)),
                           admitted=list(range(33)), accepted=33)
    row["events"][1] = ["decoder", 33]
    with pytest.raises(ValueError):
        caps.audit_row(row, uncapped_row("ibs_cap_32", case), 32)


def test_audit_rejects_a_nonbinding_output_change():
    case = task()
    row = make_row("ibs_cap_32", case)
    original = uncapped_row("ibs_cap_32", case)
    original["output_ids"] = [9, 9, 9]
    with pytest.raises(ValueError, match="Nonbinding"):
        caps.audit_row(row, original, 32)


def test_audit_rejects_a_billing_mismatch():
    case = task()
    row = make_row("obs_cap_64", case)
    row["generation_query_rows"] += 1
    row["generation_physical_padded_rows"] += 1
    with pytest.raises(ValueError):
        caps.audit_row(row, uncapped_row("obs_cap_64", case), 64)


def test_verify_run_accepts_the_synthetic_matrix(tmp_path):
    data, folder, tasks = synthetic_run(tmp_path)
    meta = caps.verify_run(folder, data, tasks)
    assert meta["capped_plans"] == [[32], [64]]
    assert meta["hard_q_cap"] is True


def test_verify_run_rejects_an_incomplete_arm(tmp_path):
    data, folder, tasks = synthetic_run(tmp_path)
    (folder / "complete-ibs_cap_64.json").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        caps.verify_run(folder, data, tasks)


def test_inspect_uncapped_counts_binding_forwards(tmp_path):
    data = tmp_path / "prefix90"
    tasks = [task(0), task(1)]
    for policy, (folder, arm) in caps.UNCAPPED.items():
        cap = caps.cap_of(policy)
        put(data / folder / f"00-{arm}.json",
            dict(trace=[dict(q=cap), dict(q=cap + 1), dict(q=cap)],
                 max_decoder_q=cap + 1))
        put(data / folder / f"01-{arm}.json",
            dict(trace=[dict(q=cap)], max_decoder_q=cap))
    stats = {row["policy"]: row for row in caps.inspect_uncapped(data, tasks)}
    for policy in caps.POLICIES:
        assert stats[policy]["binding_requests"] == 1
        assert stats[policy]["excess_rows"] == 1
        assert stats[policy]["max_decoder_q"] == caps.cap_of(policy) + 1


def test_verify_predecessor_accepts_a_complete_mask_pair(tmp_path):
    data = tmp_path / "prefix90"
    folder = data / "gemma-fixed512-mask-pair"
    for policy in ("active_w2_mask_on", "active_w2_mask_off"):
        put(folder / f"complete-{policy}.json",
            dict(policy=policy, requests=90))
    put(folder / "qualified.json",
        dict(requests=90, historical_replayed=90))
    put(folder / "pair-audit.json", dict(cases=[dict(index=i) for i in range(90)]))
    caps.verify_predecessor(data, caps.PREDECESSOR)


def test_verify_predecessor_rejects_an_incomplete_mask_pair(tmp_path):
    data = tmp_path / "prefix90"
    with pytest.raises(ValueError, match="Incomplete predecessor"):
        caps.verify_predecessor(data, caps.PREDECESSOR)


def test_verify_predecessor_is_a_noop_for_another_after(tmp_path):
    caps.verify_predecessor(tmp_path / "prefix90", "prefix90-other")


def test_neutrality_counts_binding_and_divergent_requests(tmp_path):
    data, folder, tasks = synthetic_run(tmp_path)
    case = tasks[0]
    binding = make_row("ibs_cap_32", case, binding=True)
    original = uncapped_row("ibs_cap_32", case, binding=True)
    original["output_ids"] = [9, 9, 9]
    put(folder / "00-ibs_cap_32.json", binding)
    put(data / "gemma-optimus-reference90" / "00-ibs_32.json", original)
    stats = {row["policy"]: row for row in caps.neutrality(data, folder, tasks)}
    assert stats["ibs_cap_32"]["binding_requests"] == 1
    assert stats["ibs_cap_32"]["same_output"] == 0
    assert stats["ibs_cap_32"]["same_events"] == 1
