"""CPU contracts for the formal TopTransition frozen90 external-offer runner."""
import csv
import json
from types import SimpleNamespace

import pytest
import torch

from bdlm_serving.methods.t2t.online import package
from bdlm_serving.methods.t2t.protocol import verify_work
from experiments.t2t.serving_slots import top_transition90 as tt
from experiments.t2t.serving_slots.llada_top_transition import TopTransitionKernel
from tests.test_llada_top_transition import CPUModel, create
from tools.run_dual_t2t import saturation_summary


class Tok:
    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


def task(index=0, dataset="gsm8k", seed=11):
    return dict(index=index, dataset=dataset, task_id=f"task-{index}", seed=seed)


def toy_row(t, trace, variant, gen_length=96):
    model = CPUModel().eval()
    kernel = TopTransitionKernel(model, fuse_growth=(variant == "fused"))
    state = create(kernel, length=gen_length)
    kernel.run(state, tt.offer(trace))
    output_ids = kernel.output(state)[0].tolist()
    work = [(event["stage"], event["physical_q"]) for event in state.events
            if event["phase"] != "prefill"]
    prefill = sum(event["physical_q"] for event in state.events
                  if event["phase"] == "prefill")
    row = package(Tok(), output_ids, work, (), set(), prefill,
                  events=state.events,
                  t2t_edits=sum(int(event.get("edited", 0)) for event in state.events))
    row.update({key: t[key] for key in ("index", "dataset", "task_id", "seed")},
               model="llada21", policy=tt.policy_name(trace, variant))
    offers = sorted(set(trace))
    if len(offers) == 1:
        row["nominal_q"] = offers[0]
    row["correct"] = True
    row["top_transition"] = tt.top_metadata(state.events, trace, variant)
    row["dual"] = dict(max_offer=tt.PAIR, pair_blocks=2, gen_length=gen_length,
                       mode="rolling_pair", external_trace=list(trace))
    row["dual_saturation"] = saturation_summary(state.events)
    verify_work(row)
    return row


def test_offer_clock_and_policy_names():
    assert tt.policy_name((32, 32, 64, 64), "fused") == "toptrace_32_32_64_64_fused"
    clock = tt.offer((32, 32, 64, 64))
    assert [clock(SimpleNamespace(tick=tick, block=1, phase="denoise")) for tick in range(9)] == [
        32, 32, 64, 64, 32, 32, 64, 64, 32]
    assert tt.resolve_traces(None) == ((32, 32, 64, 64), (64, 64, 32, 32))
    assert tt.resolve_traces(["fixed32", "32,64,32,32,64"]) == (
        (32,), (32, 64, 32, 32, 64))
    assert tt.resolve_traces(["fixed32", "fixed64"]) == ((32,), (64,))
    with pytest.raises(ValueError, match="32/64"):
        tt.resolve_traces(["16"])
    with pytest.raises(ValueError, match="Duplicate"):
        tt.resolve_traces(["llhh", "32,32,64,64"])
    assert tt.resolve_variants(None) == ("fused", "split")
    with pytest.raises(ValueError, match="unique"):
        tt.resolve_variants(["fused", "fused"])


def test_audit_accepts_fused_and_split():
    trace = (32, 64, 32, 32, 64)
    fused = toy_row(task(0), trace, "fused")
    split = toy_row(task(0), trace, "split")
    fused_audit = tt.audit_top(fused, trace, True, policy=fused["policy"])
    split_audit = tt.audit_top(split, trace, False, policy=split["policy"])
    assert fused_audit["top_actions"].get("fused_growth", 0) >= 1
    assert split_audit["top_actions"].get("split_growth", 0) >= 1
    assert fused_audit["growth_requests"] == 1
    assert split_audit["growth_requests"] == 1
    common = min(len(fused["top_transition"]["offers"]),
                 len(split["top_transition"]["offers"]))
    assert (fused["top_transition"]["offers"][:common]
            == split["top_transition"]["offers"][:common])


def test_audit_rejects_tampering():
    trace = (32, 64, 32, 32, 64)
    row = toy_row(task(0), trace, "fused")

    bad = json.loads(json.dumps(row))
    event = next(e for e in bad["events"] if e["phase"] != "prefill")
    event["q_budget"] = 64 if event["q_budget"] == 32 else 32
    with pytest.raises(ValueError, match="q_budget"):
        tt.audit_top(bad, trace, True, policy=bad["policy"])

    bad = json.loads(json.dumps(row))
    bad["generation_forwards"] += 1
    with pytest.raises(ValueError, match="generation_forwards|Work counters"):
        tt.audit_top(bad, trace, True, policy=bad["policy"])

    bad = json.loads(json.dumps(row))
    event = next(e for e in bad["events"] if e.get("top_action") == "fused_growth")
    event["refresh_only_positions"] = [event["absolute_positions"][0] - 1]
    with pytest.raises(ValueError, match="refresh_only"):
        tt.audit_top(bad, trace, True, policy=bad["policy"])

    bad = json.loads(json.dumps(row))
    event = next(e for e in bad["events"] if e.get("top_action") == "fused_growth")
    event["top_action"] = "split_growth"
    with pytest.raises(ValueError, match="split_growth in a fused arm"):
        tt.audit_top(bad, trace, True, policy=bad["policy"])


def test_fixed32_fused_and_split_are_identical():
    trace = (32,)
    fused = toy_row(task(0), trace, "fused")
    split = toy_row(task(0), trace, "split")
    assert fused["output_ids"] == split["output_ids"]
    assert ([e.get("top_action") for e in fused["events"]]
            == [e.get("top_action") for e in split["events"]])
    assert tt.audit_top(fused, trace, True, policy=fused["policy"])["growth_requests"] == 0
    assert tt.audit_top(split, trace, False, policy=split["policy"])["growth_requests"] == 0


def test_nominal_q_is_not_faked_for_dynamic_traces():
    fixed32 = toy_row(task(0), (32,), "fused")
    fixed64 = toy_row(task(0), (64,), "fused")
    mixed = toy_row(task(0), (32, 32, 64, 64), "fused")
    assert fixed32["nominal_q"] == 32
    assert fixed64["nominal_q"] == 64
    assert "nominal_q" not in mixed
    assert fixed32["dual"]["max_offer"] == 64
    assert fixed32["dual"]["external_trace"] == [32]
    assert mixed["dual"]["external_trace"] == [32, 32, 64, 64]


def test_top_offer_traces_environment(monkeypatch):
    monkeypatch.setenv("TOP_OFFER_TRACES", "fixed32 fixed64")
    assert tt.resolve_args_traces(SimpleNamespace(trace=None)) == ((32,), (64,))
    assert tt.resolve_args_traces(SimpleNamespace(trace=["llhh"])) == ((32, 32, 64, 64),)
    monkeypatch.setenv("TOP_OFFER_TRACES", "")
    assert tt.resolve_args_traces(SimpleNamespace(trace=None)) == (
        (32, 32, 64, 64), (64, 64, 32, 32))


def test_summary_metrics():
    rows = [dict(output_tokens=10, generation_forwards=2, generation_query_rows=20,
                 generation_physical_padded_rows=20, max_q=10, correct=True),
            dict(output_tokens=6, generation_forwards=3, generation_query_rows=30,
                 generation_physical_padded_rows=30, max_q=10, correct=False)]
    summary = tt.summarize(rows)
    assert summary["requests"] == 2
    assert summary["correct"] == 1
    assert summary["macro_tpf"] == pytest.approx((10 / 2 + 6 / 3) / 2)
    assert summary["token_efficiency"] == pytest.approx(16 / 50)
    assert summary["N"] == 16 and summary["F"] == 5 and summary["Q"] == 50


def execute_90(tmp_path, *, device="cuda", gen_length=2048, limit=None):
    trace = (32, 32, 64, 64)
    tasks = [task(i, seed=100 + i) for i in range(90)]
    base = {variant: toy_row(tasks[0], trace, variant) for variant in ("fused", "split")}

    def run_one(t, *, trace, variant, policy, gen_length):
        row = json.loads(json.dumps(base[variant]))
        row.update({key: t[key] for key in ("index", "dataset", "task_id", "seed")})
        row["policy"] = policy
        return row

    args = SimpleNamespace(output=tmp_path / "run", samples=tmp_path / "samples.json",
                           limit=limit, trace=["32,32,64,64"], variant=None,
                           gen_length=gen_length, device=device,
                           checkpoint=tmp_path / "model", samples_sha256=tt.SAMPLES_SHA)
    (tmp_path / "samples.json").write_text("{}")
    tt.execute(args, tasks, run_one)
    return args.output


def test_execute_writes_qualified_report(tmp_path):
    out = execute_90(tmp_path)
    marker = tt.read(out / "qualified.json")
    assert marker["trace_contract_checked"] is True
    assert marker["completed_requests"] == 90
    assert marker["gen_length"] == 2048
    run = tt.read(out / "run.json")
    assert run["formal"] is True
    assert run["max_offer"] == 64
    assert "nominal_q" not in run
    assert run["arms"][0]["offers"] == [32, 64]
    row = tt.read(out / "00-toptrace_32_32_64_64_fused.json")
    assert "nominal_q" not in row
    assert row["dual"]["max_offer"] == 64
    assert row["dual"]["external_trace"] == [32, 32, 64, 64]
    assert len(list(out.glob("complete-*.json"))) == 2
    assert len(list(out.glob("[0-9][0-9]-toptrace_*.json"))) == 180
    audit = tt.read(out / "top-transition-audit.json")
    assert audit["requests"] == 90
    assert all(arm["correct"] == 90 for arm in audit["arms"].values())
    rows = list(csv.DictReader((out / "top-transition-summary.csv").open()))
    assert len(rows) == 2
    assert all(int(row["correct"]) == 90 for row in rows)
    pairs = list(csv.DictReader((out / "top-transition-pairs.csv").open()))
    assert len(pairs) == 90
    assert all(row["same_offers_prefix"] == "1" for row in pairs)

    rescored = out / "rescore"
    tt.score(SimpleNamespace(run=out, out=rescored))
    assert (rescored / "top-transition-summary.csv").is_file()
    assert (rescored / "top-transition-pairs.csv").is_file()
    assert tt.read(rescored / "top-transition-audit.json")["requests"] == 90


def test_execute_smoke_does_not_write_qualified(tmp_path):
    trace = (32, 64, 32, 32, 64)
    tasks = [task(0), task(1)]
    base = toy_row(tasks[0], trace, "fused")

    def run_one(t, *, trace, variant, policy, gen_length):
        row = json.loads(json.dumps(base))
        row.update({key: t[key] for key in ("index", "dataset", "task_id", "seed")})
        row["policy"] = policy
        return row

    args = SimpleNamespace(output=tmp_path / "run", samples=tmp_path / "samples.json",
                           limit=2, trace=["wave"], variant=["fused"],
                           gen_length=96, device="cuda", checkpoint=tmp_path / "model",
                           samples_sha256="fake-sha")
    (tmp_path / "samples.json").write_text("{}")
    tt.execute(args, tasks, run_one)
    out = args.output
    marker = tt.read(out / "smoke.json")
    assert marker["trace_contract_checked"] is True
    assert marker["completed_requests"] == 2
    assert marker["reason"] == "generation_cap"
    assert not (out / "qualified.json").exists()
    assert not list(out.glob("complete-*.json"))
    assert tt.read(out / "run.json")["formal"] is False
    assert tt.read(out / "top-transition-audit.json")["requests"] == 2


def test_execute_cpu_never_writes_qualified(tmp_path):
    out = execute_90(tmp_path, device="cpu")
    marker = tt.read(out / "smoke.json")
    assert marker["trace_contract_checked"] is True
    assert marker["completed_requests"] == 90
    assert marker["reason"] == "device"
    assert not (out / "qualified.json").exists()
    assert not list(out.glob("complete-*.json"))
    assert tt.read(out / "run.json")["formal"] is False


def test_execute_non90_never_writes_qualified(tmp_path):
    out = execute_90(tmp_path, limit=2)
    marker = tt.read(out / "smoke.json")
    assert marker["trace_contract_checked"] is True
    assert marker["completed_requests"] == 2
    assert marker["reason"] == "requests"
    assert not (out / "qualified.json").exists()
    assert not list(out.glob("complete-*.json"))
    assert tt.read(out / "run.json")["formal"] is False
