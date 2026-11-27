"""CPU tests for the Q4/8/16/32 offer-mask contrast (no GPU, holder or queue)."""
from copy import deepcopy
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.t2t.serving_slots import llada_offer_masks as om
from test_llada_top_transition import CPUModel


class Attention(nn.Module):
    """A callable attention module so the real-mask witness hook fires."""

    def __init__(self):
        super().__init__()
        self.query_key_value = nn.Identity()

    def forward(self, hidden, *, attention_mask, position_ids):
        return self.query_key_value(hidden)


class PairModel(CPUModel):
    """CPUModel without the block-causal mask assertion: the open arm releases."""

    def __init__(self):
        super().__init__()
        for layer in self.model.layers:
            layer.attention = Attention()

    def forward(self, *, inputs_embeds, position_ids, attention_mask,
                past_key_values, use_cache, output_attentions):
        positions = position_ids[0]
        q, end = positions.numel(), attention_mask.shape[-1]
        assert use_cache and output_attentions
        self.calls.append((positions.tolist(), end))
        hidden = inputs_embeds
        for layer_id, layer in enumerate(self.model.layers):
            kv = layer.attention(hidden, attention_mask=attention_mask.bool(),
                                 position_ids=position_ids).unsqueeze(1)
            key, value = past_key_values.update(kv, kv + 100, layer_id)
            assert key.shape[-2] == value.shape[-2] == end
            if layer_id == 0:
                keys = key[0, 0, :, 0]
                prior = [int(keys[p // 32 * 32 - 1]) if p >= 32 else -1
                         for p in positions.tolist()]
                hidden = hidden + torch.tensor(prior)[None, :, None]
        logits = torch.zeros(1, q, 6)
        tokens = inputs_embeds[0, :, 0].int().tolist()
        for i, (p, token, context) in enumerate(zip(positions.tolist(), tokens, prior)):
            block = p // 32
            draft = self.propose_at.get(p, self.propose(block, token, context))
            logits[0, i, draft] = 12.0 if token or block in self.fill else 0.05
        return SimpleNamespace(logits=logits)


class Tok:
    eos_token_id = 0
    all_special_ids = [0]

    def convert_tokens_to_ids(self, token):
        return 0

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


def generate(model, arm, *, length=160, seed=11):
    task = dict(index=0, dataset="toy", task_id="t", seed=seed)
    return om.generate(model, Tok(), task, [5] * 32, arm, cap=length)


@pytest.fixture(scope="module")
def cpu_rows():
    model = PairModel().eval()
    return {arm: generate(model, arm) for arm in om.ARMS}


def test_trace_contract():
    assert om.TRACE == om.PLAN.trace
    assert om.TARGETS == om.PLAN.targets
    assert om.ARMS[om.CAUSAL] is None
    assert all(om.ARMS[f"open{q}"] == q for q in om.TARGETS)
    assert om.PLAN.folder.startswith("llada21-")


def test_mask_allowed_dtypes_and_full_visible_additive():
    native = torch.ones(2, 3, dtype=torch.bool)
    assert torch.equal(om.mask_allowed(torch.ones(2, 3, dtype=torch.bool), native), native)
    assert torch.equal(om.mask_allowed(torch.ones(2, 3, dtype=torch.int64), native), native)
    # The real path may send a fully visible additive float matrix (all zeros).
    assert torch.equal(om.mask_allowed(torch.zeros(2, 3), native), native)
    bias = torch.zeros(2, 3).masked_fill(~native, torch.finfo(torch.float32).min)
    assert torch.equal(om.mask_allowed(bias, native), native)
    with pytest.raises(ValueError, match="Unsupported"):
        om.mask_allowed(torch.ones(2, 3, dtype=torch.complex64), native)
    with pytest.raises(ValueError, match="additive"):
        om.mask_allowed(torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.bool))


def test_real_pair_releases_only_at_target(cpu_rows):
    assert all(e.get("opened_edges", 0) == 0 for e in cpu_rows[om.CAUSAL]["events"])
    for target in om.TARGETS:
        result = om.pair(cpu_rows[om.CAUSAL], cpu_rows[f"open{target}"], target, cap=160)
        eligible = cpu_rows[om.CAUSAL]["offer_mask_witness"][str(target)]["rounds"]
        assert result["rounds"] == eligible
        if eligible:
            assert result["exposed"] and result["released_rounds"] >= 1
            opened = [e for e in cpu_rows[f"open{target}"]["events"] if e.get("opened_edges")]
            assert opened and all(e["q_budget"] == target for e in opened)
            assert all(e["opened_edges"] == e["mask_window"]["opened_edges"] for e in opened)
        else:
            # A target with no eligible forward is a true negative control.
            assert not result["exposed"] and result["released_rounds"] == 0
            assert cpu_rows[f"open{target}"]["output_ids"] == cpu_rows[om.CAUSAL]["output_ids"]
            assert (cpu_rows[om.CAUSAL]["offer_mask_witness"][str(target)]["trajectory_sha256"]
                    == cpu_rows[f"open{target}"]["offer_mask_witness"][str(target)]["trajectory_sha256"])
    for target in om.TARGETS:
        assert all(e.get("opened_edges", 0) == 0
                   for e in cpu_rows[f"open{target}"]["events"] if e.get("q_budget") != target)


def test_pair_rejects_changed_pretrace(cpu_rows):
    trial = deepcopy(cpu_rows["open32"])
    trial["events"][1]["quote"]["mandatory_q"] += 1
    with pytest.raises(ValueError, match="witness|pre-exposure"):
        om.pair(cpu_rows[om.CAUSAL], trial, 32, cap=160)


def test_audit_rejects_causal_open(cpu_rows):
    row = deepcopy(cpu_rows[om.CAUSAL])
    row["events"][3]["opened_edges"] = 1
    with pytest.raises(ValueError, match="release decision"):
        om.audit(row, om.CAUSAL, cap=160)


def test_audit_rejects_wrong_witness(cpu_rows):
    row = deepcopy(cpu_rows["open32"])
    row["offer_mask_witness"]["32"]["rounds"] += 1
    with pytest.raises(ValueError, match="witness"):
        om.audit(row, "open32", cap=160)


def test_witness_adapter(cpu_rows):
    schema = om.witness(cpu_rows["open32"])
    assert set(schema) == {str(q) for q in om.TARGETS}
    assert all(set(v) == {"first_tick", "pretrace_sha256", "geometry_sha256",
                          "rounds", "edge_pairs"} for v in schema.values())
    assert schema["32"]["rounds"] >= 1
    assert om.released(cpu_rows["open32"])["4"]["released_rounds"] == 0
    records = om.trace(cpu_rows["open32"])
    assert all(set(x) == {"tick", "offer", "opportunity", "opened"} for x in records)


def test_check_report(capsys):
    report = om.check()
    assert report["trace"] == list(om.TRACE)
    assert report["arms"] == {k: v for k, v in om.ARMS.items()}
    assert report["plan"] == om.PLAN.folder
    assert report["predecessor_state"] in ("running", "returned")
    capsys.readouterr()


def test_cli_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["llada_offer_masks.py", "--help"])
    with pytest.raises(SystemExit) as error:
        om.main()
    assert error.value.code == 0
    capsys.readouterr()


def test_summary_tables(tmp_path, cpu_rows):
    tasks = {i: dict(index=i, dataset="toy", task_id=f"t{i}", seed=11) for i in range(90)}
    rows = {arm: {i: dict(cpu_rows[arm], correct=True) for i in range(90)} for arm in om.ARMS}
    pairs = {q: {i: dict(target=q, exposed=True, first_tick=2, rounds=1, edge_pairs=32,
                         released_rounds=1, released_edge_pairs=32)
                 for i in range(90)} for q in om.TARGETS}
    summary = om.summarize(rows, pairs, tasks)
    assert set(summary["cohorts"]) == {"original30", "new60", "full90"}
    assert summary["cohorts"]["original30"]["32"]["requests"] == 30
    assert summary["cohorts"]["new60"]["32"]["requests"] == 60
    om.write_tables(tmp_path, rows, pairs, tasks)
    assert (tmp_path / "per_question.csv").exists()
    assert (tmp_path / "summary.csv").exists()
