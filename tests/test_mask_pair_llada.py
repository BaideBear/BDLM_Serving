"""CPU contracts for the LLaDA2.1 Q64 head->tail mask ablation."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bdlm_serving.methods.t2t.llada_dual import DualBlockKernel
from experiments.t2t.serving_slots import mask_pair_llada as mp


MASK = 0
CFG = dict(confidence_threshold=.8, editing_threshold=.8, max_post_steps=16)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Module()
        self.attention.query_key_value = nn.Identity()


class _Toy(nn.Module):
    """Deterministic B32 model that records every visibility matrix."""

    def __init__(self, accept_blocks=(), edit=False, flip=False, to_mask=False):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(), _Layer()])
        self.embedding = nn.Embedding(4, 1)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(4).view(4, 1))
        self.accept_blocks = set(accept_blocks)
        self.edit = edit
        self.flip = flip
        self.to_mask = to_mask
        self.masks = []
        self.positions = []

    @property
    def device(self):
        return self.embedding.weight.device

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds, position_ids, attention_mask,
                past_key_values, use_cache, output_attentions):
        positions = position_ids[0]
        q, end = positions.numel(), attention_mask.shape[-1]
        assert attention_mask.shape == (1, 1, q, end)
        self.masks.append(attention_mask.clone())
        self.positions.append(position_ids.clone())
        assert use_cache and output_attentions
        for i, layer in enumerate(self.model.layers):
            hidden = inputs_embeds + i * 10
            kv = layer.attention.query_key_value(hidden).unsqueeze(1)
            key, value = past_key_values.update(kv, kv + 100, i)
            assert key.shape[-2] == value.shape[-2] == end
            assert torch.equal(key.index_select(-2, positions), kv)
        tokens = inputs_embeds.squeeze(-1).long()
        draft = torch.where(tokens == MASK, 1, tokens)
        if self.edit:
            draft = torch.where(tokens == 1, 2, draft)
        if self.flip:
            draft = torch.where(tokens == 1, 2,
                                torch.where(tokens == 2, 1, draft))
        if self.to_mask:
            draft = torch.where(tokens == 1, MASK, draft)
        block = positions // 32
        high = torch.tensor([int(b) in self.accept_blocks for b in block.tolist()],
                            dtype=torch.bool)
        mask_conf = torch.where(high, 12.0, 0.0)
        conf = torch.where(tokens == MASK, mask_conf, 12.0)
        logits = torch.zeros(1, q, 4).scatter_(-1, draft[..., None], conf[..., None])
        return SimpleNamespace(logits=logits)

    @staticmethod
    def _sample_with_temperature_topk_topp(logits, *, temperature, top_k, top_p):
        assert temperature == 0.0 and top_k is top_p is None
        draft = logits.argmax(-1)
        return draft, logits.softmax(-1).gather(-1, draft[..., None]).squeeze(-1)


def _kernel(accept_blocks=(1, 2, 3), seed=7):
    model = _Toy(accept_blocks=accept_blocks).eval()
    mp.install_mask_pair_hook(model)
    kernel = DualBlockKernel(model, 64, 0, 64, "hard_noedit", selection_seed=seed)
    return model, kernel


def _state(kernel, prompt_length=32, gen_length=64, seed=29):
    return kernel.create(torch.full((1, prompt_length), 3), CFG, gen_length,
                         seed, MASK, ())


def _block_causal(positions, end):
    positions = torch.as_tensor(positions)
    keys = torch.arange(end)
    return (positions[:, None] // 32 >= keys[None, :] // 32)[None, None]


def test_head_to_tail_opens_only_generated_head_rows_in_active_pair():
    positions = torch.arange(32, 96)
    mask = _block_causal(positions, 96)
    updated, info = mp.head_to_tail_mask(mask, positions[None], prompt_length=34, block=1)

    assert info == dict(block=1, left_query_count=30, left_query_first=34,
                        left_query_last=63, right_key_start=64, right_key_end=96,
                        opened_edges=30 * 32)
    # Prompt-tail rows 32 and 33 remain causal.
    assert not bool(updated[0, 0, 0, 64])
    assert not bool(updated[0, 0, 1, 64])
    # Generation head row 34 now sees the tail.
    assert bool(updated[0, 0, 2, 64])
    assert bool(updated[0, 0, 2, 95])
    # Nothing before the active pair end changes; right/addressed rows are untouched.
    assert torch.equal(updated[0, 0, :, :64], mask[0, 0, :, :64])
    assert torch.equal(updated[0, 0, 32:, :], mask[0, 0, 32:, :])


def test_head_to_tail_is_a_noop_without_a_tail_or_generated_head_rows():
    # No right B32 in the visible range.
    positions = torch.arange(32, 64)
    mask = _block_causal(positions, 64)
    updated, info = mp.head_to_tail_mask(mask, positions[None], prompt_length=32, block=1)
    assert info is None and updated is mask
    # Right block exists, but every left row is prompt/history.
    positions = torch.arange(32, 96)
    mask = _block_causal(positions, 96)
    updated, info = mp.head_to_tail_mask(mask, positions[None], prompt_length=64, block=1)
    assert info is None and updated is mask


def test_pre_hook_disabled_during_prefill_and_open_only_in_active_pair():
    class Sink(nn.Module):
        def __init__(self):
            super().__init__()
            self.masks = []

        def forward(self, *, attention_mask, position_ids, **kwargs):
            self.masks.append(attention_mask.clone())

    sink = Sink()
    handle = mp.install_mask_pair_hook(sink)
    try:
        positions = torch.arange(32, 96)
        mask = _block_causal(positions, 96)
        record = mp.new_record("off")
        with mp.mask_pair_context(arm="off", prompt_length=34, block=1, phase="prefill",
                                  event_index=0, enabled=False, record=record):
            sink(attention_mask=mask, position_ids=positions[None])
        assert torch.equal(sink.masks[-1], mask)
        assert record["opened_edges"] == 0

        with mp.mask_pair_context(arm="off", prompt_length=34, block=1, phase="denoise",
                                  event_index=3, enabled=True, record=record):
            sink(attention_mask=mask, position_ids=positions[None])
        assert not torch.equal(sink.masks[-1], mask)
        assert record["opened_edges"] == 30 * 32
        assert record["first_trigger"]["phase"] == "denoise"
        assert record["first_trigger"]["event_index"] == 3
    finally:
        handle.remove()


def test_dual_kernel_off_records_first_trigger_and_keeps_prefill_causal():
    model, kernel = _kernel()
    state = _state(kernel, prompt_length=32, gen_length=64)
    record = mp.new_record("off")
    mp.run_kernel_steps(kernel, state, "off", record)

    assert record["opened_edges"] > 0
    first = record["first_trigger"]
    assert first["stage"] == "bootstrap"
    assert first["phase"] == "denoise"
    assert first["block"] == 1
    assert first["event_index"] == 1
    assert first["left_query_first"] == 32 and first["left_query_last"] == 63
    assert record["edge_forward_stages"]["bootstrap"] >= 1

    # The first forward is prefill; it must remain block-causal.
    prefill_positions = model.positions[0][0]
    assert torch.equal(model.masks[0][0, 0],
                       _block_causal(prefill_positions, model.masks[0].shape[-1])[0, 0])
    # The pair bootstrap is the first opened forward.
    bootstrap_positions = model.positions[1][0]
    bootstrap_mask = model.masks[1]
    assert bool(bootstrap_mask[0, 0, 0, 64])      # pos 32 -> key 64
    assert torch.equal(bootstrap_mask[0, 0, :, :32],
                       _block_causal(bootstrap_positions, 96)[0, 0, :, :32])


def test_prompt_tail_rows_in_left_block_never_see_right_block():
    model, kernel = _kernel()
    state = _state(kernel, prompt_length=34, gen_length=64)
    record = mp.new_record("off")
    mp.run_kernel_steps(kernel, state, "off", record)

    assert record["first_trigger"]["left_query_first"] == 34
    bootstrap = model.masks[1][0, 0]
    # positions 32 and 33 are the prompt tail; 34 is the first generated row.
    assert not bool(bootstrap[0, 64])
    assert not bool(bootstrap[1, 64])
    assert bool(bootstrap[2, 64])


def test_single_block_off_has_zero_edges_and_matches_on_arm():
    results = {}
    for arm in ("on", "off"):
        model, kernel = _kernel()
        state = _state(kernel, prompt_length=32, gen_length=32)
        record = mp.new_record(arm)
        mp.run_kernel_steps(kernel, state, arm, record)
        results[arm] = (state, record)

    on_state, on_record = results["on"]
    off_state, off_record = results["off"]
    assert on_record["opened_edges"] == off_record["opened_edges"] == 0
    assert off_record["first_trigger"] is None
    assert torch.equal(on_state.x, off_state.x)
    assert on_state.events == off_state.events


def test_zero_edge_identity_rejects_output_or_work_drift():
    row = {key: None for key in mp.PARITY_KEYS}
    row.update(output_ids=[1, 2], output_tokens=2, generation_forwards=1,
               generation_query_rows=32, generation_physical_padded_rows=32,
               max_q=32, q_histogram={"32": 1}, forwards_by_phase={"selected": 1},
               q_by_phase={"selected": 32}, prefill_rows=32, t2t_edits=0,
               eos=True, truncated=False,
               events=[{key: None for key in mp.EVENT_KEYS}])
    baseline = {key: (list(value) if isinstance(value, list) else value)
                for key, value in row.items()}
    baseline["events"] = [dict(event) for event in row["events"]]
    row["mask_pair"] = {"arm": "off", "first_trigger": {"stage": "bootstrap"}}
    baseline["mask_pair"] = {"arm": "on", "first_trigger": None}
    mp.assert_same_output_work(row, baseline)

    row["generation_forwards"] = 2
    with pytest.raises(ValueError, match="generation_forwards"):
        mp.assert_same_output_work(row, baseline)


def test_labels_and_scope():
    assert mp.POLICY_ON == "dual64_noedit"
    assert mp.POLICY_OFF == "dual64_noedit_bidir"
    assert mp._arm_description("on") == "block_causal"
    assert mp._arm_description("off") == "bidirectional_active_pair_head_to_tail"
    assert mp.EDGE == "head_to_tail"

def test_arm_resolution_and_summary_rows(tmp_path):
    assert mp._resolve_arms("both") == ["on", "off"]
    assert mp._resolve_arms("off") == ["off"]
    with pytest.raises(ValueError, match="Duplicate"):
        mp._resolve_arms("on,on")
    with pytest.raises(ValueError, match="must contain"):
        mp._resolve_arms("both,on")

    rows = []
    for index, stage in ((0, "bootstrap"), (1, "selected")):
        rows.append(dict(
            index=index, dataset="gsm8k", correct=True,
            output_tokens=2, generation_forwards=3, generation_query_rows=64,
            mask_pair=dict(
                arm="off", opened_edges=2, opened_forward_count=1,
                edge_forward_stages={stage: 1},
                first_trigger=dict(stage=stage, phase="denoise", block=1),
            ),
        ))
    summary, = [mp._summary_rows("off", rows)]
    assert summary["requests"] == 2
    assert summary["opened_edges"] == 4
    assert summary["first_trigger_stages"] == {"bootstrap": 1, "selected": 1}
    assert summary["edge_forward_stages"] == {"bootstrap": 1, "selected": 1}
    path = tmp_path / "summary.csv"
    mp._write_summary_csv(path, rows)
    text = path.read_text()
    assert "bootstrap" in text and "selected" in text and "dual64_noedit_bidir" not in text

def test_audit_dual64_accepts_off_label_and_extra_metadata():
    from tools.run_dual_t2t import audit_dual64, saturation_summary

    positions = list(range(32, 96))
    events = [dict(stage="selected", phase="denoise", F=1, actual_q=64,
                   effective_q=64, physical_q=64, absolute_positions=positions,
                   layer_physical_q=[64, 64], active_width=64,
                   saturated=True, full_two_block=True)]
    row = dict(
        policy="dual64_noedit_bidir",
        dual=dict(nominal_q=64, pair_blocks=2, gen_length=2048, mode="rolling_pair"),
        dual_saturation=saturation_summary(events), events=events,
        generation_query_rows=64, generation_physical_padded_rows=64,
        generation_forwards=1, max_q=64,
        mask_pair=dict(arm="off", opened_edges=1,
                       first_trigger=dict(stage="bootstrap")),
    )
    audit_dual64(row)


def test_prompt_reference_requires_all_90_core_cases(tmp_path, monkeypatch):
    short = tmp_path / "short"
    short.mkdir()
    for index in range(30):
        (short / f"{index:02d}-native32.json").write_text("{}")
    samples = tmp_path / "prefix90" / "samples.json"
    samples.parent.mkdir()
    fallback = tmp_path / "frozen90-native" / "llada21"
    fallback.mkdir(parents=True)
    for index in range(90):
        (fallback / f"{index:02d}-native32.json").write_text("{}")

    monkeypatch.setenv("MASK_PAIR_REFERENCE", str(short))
    args = SimpleNamespace(reference=short)
    assert mp._prompt_reference(args, samples) == fallback


def test_queue_prerequisites_uses_private_bundle_beside_prefix90(tmp_path):
    prefix = tmp_path / "prefix90"
    prefix.mkdir()
    prefix16 = prefix / "llada21-native-high-prefix16"
    prefix16.mkdir()
    for name in ("complete-slottrace_16r1.json", "complete-slottrace_32n.json",
                 "complete-slottrace_32n_32n_16r1_16r1.json",
                 "complete-slottrace_16r1_16r1_32n_32n.json"):
        (prefix16 / name).write_text("{}")
    groups = {
        "anchor": ("ibs_32", "obs_32"),
        "endpoints": ("ibs_8", "obs_8", "ibs_16", "obs_16"),
        "low": ("ibs_4", "obs_4"),
        "switch": ("ibs_8_8_16_16", "obs_8_8_16_16",
                   "ibs_16_16_8_8", "obs_16_16_8_8"),
    }
    for group, policies in groups.items():
        folder = prefix / f"llada21-optimus-hardcap-{group}"
        folder.mkdir()
        for policy in policies:
            (folder / f"complete-optcap_{policy}.json").write_text("{}")
    (prefix / "llada21-optimus-hardcap-anchor" / "replay-qualified.json").write_text("{}")

    bundle = tmp_path / "private-llada-native-slot90-v1"
    bundle.mkdir()
    (bundle / "verified.json").write_text("{}")
    assert mp._queue_prerequisites(prefix)["ready"] is True

    (bundle / "verified.json").unlink()
    result = mp._queue_prerequisites(prefix)
    assert result["ready"] is False
    assert result["required"] == ["private-llada-native-slot90-v1/verified.json"]

def test_head_to_tail_infers_pair_block_from_visible_end():
    positions = torch.arange(32, 96)
    mask = _block_causal(positions, 96)
    updated, info = mp.head_to_tail_mask(mask, positions[None],
                                         prompt_length=34, block=None)
    assert info["block"] == 1 and info["opened_edges"] == 30 * 32
    # A one-block commit ends at 64; positions 32..63 are not a left query set.
    commit_positions = torch.arange(32, 64)
    commit_mask = _block_causal(commit_positions, 64)
    _, info = mp.head_to_tail_mask(commit_mask, commit_positions[None],
                                   prompt_length=34, block=None)
    assert info is None
    # A 64-token prefill at key_end=64 has only prompt rows in the left block.
    prefill_positions = torch.arange(0, 64)
    prefill_mask = _block_causal(prefill_positions, 64)
    _, info = mp.head_to_tail_mask(prefill_mask, prefill_positions[None],
                                   prompt_length=64, block=None)
    assert info is None


def test_prompt_reference_rejects_explicit_30_case_only(tmp_path, monkeypatch):
    short = tmp_path / "short"
    short.mkdir()
    for index in range(30):
        (short / f"{index:02d}-native32.json").write_text("{}")
    samples = tmp_path / "isolated" / "prefix90" / "samples.json"
    samples.parent.mkdir(parents=True)
    monkeypatch.setenv("MASK_PAIR_REFERENCE", str(short))
    with pytest.raises(FileNotFoundError, match="MASK_PAIR_REFERENCE"):
        mp._prompt_reference(SimpleNamespace(reference=None), samples)


def test_runtime_provenance_pins_repo_modules_and_launcher_venv():
    repo = Path(mp.__file__).resolve().parents[3]
    if not (repo.parent / "venv").exists():
        pytest.skip("n164 launcher venv layout not present")
    provenance = mp._runtime_provenance(SimpleNamespace(checkpoint=Path("/tmp")))
    assert provenance["venv"] == str(Path(sys.prefix).resolve())
    for paths in provenance["module_paths"].values():
        assert paths and all(path.startswith(provenance["repo"]) for path in paths)

def test_inferred_block_context_matches_production_dual_path():
    model, kernel = _kernel()
    state = _state(kernel, prompt_length=32, gen_length=64)
    record = mp.new_record("off")
    context = mp.MaskPairContext(
        arm="off", prompt_length=state.prompt_length, block=None, phase="",
        event_index=-1, enabled=True, stage_hint="", record=record)
    token = mp._CURRENT.set(context)
    try:
        while state.phase != "done":
            kernel.step(state)
    finally:
        mp._CURRENT.reset(token)

    first = record["first_trigger"]
    assert record["opened_edges"] > 0
    assert first["stage"] == "bootstrap"
    assert first["block"] == 1
    assert first["forward_index"] == 1
    assert first["event_index"] is None
    prefill_positions = model.positions[0][0]
    assert torch.equal(model.masks[0][0, 0],
                       _block_causal(prefill_positions, model.masks[0].shape[-1])[0, 0])

def test_run_writes_on_off_layout_without_loading_a_model(tmp_path, monkeypatch):
    import copy
    import hashlib
    import json
    from types import SimpleNamespace

    from tools.run_dual_t2t import saturation_summary

    class FakeTokenizer:
        eos_token_id = 0
        all_special_ids = []

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def __call__(self, text, **kwargs):
            ids = [(ord(char) % 7) + 1 for char in text][:16] or [1]
            return SimpleNamespace(input_ids=torch.tensor([ids]))

    tasks = [dict(index=index, dataset=("gsm8k", "math500", "humaneval")[index % 3],
                  task_id=f"task/{index}", seed=17000 + index,
                  prompt=f"prompt {index}")
             for index in range(90)]
    samples = tmp_path / "prefix90" / "samples.json"
    samples.parent.mkdir()
    samples.write_text(json.dumps({"samples": tasks}))
    samples_sha = hashlib.sha256(samples.read_bytes()).hexdigest()
    monkeypatch.setattr(mp, "SAMPLES_SHA256", samples_sha)

    reference = tmp_path / "frozen90-native" / "llada21"
    reference.mkdir(parents=True)
    tokenizer = FakeTokenizer()
    for task in tasks:
        ids = tokenizer(task["prompt"]).input_ids[0].tolist()
        (reference / f"{task['index']:02d}-native32.json").write_text(
            json.dumps({"prompt_ids": ids}))

    events = [dict(phase="denoise", stage="selected", F=1, physical_q=64, actual_q=64,
                   effective_q=64, absolute_positions=list(range(32, 96)),
                   layer_physical_q=[64], visible_kv=96, blocks=[1, 2],
                   active_width=64, saturated=True, full_two_block=True,
                   q_budget=64, overflow=0)]
    row = dict(status="success", output_ids=[1], output_text="1", output_tokens=1,
               generation_forwards=1, generation_query_rows=64,
               generation_physical_padded_rows=64, max_q=64,
               q_histogram={"64": 1}, forwards_by_phase={"selected": 1},
               q_by_phase={"selected": 64}, prefill_rows=32, t2t_edits=0,
               tpf=1.0, token_efficiency=1 / 64,
               eos=True, truncated=False, correct=True,
               dual=dict(nominal_q=64, pair_blocks=2, gen_length=2048,
                         mode="rolling_pair"),
               dual_saturation=saturation_summary(events), events=events)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "run.json").write_text(json.dumps(dict(
        model="llada21", suite="frozen90", samples_sha256=samples_sha,
        block=32, pair_blocks=2, nominal_q=64,
        mode="llada_dual_rolling_pair", cap=2048, batch_size=1,
        policies=["dual64_noedit"])))
    (baseline / "complete-dual64_noedit.json").write_text(
        json.dumps({"policy": "dual64_noedit", "requests": 90}))
    for task in tasks:
        item = copy.deepcopy(row)
        item.update(index=task["index"], dataset=task["dataset"],
                    task_id=task["task_id"], seed=task["seed"],
                    model="llada21", policy="dual64_noedit")
        (baseline / f"{task['index']:02d}-dual64_noedit.json").write_text(
            json.dumps(item))

    monkeypatch.setattr("bdlm_serving.evaluation.frozen90.load_tasks",
                        lambda path: tasks)
    monkeypatch.setattr("transformers.AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(mp, "_runtime_provenance", lambda args: {"test": True})

    def fake_dual_backend(args, prepare):
        def run_task(task, arm):
            return copy.deepcopy(json.loads(
                (baseline / f"{task['index']:02d}-dual64_noedit.json").read_text()))
        return run_task

    monkeypatch.setattr("tools.run_dual_t2t.dual_backend", fake_dual_backend)
    monkeypatch.setenv("MASK_PAIR_BASELINE", str(baseline))
    monkeypatch.setenv("MASK_PAIR_ARMS", "both")
    monkeypatch.setenv("MASK_PAIR_REFERENCE", str(reference))
    output = tmp_path / "out"
    args = SimpleNamespace(samples=samples, checkpoint=tmp_path / "ckpt",
                           output=output, reference=reference)

    summaries = mp.run(args)
    assert [summary["arm"] for summary in summaries] == ["on", "off"]
    assert (output / "on" / "complete-dual64_noedit.json").is_file()
    assert (output / "off" / "complete-dual64_noedit_bidir.json").is_file()
    assert (output / "on" / "00-dual64_noedit.json").is_file()
    assert (output / "off" / "00-dual64_noedit_bidir.json").is_file()
    summary = json.loads((output / "mask_pair_summary.json").read_text())
    assert summary["arms"][0]["zero_edge_requests"] == 90
    assert summary["arms"][1]["zero_edge_requests"] == 90
    header = (output / "mask_pair_summary.csv").read_text().splitlines()[0]
    assert "novel" in header and "first_trigger_stage" in header

def test_launcher_fails_closed_before_exec_when_tail_missing(tmp_path, monkeypatch):
    samples = tmp_path / "prefix90" / "samples.json"
    samples.parent.mkdir()
    samples.write_text("{}")
    monkeypatch.setenv("FROZEN90_SAMPLES", str(samples))
    monkeypatch.setenv("PREFIX_MODE", "llrefs")
    monkeypatch.setenv("TRACE_OUT", str(tmp_path / "out"))
    monkeypatch.setattr(mp, "_queue_prerequisites",
                        lambda data: {"ready": False, "required": ["tail"]})
    with pytest.raises(RuntimeError, match="tail prerequisites"):
        mp.launcher_main(["llada21", "4"])


def test_launcher_requires_trace_out_before_exec(tmp_path, monkeypatch):
    samples = tmp_path / "prefix90" / "samples.json"
    samples.parent.mkdir()
    samples.write_text("{}")
    monkeypatch.setenv("FROZEN90_SAMPLES", str(samples))
    monkeypatch.setenv("PREFIX_MODE", "llrefs")
    monkeypatch.delenv("TRACE_OUT", raising=False)
    monkeypatch.setattr(mp, "_queue_prerequisites",
                        lambda data: {"ready": True, "required": []})
    with pytest.raises(RuntimeError, match="TRACE_OUT"):
        mp.launcher_main(["llada21", "4"])
