"""CPU contracts for the two-B32 rolling LLaDA runner."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bdlm_serving.methods.t2t.llada_dual import DualBlockKernel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MASK = 0
CFG = dict(confidence_threshold=.8, editing_threshold=.8, max_post_steps=16)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Module()
        self.attention.query_key_value = nn.Identity()


class _Toy(nn.Module):
    """Deterministic two-block model with block-causal mask assertions."""

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
        expected = positions[:, None] // 32 >= torch.arange(end)[None, :] // 32
        assert torch.equal(attention_mask[0, 0], expected)
        self.masks.append(attention_mask.clone())
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


def _kernel(accept_blocks=(), edit=False, policy="hard_frontier", seed=7,
            flip=False, to_mask=False):
    model = _Toy(accept_blocks=accept_blocks, edit=edit,
                 flip=flip, to_mask=to_mask).eval()
    mask = 56 if policy == "hard_frontier" else 64
    kernel = DualBlockKernel(model, 64, 0, mask, policy, selection_seed=seed)
    return model, kernel


def _state(kernel, prompt_length=32, gen_length=64, seed=29):
    return kernel.create(torch.full((1, prompt_length), 3), CFG, gen_length,
                         seed, MASK, ())


def test_pair_bootstrap_q64_visible_mask_and_commit_sequence():
    model, kernel = _kernel(accept_blocks={1, 2}, edit=False)
    state = _state(kernel, prompt_length=32, gen_length=64)
    assert kernel.step(state) == "prefill"
    assert kernel.step(state) == "denoise"
    bootstrap = state.events[-1]
    assert bootstrap["stage"] == "bootstrap"
    assert bootstrap["physical_q"] == 64
    assert bootstrap["blocks"] == [1, 2]
    assert bootstrap["active_width"] == 64
    assert bootstrap["effective_q"] == 64
    assert bootstrap["saturated"] is True
    assert bootstrap["full_two_block"] is True
    assert state.resident[0][0].shape[-2] == 96

    mask = model.masks[-1][0, 0]
    # A left query (pos 32) cannot see a right key (pos 64).
    assert not bool(mask[0, 64])
    # A right query (pos 64) can see the left key (pos 32).
    assert bool(mask[32, 32])

    # The mask-filling bootstrap cannot commit: it needs a no-MASK pass.
    assert not state.left_done and not state.right_done
    assert kernel.step(state) == "denoise"
    verify = state.events[-1]
    assert verify["stage"] == "selected"
    assert verify["physical_q"] == 64
    assert verify["masked_before"] == verify["masked_after"] == 0
    assert state.left_done and state.right_done

    assert kernel.step(state) == "commit"
    assert state.events[-1]["stage"] == "commit"
    assert state.cache.get_seq_length() == 64
    assert kernel.step(state) == "reencode"
    assert state.events[-1]["stage"] == "reencode"
    assert kernel.step(state) == "commit"
    assert state.phase == "done"
    assert state.cache.get_seq_length() == 96
    assert [event["stage"] for event in state.events] == [
        "prefill", "bootstrap", "selected", "commit", "reencode", "commit"]
    assert sum(event["physical_q"] for event in state.events) == 256


def test_selected_q64_is_saturated_and_noedit_matches_frontier():
    noedit_model, noedit = _kernel(policy="hard_noedit")
    frontier_model, frontier = _kernel(policy="hard_frontier")
    noedit_state = _state(noedit, prompt_length=32, gen_length=64)
    frontier_state = _state(frontier, prompt_length=32, gen_length=64)
    for kernel, state in ((noedit, noedit_state), (frontier, frontier_state)):
        assert kernel.step(state) == "prefill"
        assert kernel.step(state) == "denoise"
        assert kernel.step(state) == "denoise"
        event = state.events[-1]
        assert event["stage"] == "selected"
        assert event["physical_q"] == event["effective_q"] == event["active_width"] == 64
        assert event["saturated"] is True
        assert event["full_two_block"] is True
        assert event["blocks"] == [1, 2]
    noedit_event = noedit_state.events[-1]
    frontier_event = frontier_state.events[-1]
    assert noedit_event["absolute_positions"] == frontier_event["absolute_positions"]
    assert noedit_event["reserved_extra_edit_positions"] == []
    assert set(frontier_event["reserved_extra_edit_positions"]) <= set(
        frontier_event["absolute_positions"])
    assert torch.equal(noedit_state.x, frontier_state.x)
    assert noedit_state.last_query == frontier_state.last_query


def test_prompt_tail_caps_q64_without_fabricating_rows():
    _, kernel = _kernel(accept_blocks=(), edit=False)
    state = _state(kernel, prompt_length=34, gen_length=64)
    kernel.step(state)  # prefill
    kernel.step(state)  # pair bootstrap
    kernel.step(state)  # selected
    event = state.events[-1]
    assert event["stage"] == "selected"
    assert event["active_width"] == 62
    assert event["effective_q"] == event["physical_q"] == 62
    assert event["saturated"] is True
    assert event["full_two_block"] is False
    assert 32 not in event["absolute_positions"]
    assert 33 not in event["absolute_positions"]
    assert 34 in event["absolute_positions"]
    assert 95 in event["absolute_positions"]


def test_commit_left_reencode_right_then_bootstrap_new_right():
    _, kernel = _kernel(accept_blocks={1}, edit=False)
    state = _state(kernel, prompt_length=32, gen_length=128)
    kernel.step(state)  # prefill
    kernel.step(state)  # bootstrap: left has no MASK but is not yet verified
    assert not state.left_done and not state.right_done
    kernel.step(state)  # no-MASK verification of the left
    assert state.events[-1]["stage"] == "selected"
    assert state.left_done and not state.right_done
    kernel.step(state)  # commit left
    commit = state.events[-1]
    assert commit["stage"] == "commit"
    assert commit["blocks"] == [1]
    assert state.cache.get_seq_length() == 64
    assert state.committed_end == 64
    # Simulate a right block whose old-prefix no-MASK budget was exhausted.
    # The left commit changes the prefix; reencode must reset this history.
    state.right_post_steps = 7
    state.right_done = True
    kernel.step(state)  # reencode right
    reencode = state.events[-1]
    assert reencode["stage"] == "reencode"
    assert reencode["blocks"] == [2]
    assert state.right_post_steps == 0
    assert not state.right_done
    assert state.block == 2
    assert state.committed_end == 64
    assert state.resident[0][0].shape[-2] == 96
    kernel.step(state)  # bootstrap the new right block 3
    boot = state.events[-1]
    assert boot["stage"] == "bootstrap"
    assert boot["blocks"] == [3]
    assert boot["physical_q"] == 32
    assert state.resident[0][0].shape[-2] == 128
    assert state.cache.get_seq_length() == 64
    assert state.last_query[0] >= 0 and state.last_query[32] >= 0
    assert [event["stage"] for event in state.events] == [
        "prefill", "bootstrap", "selected", "commit", "reencode", "bootstrap"]


def test_edit_semantics_prompt_fixed_and_reserved_subset():
    model, kernel = _kernel(accept_blocks=(), edit=True)
    state = _state(kernel, prompt_length=32, gen_length=64)
    state.x[0, 34:38] = 1
    state.x[0, 66:70] = 1
    prompt = state.x[0, :32].clone()
    kernel.step(state)  # prefill
    kernel.step(state)  # pair bootstrap
    bootstrap = state.events[-1]
    assert bootstrap["hard_strict_edits"] == 8
    assert bootstrap["reserved_extra_edit_positions"] == []
    assert torch.equal(state.x[0, :32], prompt)
    assert state.x[0, 34:38].tolist() == [2, 2, 2, 2]
    assert state.x[0, 66:70].tolist() == [2, 2, 2, 2]
    assert {2, 3, 4, 5} <= state.left_history.fresh
    assert {2, 3, 4, 5} <= state.right_history.fresh

    kernel.step(state)  # selected
    event = state.events[-1]
    assert set(event["reserved_extra_edit_positions"]) <= set(event["absolute_positions"])
    assert set(event["reserved_strict_edit_positions"]) <= set(
        event["reserved_extra_edit_positions"])
    assert event["hard_strict_edits"] >= len(event["reserved_strict_edit_positions"])


def test_commit_refuses_mask():
    _, kernel = _kernel(accept_blocks=(), edit=False)
    state = _state(kernel, prompt_length=32, gen_length=64)
    with pytest.raises(RuntimeError, match="Refusing to commit"):
        kernel._commit_block(state, state.block)


def test_saturation_summary_distinguishes_full_and_capped():
    from tools.run_dual_t2t import saturation_summary

    events = [
        dict(stage="bootstrap", active_width=64, effective_q=64,
             saturated=True, full_two_block=True, overflow=0),
        dict(stage="selected", active_width=64, effective_q=64,
             saturated=True, full_two_block=True, overflow=0),
        dict(stage="selected", active_width=62, effective_q=62,
             saturated=True, full_two_block=False, overflow=0),
    ]
    summary = saturation_summary(events)
    assert summary["selected_steps"] == 2
    assert summary["saturated_steps"] == 2
    assert summary["unsaturated_steps"] == 0
    assert summary["full_two_block_steps"] == 1
    assert summary["capped_steps"] == 1
    assert summary["effective_q_max"] == 64
    assert summary["active_width_max"] == 64


def test_nonempty_fresh_forces_no_mask_recheck():
    _, kernel = _kernel(accept_blocks={1, 2}, edit=True)
    state = _state(kernel, prompt_length=32, gen_length=64)
    kernel.step(state)  # prefill
    kernel.step(state)  # bootstrap fills MASKs; must not commit
    assert not state.left_done and not state.right_done
    kernel.step(state)  # first no-MASK pass edits fresh tokens
    first = state.events[-1]
    assert first["stage"] == "selected"
    assert first["masked_before"] == 0
    assert first["edited"] > 0
    assert first["hard_strict_edits"] > 0
    assert first["post_steps"] == 1
    assert not state.left_done and not state.right_done
    kernel.step(state)  # second no-MASK pass has no edits and commits
    second = state.events[-1]
    assert second["stage"] == "selected"
    assert second["edited"] == 0
    assert second["post_steps"] == 2
    assert state.left_done and state.right_done


def test_max_post_cap_does_not_add_extra_forward():
    _, kernel = _kernel(accept_blocks={1, 2}, flip=True)
    state = _state(kernel, prompt_length=32, gen_length=64)
    state.config["max_post_steps"] = 2
    kernel.step(state)  # prefill
    kernel.step(state)  # bootstrap fills MASKs
    assert not state.left_done and not state.right_done
    kernel.step(state)  # no-MASK edit, post_steps=1
    assert not state.left_done
    kernel.step(state)  # no-MASK edit, post_steps=2 -> done
    assert state.left_done and state.right_done
    selected = [event for event in state.events if event["stage"] == "selected"]
    assert len(selected) == 2
    assert [event["post_steps"] for event in selected] == [1, 2]
    assert kernel.step(state) == "commit"
    assert state.events[-1]["stage"] == "commit"


def test_no_mask_pass_writing_mask_is_not_done():
    _, kernel = _kernel(accept_blocks={1, 2}, to_mask=True)
    state = _state(kernel, prompt_length=32, gen_length=64)
    kernel.step(state)  # prefill
    kernel.step(state)  # bootstrap fills MASKs
    assert not state.left_done and not state.right_done
    kernel.step(state)  # no-MASK pass writes MASK back
    event = state.events[-1]
    assert event["stage"] == "selected"
    assert event["masked_before"] == 0
    assert event["masked_after"] == 64
    assert event["edited"] == 64
    assert not state.left_done and not state.right_done


def test_default_paired_q64_arms():
    from tools.run_dual_t2t import default_arms, validate_arms

    noedit, frontier = validate_arms(default_arms())
    assert (noedit.selection, noedit.budget, noedit.edit) == ("hard_noedit", 64, 0)
    assert (frontier.selection, frontier.budget, frontier.edit) == ("hard_frontier", 64, 8)


def test_paired_and_summary_rows_keep_saturation_fields():
    from tools.run_dual_t2t import DualArm, paired_row, summary_row

    task = dict(index=0, dataset="gsm8k", task_id="toy", seed=1)
    events = [dict(stage="selected", phase="denoise",
                   absolute_positions=[32, 64],
                   reserved_strict_edit_positions=[])]
    saturation = dict(selected_steps=1, saturated_steps=1, unsaturated_steps=0,
                      full_two_block_steps=1, capped_steps=0,
                      effective_q_max=64, active_width_max=64)
    noedit = dict(output_ids=[1, 2], output_tokens=2, generation_forwards=4,
                  generation_query_rows=160, correct=True, events=events,
                  dual_saturation=saturation)
    frontier = dict(output_ids=[1, 2], output_tokens=2, generation_forwards=4,
                    generation_query_rows=160, correct=True, events=events,
                    dual_saturation=saturation)
    paired = paired_row(task, noedit, frontier)
    assert paired["same_output"] == paired["same_selected"] == paired["same_stages"] == 1
    assert paired["delta_correct"] == 0
    summary = summary_row(
        "llada21", DualArm(64, 0, 64, "hard_noedit"),
        DualArm(64, 0, 56, "hard_frontier"), [noedit], [frontier], [paired])
    assert summary["noedit_correct"] == summary["frontier_correct"] == 1
    assert summary["same_output"] == 1
    assert summary["noedit_full_two_block_steps"] == 1


def test_main_writes_pair_files_with_fake_backend(tmp_path, monkeypatch):
    import tools.run_dual_t2t as tool

    tasks = [dict(index=i, dataset="gsm8k", task_id="t%02d" % i, seed=i,
                  prompt="toy") for i in range(30)]
    monkeypatch.setattr("bdlm_serving.evaluation.frozen.load_tasks",
                        lambda path: tasks)

    def fake_backend(args, prepare):
        def run(task, arm):
            events = [dict(stage="selected", phase="denoise",
                           absolute_positions=[32, 64],
                           reserved_strict_edit_positions=[], edited=0)]
            return dict(
                output_ids=[1], output_text="1", output_tokens=1,
                generation_forwards=1, generation_query_rows=64,
                generation_physical_padded_rows=64, max_q=64, t2t_edits=0,
                tpf=1.0, token_efficiency=1 / 64, events=events, correct=True,
                dual_saturation=dict(
                    selected_steps=1, saturated_steps=1, unsaturated_steps=0,
                    full_two_block_steps=1, capped_steps=0,
                    effective_q_max=64, active_width_max=64))
        return run

    monkeypatch.setattr(tool, "dual_backend", fake_backend)
    output = tmp_path / "dual-run"
    args = SimpleNamespace(
        model="llada21", checkpoint=tmp_path, samples=tmp_path / "samples.json",
        reference=tmp_path, output=output, arm=None, gen_length=64)
    tool.main(args)
    assert (output / "run.json").exists()
    assert (output / "paired.csv").exists()
    assert (output / "summary.csv").exists()
    assert (output / "complete-t2t_q64_f0_m64_hard_noedit.json").exists()
    assert (output / "00-t2t_q64_f0_m56_hard_frontier.json").exists()
