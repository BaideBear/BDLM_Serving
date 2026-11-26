"""CPU-only adversarial contracts for the unified Gemma 32..512 executor.

The scheduler is real. These tests inject a recording KV backend to exercise
offer clocks, mandatory rows, parking and encoder accounting. Real attention
is covered by test_gemma_wave_kv and by the separate checkpoint projection audit.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import types

import pytest
import torch
from torch import nn


B = 256
TIERS = (32, 64, 128, 256, 512)
MASK_ID = 0
VOCAB = 16
PROMPT = torch.tensor([[5, 6, 7]], dtype=torch.long)
DATA = {"input_ids": PROMPT, "attention_mask": torch.ones_like(PROMPT)}
REQUIRED = {
    "tick", "offer", "q", "head", "visible", "dirty_before", "dirty_after",
    "selected", "sampled", "ages_before", "ages_after", "encoded",
}


# ---------------------------------------------------------------------------
# Engine loading.  The engine is real; only a missing KV backend is faked.
# ---------------------------------------------------------------------------


class _FakeWaveKV:
    instances: list["_FakeWaveKV"] = []

    def __init__(self, block):
        self.block = block
        self.layers = 0
        self.visited: set[int] = set()
        self.begins: list[types.SimpleNamespace] = []
        self.drops: list[int] = []
        type(self).instances.append(self)

    def begin(self, selected, origin, device, *, causal=True):
        self.begins.append(types.SimpleNamespace(
            selected=list(selected), origin=origin, device=device, causal=causal))
        # The fake model does not run decoder attention layers, so satisfy the
        # engine's "every layer was intercepted" check explicitly.
        self.visited = set(range(self.layers))

    def drop_before(self, length):
        self.drops.append(length)


@contextmanager
def _fake_wave_attention(model, bank):
    bank.layers = len(model.model.decoder.layers)
    yield


@dataclass
class Engine:
    generate: object
    fake_kv: bool
    banks: list[_FakeWaveKV]


@pytest.fixture(scope="module")
def engine() -> Engine:
    from experiments.t2t.serving_slots import gemma_unified
    original = (gemma_unified.WaveKV, gemma_unified.wave_attention)
    gemma_unified.WaveKV = _FakeWaveKV
    gemma_unified.wave_attention = _fake_wave_attention
    try:
        yield Engine(gemma_unified.generate, fake_kv=True, banks=_FakeWaveKV.instances)
    finally:
        gemma_unified.WaveKV, gemma_unified.wave_attention = original


# ---------------------------------------------------------------------------
# Physical fake model.  It records every invocation and is protocol-compatible
# with both the unified engine and `diffusiongemma_wave`.
# ---------------------------------------------------------------------------


class _Cache:
    def __init__(self, layers=2):
        self.layers = [
            types.SimpleNamespace(keys=torch.empty(1, 1, 0, 1),
                                  values=torch.empty(1, 1, 0, 1))
            for _ in range(layers)
        ]
        self.is_compileable = False
        self.length = 0

    def get_seq_length(self):
        return self.length


class _Layer(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.self_attn = nn.Identity()
        self.self_attn.config = types.SimpleNamespace(
            _attn_implementation="sdpa", layer_idx=index, is_causal=False,
            num_key_value_groups=1,
        )


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_conditioning = nn.Identity()
        self.embed_tokens = nn.Embedding(VOCAB, 2)
        self.layers = nn.ModuleList([_Layer(i) for i in range(2)])


class _Sampler:
    def __init__(self, model):
        self.model = model
        self.ordinal = len(model.samplers)
        model.samplers.append(self)
        self.accepted_token_mask = torch.zeros(1, B, dtype=torch.bool)
        self.vocab_size = VOCAB
        self.calls = 0

    def initialize_canvas(self, batch, device):
        return torch.full((batch, B), MASK_ID, dtype=torch.long, device=device)


class FakeModel:
    """A physical decoder/encoder tape with controllable admissions and edits."""

    def __init__(self, *, steps=5, accept_first=0, accept_ordinals=None,
                 t2t_ordinals=None, t2t_call=2, finish_after=None):
        self.config = types.SimpleNamespace(
            canvas_length=B, text_config=types.SimpleNamespace(
                layer_types=["full", "sliding"], vocab_size=VOCAB),
        )
        self.cfg = types.SimpleNamespace(
            max_denoising_steps=steps, eos_token_id=[VOCAB - 1], pad_token_id=0,
            canvas_length=B, cache_implementation=None,
        )
        self.model = types.SimpleNamespace(encoder=self.encode, decoder=_Decoder())
        self.dtype = torch.float32
        self.training = False

        self.samplers: list[_Sampler] = []
        self.decoder_calls: list[types.SimpleNamespace] = []
        self.encoder_calls: list[tuple[int, list[int] | None]] = []
        self.prompt_len: int | None = None
        self.last_positions: torch.Tensor | None = None
        self.accept_first = accept_first
        self.accept_ordinals = set(range(64)) if accept_ordinals is None else set(accept_ordinals)
        self.t2t_ordinals = set() if t2t_ordinals is None else set(t2t_ordinals)
        self.t2t_call = t2t_call
        self.finish_after = finish_after

    # -- native protocol hooks -------------------------------------------------
    def _prepare_generation_config(self, *args, **kwargs):
        return self.cfg, {}

    def _prepare_cache_for_generation(self, *args, **kwargs):
        return _Cache()

    def _prepare_sampler(self, cfg):
        return _Sampler(self)

    def _prepare_diffusion_stopping_criteria(self, cfg):
        return types.SimpleNamespace()

    def _prepare_logits_processor(self, cfg):
        return None

    def _prepare_encoder_inputs(self, ids, mask, pos, cache, first, block, batch):
        return ids[:, -pos.numel():], None

    def encode(self, input_ids, past_key_values=None, position_ids=None,
               attention_mask=None, **kwargs):
        length = int(input_ids.shape[1])
        self.encoder_calls.append((
            length,
            None if position_ids is None else position_ids[0].tolist(),
        ))
        if self.prompt_len is None:
            self.prompt_len = length
        past_key_values.length += length
        for layer in past_key_values.layers:
            layer.keys = torch.empty(1, 1, past_key_values.length, 1)
            layer.values = torch.empty(1, 1, past_key_values.length, 1)
        return types.SimpleNamespace(past_key_values=past_key_values)

    def __call__(self, decoder_input_ids, self_conditioning_logits=None,
                 decoder_attention_mask=None, decoder_position_ids=None,
                 past_key_values=None, **kwargs):
        q = int(decoder_input_ids.shape[1])
        positions = (torch.arange(q)[None] if decoder_position_ids is None
                     else decoder_position_ids)
        self.last_positions = positions.clone()
        self.decoder_calls.append(types.SimpleNamespace(
            q=q,
            ids=decoder_input_ids.clone(),
            positions=positions.clone(),
            mask=decoder_attention_mask,
            cache_length=past_key_values.length if past_key_values is not None else 0,
            sc=None if self_conditioning_logits is None else self_conditioning_logits.clone(),
        ))
        logits = torch.full((1, q, VOCAB), -5.0)
        target = 2 + (positions[0] % (VOCAB - 2))
        logits.scatter_(2, target[None, :, None], 10.0)
        return types.SimpleNamespace(logits=logits)

    def _denoising_step(self, decoder_forward, current_canvas, sampler,
                        argmax_canvas=None, input_ids=None,
                        decoder_position_ids=None, self_conditioning_logits=None,
                        mask_mapping=None, past_key_values=None,
                        finished_denoising=None, cur_step=None,
                        logits_processor=None, diffusion_stopping_criteria=None,
                        **kwargs):
        raw = decoder_forward().logits
        sampler.calls += 1
        if self.last_positions is None:
            raise RuntimeError("Fake denoising step has no projected positions")
        start = int(self.prompt_len) + sampler.ordinal * B
        local = [int(p) - start for p in self.last_positions[0]
                 if start <= int(p) < start + B]
        if any(not 0 <= row < B for row in local):
            raise RuntimeError("Fake model saw an impossible block-local row")

        ids = current_canvas.clone()
        accepted = sampler.accepted_token_mask.clone()
        if sampler.ordinal in self.accept_ordinals and sampler.calls == 1 and self.accept_first:
            new_rows = [row for row in local if not bool(accepted[0, row])]
            for row in new_rows[:self.accept_first]:
                accepted[0, row] = True
                ids[0, row] = 1 + sampler.ordinal * 7 + sampler.calls
        if sampler.ordinal in self.t2t_ordinals and sampler.calls == self.t2t_call:
            edited = [row for row in local if bool(accepted[0, row])]
            if edited:
                row = edited[0]
                ids[0, row] = 2 + (int(ids[0, row]) - 1) % (VOCAB - 3)
        sampler.accepted_token_mask = accepted

        sc = raw if raw.shape[1] == B else torch.zeros(1, B, VOCAB)
        finished = torch.tensor([
            self.finish_after is not None and sampler.calls >= self.finish_after
        ])
        return ids, ids.clone(), sc, finished


def plan(cap=768):
    return dict(block_size=B, cap=cap, flowblock_spawn_threshold=0.0, batch_size=1)


@dataclass
class OfferCall:
    tick: int
    head: int
    resident: int | None
    mandatory: tuple[int, ...] | None
    may_open: bool | None


@dataclass
class Run:
    ids: list[int]
    trace: list[dict]
    calls: list[OfferCall]
    model: FakeModel
    banks: list[_FakeWaveKV]


def run(engine: Engine, model: FakeModel, offers, *, cap=768, causal=True, seed=0) -> Run:
    values = tuple(offers)
    if not values:
        raise ValueError("offers cannot be empty")
    calls: list[OfferCall] = []
    first_bank = len(engine.banks)

    def next_budget(info):
        calls.append(OfferCall(
            getattr(info, "tick"), getattr(info, "head"),
            getattr(info, "resident", None), getattr(info, "mandatory", None),
            getattr(info, "may_open", None),
        ))
        return values[min(int(info.tick), len(values) - 1)]

    ids, trace = engine.generate(
        model, DATA, plan(cap), next_budget=next_budget, causal=causal, seed=seed,
    )
    return Run(ids, trace, calls, model, engine.banks[first_bank:])


# ---------------------------------------------------------------------------
# Trace validation.  These are the adversarial checks; they are deliberately
# independent of engine helper names and inspect only the public tape.
# ---------------------------------------------------------------------------


def _as_rows(value, field):
    if isinstance(value, int):
        raise AssertionError(f"{field}: expected row positions, got count {value}")
    return [int(row) for row in value]


def _per_block(value, count, field):
    if value is None:
        return [set() for _ in range(count)]
    seq = list(value)
    if len(seq) != count or any(not isinstance(part, (list, tuple, set)) for part in seq):
        raise AssertionError(f"{field}: expected {count} per-canvas row lists, got {value!r}")
    return [set(int(row) for row in part) for part in seq]


def _ages(value, count, field):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise AssertionError(f"{field}: expected {count} per-canvas ages, got {value!r}")
    if any(type(age) is not int or age < 0 for age in value):
        raise AssertionError(f"{field}: invalid ages {value!r}")
    return [int(age) for age in value]


def _visible_window(record, origin):
    visible = _as_rows(record["visible"], "visible")
    if visible != sorted(set(visible)):
        raise AssertionError("visible: rows are not sorted unique")
    head, count = int(record["head"]), len(record["ages_before"])
    expected = list(range(origin + head * B, origin + (head + count) * B))
    if visible != expected:
        raise AssertionError(
            f"visible: expected resident window {expected[0]}..{expected[-1]} "
            f"for head={head} canvases={count}, got {visible[:2]}..{visible[-2:]}")
    return visible


def _sampled_blocks(sampled, origin, head, count):
    blocks = set()
    for row in sampled:
        if not origin + head * B <= row < origin + (head + count) * B:
            raise AssertionError(f"sampled row {row} escapes the resident window")
        blocks.add((row - origin) // B - head)
    return blocks


def check_run(run_value: Run, *, causal=True):
    trace, model = run_value.trace, run_value.model
    if not trace:
        raise AssertionError("empty trace")
    if model.prompt_len is None:
        raise AssertionError("model never encoded the prompt")
    origin = int(model.prompt_len)
    if len(trace) != len(model.decoder_calls) or len(trace) != len(run_value.calls):
        raise AssertionError("trace, decoder calls and offer-clock calls disagree")

    pending: dict[int, set[int]] = {}
    next_encoded = int(trace[0]["head"])
    for tick, record in enumerate(trace):
        missing = REQUIRED - set(record)
        if missing:
            raise AssertionError(f"tick {tick}: trace misses {sorted(missing)}")
        if record["tick"] != tick:
            raise AssertionError(f"tick {record['tick']} != trace index {tick}")
        call = run_value.calls[tick]
        if call.tick != tick or call.head != record["head"]:
            raise AssertionError(f"tick {tick}: offer clock head/tick differs from trace")
        if record["head"] < 0:
            raise AssertionError("negative head ordinal")
        if tick == 0 and record.get("prefill") != origin:
            raise AssertionError("first trace row must charge the prompt prefill")
        if tick and record.get("prefill", 0) != 0:
            raise AssertionError("only the first trace row may report prefill")

        offer, q = record["offer"], record["q"]
        if type(offer) is not int or offer not in TIERS:
            raise AssertionError(f"tick {tick}: invalid offer {offer!r}")
        if type(q) is not int or not 0 < q <= 2 * B:
            raise AssertionError(f"tick {tick}: physical Q {q!r} outside (0, 2B]")
        if q != model.decoder_calls[tick].q:
            raise AssertionError(f"tick {tick}: trace q {q} != physical decoder q "
                                 f"{model.decoder_calls[tick].q}")
        selected = _as_rows(record["selected"], "selected")
        sampled = _as_rows(record["sampled"], "sampled")
        if selected != sorted(set(selected)) or len(selected) != q:
            raise AssertionError(f"tick {tick}: selected rows do not match physical Q")
        if sampled != sorted(set(sampled)) or not set(sampled) <= set(selected):
            raise AssertionError(f"tick {tick}: sampled rows are not a subset of selected")
        refresh = [row for row in selected if row not in set(sampled)]
        if _as_rows(record.get("refresh_only", refresh), "refresh_only") != refresh:
            raise AssertionError(f"tick {tick}: refresh_only is not selected - sampled")

        dirty_before = _per_block(record["dirty_before"], len(record["ages_before"]), "dirty_before")
        dirty_after = _per_block(record["dirty_after"], len(record["ages_after"]), "dirty_after")
        ages_before = _ages(record["ages_before"], len(dirty_before), "ages_before")
        ages_after = _ages(record["ages_after"], len(dirty_after), "ages_after")
        count = len(dirty_before)
        if count not in (1, 2):
            raise AssertionError(f"tick {tick}: expected one or two live canvases, got {count}")
        if call.resident is not None and call.resident != count:
            # The offer is consumed before a 512-spawned tail enters `live`.
            if not (offer == 512 and call.may_open and count == call.resident + 1):
                raise AssertionError(f"tick {tick}: offer clock resident count differs")
            if dirty_before[-1]:
                raise AssertionError(f"tick {tick}: newly spawned tail started dirty")
            if call.mandatory is not None and tuple(call.mandatory) != tuple(
                    len(rows) for rows in dirty_before[:call.resident]):
                raise AssertionError(f"tick {tick}: pre-spawn mandatory counts differ")
        elif call.mandatory is not None and tuple(call.mandatory) != tuple(
                len(rows) for rows in dirty_before):
            raise AssertionError(f"tick {tick}: offer clock mandatory counts differ")

        visible = _visible_window(record, origin)
        if not set(selected) <= set(visible):
            raise AssertionError(f"tick {tick}: selected rows escape visible KV")
        if record.get("overflow", max(0, q - offer)) != max(0, q - offer):
            raise AssertionError(f"tick {tick}: overflow disagrees with q-offer")

        # Mandatory M2T/T2T rows must be projected.  Sparse overflow is bounded
        # by those rows; a whole-canvas bootstrap/verify may also exceed offer.
        dirty_total = sum(len(rows) for rows in dirty_before)
        for index, rows in enumerate(dirty_before):
            absolute = {origin + (record["head"] + index) * B + row for row in rows}
            if not absolute <= set(selected):
                raise AssertionError(f"tick {tick}: dropped mandatory dirty rows "
                                     f"from canvas {index}")
        if q > offer and q != B and not (offer in (256, 512) and q in (256, 512)):
            if q - offer > dirty_total:
                raise AssertionError(
                    f"tick {tick}: overflow {q - offer} exceeds mandatory rows {dirty_total}")
        if q != B and len(sampled) > offer + dirty_total:
            raise AssertionError(
                f"tick {tick}: sampled {len(sampled)} rows above offer {offer} "
                f"plus {dirty_total} mandatory rows")

        sampled_blocks = _sampled_blocks(sampled, origin, int(record["head"]), count)
        if any(block != 0 for block in sampled_blocks) and offer != 512:
            raise AssertionError(f"tick {tick}: tail was sampled at offer {offer}")
        for index in range(count):
            delta = ages_after[index] - ages_before[index]
            if index in sampled_blocks:
                if delta not in (0, 1):
                    raise AssertionError(f"tick {tick}: sampled canvas {index} age delta {delta}")
            else:
                if delta != 0:
                    raise AssertionError(f"tick {tick}: parked canvas {index} advanced age")
                if dirty_after[index]:
                    raise AssertionError(
                        f"tick {tick}: parked canvas {index} retained dirty rows "
                        f"{sorted(dirty_after[index])}")

        ordinals = [int(record["head"]) + index for index in range(count)]
        for ordinal, rows in zip(ordinals, dirty_before, strict=True):
            if not pending.get(ordinal, set()) <= rows:
                raise AssertionError(
                    f"tick {tick}: canvas {ordinal} dirty_before does not include "
                    f"the prior frontier {sorted(set(pending.get(ordinal, set())) - rows)}")
        for ordinal, rows in zip(ordinals, dirty_after, strict=True):
            pending[ordinal] = rows
        for ordinal in record["encoded"]:
            if ordinal != next_encoded:
                raise AssertionError(f"tick {tick}: encoded {ordinal} out of order, "
                                     f"expected {next_encoded}")
            next_encoded += 1
            pending.pop(ordinal, None)

        if tick + 1 < len(trace) and record["encoded"]:
            if trace[tick + 1]["head"] != record["head"] + len(record["encoded"]):
                raise AssertionError(f"tick {tick}: encoded block did not advance head")
        promoted = record.get("promoted", [])
        if promoted:
            if not record["encoded"] or promoted != [record["head"] + 1]:
                raise AssertionError(f"tick {tick}: promotion bypassed encoded head")

    encoders = [length for length, _ in model.encoder_calls]
    if encoders[0] != origin:
        raise AssertionError("first encoder call is not the prompt prefill")
    if any(length != B for length in encoders[1:]):
        raise AssertionError("a commit re-encoded something other than one canvas")
    committed = sum(len(record["encoded"]) for record in trace)
    if len(encoders) - 1 != committed:
        raise AssertionError("encoder charge differs from encoded blocks")
    decoder_work = sum(call.q for call in model.decoder_calls)
    if decoder_work != sum(record["q"] for record in trace):
        raise AssertionError("observed decoder work differs from trace")

    if causal:
        # The real KV backend is responsible for the tensor mask.  The engine
        # must at least request a causal map for every physical projection.
        for bank in run_value.banks:
            if any(not begin.causal for begin in bank.begins):
                raise AssertionError("causal=True run requested a bidirectional KV map")
    return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_public_interface_and_clock(engine):
    signature = inspect.signature(engine.generate)
    assert [p.name for p in signature.parameters.values()] == [
        "model", "data", "plan", "next_budget", "causal", "seed"]
    assert signature.parameters["next_budget"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["next_budget"].default is inspect.Parameter.empty
    assert signature.parameters["causal"].default is True
    assert signature.parameters["seed"].default == 0

    result = run(engine, FakeModel(steps=6), (32, 64, 128, 256, 512), cap=1024)
    check_run(result)
    for tick, (call, record) in enumerate(zip(result.calls, result.trace, strict=True)):
        assert call.tick == tick and call.head == record["head"]
        if record["offer"] == 512 and call.may_open and len(record["ages_before"]) == 2:
            assert call.resident == 1
        else:
            assert call.resident == len(record["ages_before"])


def test_low_offers_project_exactly_the_offer_after_bootstrap(engine):
    model = FakeModel(steps=6)
    result = run(engine, model, (32,), cap=768)
    check_run(result)
    after_bootstrap = [record for record in result.trace[1:] if record["q"] != B]
    assert len(after_bootstrap) >= 2, "engine never entered sparse low-offer execution"
    for record in after_bootstrap:
        if record["q"] == B:
            continue
        assert record["q"] == 32, record
        assert len(record["sampled"]) == 32
        assert not record["refresh_only"]
        assert not any(record["dirty_before"])
        assert record["ages_after"][0] == record["ages_before"][0] + 1


def test_512_opens_a_second_real_canvas_instead_of_silently_using_256(engine):
    model = FakeModel(steps=6)
    result = run(engine, model, (512, 512), cap=1024)
    check_run(result)
    first, second = result.trace[0], result.trace[1]
    assert first["q"] == B and len(first["ages_before"]) == 1
    assert second["head"] == 0 and len(second["ages_before"]) == 2
    assert second["q"] == 2 * B, "offer 512 with a resident tail became q=256"
    assert len(second["selected"]) == 2 * B
    assert len(second["sampled"]) == 2 * B
    assert second["visible"] == list(range(model.prompt_len, model.prompt_len + 2 * B))
    assert second["ages_after"] == [first["ages_after"][0] + 1, 1]


def test_mandatory_tail_refresh_at_32_never_samples_the_parked_tail(engine):
    model = FakeModel(steps=6, accept_first=2, accept_ordinals={1})
    result = run(engine, model, (512, 512, 32, 32, 512), cap=1024)
    check_run(result)
    # The second 512-sampled tick dirties the parked tail.
    dirty_tick = next(i for i, record in enumerate(result.trace)
                      if len(record["ages_before"]) == 2 and record["dirty_after"][1])
    assert dirty_tick >= 1
    tail_dirty = list(result.trace[dirty_tick]["dirty_after"][1])
    assert tail_dirty

    found = None
    for record in result.trace[dirty_tick + 1:]:
        if len(record["ages_before"]) != 2:
            continue
        if not record["dirty_before"][1]:
            continue
        tail = {
            model.prompt_len + (record["head"] + 1) * B + row
            for row in record["dirty_before"][1]
        }
        if tail <= set(record["sampled"]):
            continue  # tail was sampled; not the required parked-refresh case
        found = record
        break
    assert found is not None, "no low-offer forward refreshed a dirty parked tail"

    assert found["offer"] == 32
    debt = len(found["dirty_before"][1])
    assert found["q"] == max(32, debt), "tail refresh and head must share the offer"
    assert len(found["sampled"]) == max(0, 32 - debt)
    assert found["ages_after"][1] == found["ages_before"][1]
    assert found["dirty_after"][1] == []
    assert set(found["refresh_only"]) >= {
        model.prompt_len + (found["head"] + 1) * B + row
        for row in found["dirty_before"][1]
    }
    assert not any(
        model.prompt_len + (found["head"] + 1) * B <= row
        < model.prompt_len + (found["head"] + 2) * B
        for row in found["sampled"]
    ), "parked tail rows were sampled"


def test_strict_t2t_tail_edit_is_refreshed_not_sampled(engine):
    model = FakeModel(steps=7, accept_first=2, accept_ordinals={1},
                      t2t_ordinals={1}, t2t_call=2)
    result = run(engine, model, (512, 512, 512, 32, 32), cap=1024)
    check_run(result)
    edited_tick = next(
        (i for i, record in enumerate(result.trace)
         if len(record["ages_before"]) == 2 and record["dirty_after"][1]),
        None,
    )
    assert edited_tick is not None, "fake never produced a strict tail edit"
    refresh = next(
        (record for record in result.trace[edited_tick + 1:]
         if len(record["ages_before"]) == 2 and record["dirty_before"][1]
         and record["ages_after"][1] == record["ages_before"][1]),
        None,
    )
    assert refresh is not None, "strict T2T dirty tail was never refreshed"
    assert refresh["offer"] < 256
    assert refresh["ages_after"][1] == refresh["ages_before"][1]
    assert refresh["dirty_after"][1] == []
    assert not any(
        model.prompt_len + (refresh["head"] + 1) * B <= row
        < model.prompt_len + (refresh["head"] + 2) * B
        for row in refresh["sampled"]
    )


def test_mandatory_head_overflow_is_charged_not_truncated(engine):
    model = FakeModel(steps=6, accept_first=40, accept_ordinals={0})
    result = run(engine, model, (256, 32, 32), cap=768)
    check_run(result)
    first_sparse = next(record for record in result.trace[1:] if record["q"] != B)
    assert first_sparse["offer"] == 32
    assert first_sparse["q"] == 40
    assert first_sparse["overflow"] == 8
    assert len(first_sparse["sampled"]) == 40
    # Every mandatory row is present even though 40 > the nominal 32.
    mandatory = {
        model.prompt_len + first_sparse["head"] * B + row
        for row in first_sparse["dirty_before"][0]
    }
    assert mandatory <= set(first_sparse["selected"])


def test_shared_budget_head_optional_fits_after_tail_debt(engine):
    """Tail refresh is charged first; optional head rows take only the rest."""
    model = FakeModel(steps=8, accept_first=20, accept_ordinals={1})
    result = run(engine, model, (512, 512, 32), cap=1024)
    check_run(result)
    event = next(record for record in result.trace
                 if record["offer"] == 32 and len(record["ages_before"]) == 2)
    assert event["dirty_before"][0] == []
    assert len(event["dirty_before"][1]) == 20
    assert event["q"] == 32, "head optional rows ignored the tail-refresh debt"
    assert event["overflow"] == 0
    assert len(event["sampled"]) == 12, "head sampled more than offer minus tail debt"
    tail_start = model.prompt_len + (event["head"] + 1) * B
    tail_rows = set(range(tail_start, tail_start + B))
    assert not tail_rows & set(event["sampled"])
    assert tail_rows & set(event["refresh_only"])
    assert event["ages_after"][1] == event["ages_before"][1]
    assert event["dirty_after"][1] == []


def test_tail_that_consumes_offer_parks_a_clean_head(engine):
    """Debt = offer: no optional head rows and no forced 256-row verify."""
    model = FakeModel(steps=8, accept_first=32, accept_ordinals={1})
    result = run(engine, model, (512, 512, 32, 32), cap=1024)
    check_run(result)
    low = [record for record in result.trace
           if record["offer"] == 32 and len(record["ages_before"]) == 2]
    assert low
    first = low[0]
    assert first["dirty_before"][0] == []
    assert len(first["dirty_before"][1]) == 32
    assert first["q"] == 32, "tail already consumed the whole shared offer"
    assert first["q"] != B, "gratuitous full-canvas verify under a shared 32 budget"
    assert first["sampled"] == [], "clean head was sampled after full tail debt"
    assert first["ages_after"][0] == first["ages_before"][0]
    assert first["ages_after"][1] == first["ages_before"][1]
    assert first["dirty_after"][1] == []
    assert len(first["refresh_only"]) == 32

    # Once the tail is clean the whole offer is available to the head again.
    second = low[1]
    assert second["dirty_before"] == [[], []]
    assert second["q"] == 32
    assert len(second["sampled"]) == 32
    assert second["ages_after"][0] == second["ages_before"][0] + 1
    assert second["ages_after"][1] == second["ages_before"][1]


def test_sparse_refresh_is_not_dense_padding(engine):
    """At offer 256 a dirty tail is refreshed row-wise, not as a padded block."""
    from bdlm_serving.runtime.diffusiongemma_wave import generate as wave_generate
    offers = (512, 512, 256)
    unified_model = FakeModel(steps=6, accept_first=2, accept_ordinals={1})
    unified = run(engine, unified_model, offers, cap=1024)
    check_run(unified)

    wave_model = FakeModel(steps=6, accept_first=2, accept_ordinals={1})
    wave_trace = wave_generate(
        wave_model, DATA, "external_w2", plan(1024),
        next_budget=lambda state: offers[min(state.tick, len(offers) - 1)],
    )[1]
    assert len(wave_trace) == len(unified.trace)
    for tick, (dense, sparse) in enumerate(zip(wave_trace, unified.trace, strict=True)):
        assert dense["head"] == sparse["head"]
        assert dense["offer"] == sparse["offer"]
    event = next(
        record for record in unified.trace
        if record["offer"] == 256 and len(record["ages_before"]) == 2
        and record["dirty_before"][1]
    )
    dense = wave_trace[unified.trace.index(event)]
    assert dense["q"] == 2 * B, "dense reference did not project the whole dirty tail"
    assert dense["sample_width"] == 1
    assert event["q"] < dense["q"], "unified path silently fell back to dense padding"
    assert event["q"] == 256 + len(event["dirty_before"][1])
    assert event["ages_after"][1] == event["ages_before"][1]
    assert event["dirty_after"][1] == []
    assert dense["ages_after"][1] == dense["ages_before"][1]


def test_switching_sparse_and_dense_keeps_canvas_and_cache(engine):
    model = FakeModel(steps=6)
    offers = (32, 512, 64, 512, 128)
    result = run(engine, model, offers, cap=1024)
    check_run(result)
    # One sampler per opened canvas ordinal, never a re-created ordinal.
    assert [sampler.ordinal for sampler in model.samplers] == list(range(len(model.samplers)))
    early = result.trace[:len(offers)]
    assert not any(record["encoded"] for record in early)
    # Switching itself never re-encodes; encoders only appear after a commit.
    assert [record["q"] for record in early] == [B, 2 * B, 64, 2 * B, 128]
    assert [length for length, _ in model.encoder_calls][0] == model.prompt_len
    encoded_ticks = [i for i, record in enumerate(result.trace) if record["encoded"]]
    if encoded_ticks:
        assert all(i >= len(offers) for i in encoded_ticks)


def test_encoder_charge_on_commit(engine):
    model = FakeModel(steps=2)
    result = run(engine, model, (512, 512, 512), cap=768)
    check_run(result)
    encoded = [ordinal for record in result.trace for ordinal in record["encoded"]]
    assert encoded, "short-cap run never committed a canvas"
    assert [length for length, _ in model.encoder_calls] == [model.prompt_len] + [B] * len(encoded)
    assert sum(record["q"] for record in result.trace) == sum(
        call.q for call in model.decoder_calls)


def test_causal_kv_backend_receives_exact_selected_rows(engine):
    model = FakeModel(steps=6)
    result = run(engine, model, (512, 512, 32), cap=1024, causal=True)
    check_run(result)
    assert result.banks, "engine did not use the KV backend"
    assert len(result.banks) == 1
    bank = result.banks[0]
    assert len(bank.begins) == len(result.trace)
    for record, begin in zip(result.trace, bank.begins, strict=True):
        assert begin.origin == model.prompt_len
        assert begin.causal is True
        assert begin.selected == record["selected"]
    # A causal run must never ask for bidirectional visibility.
    assert all(begin.causal for begin in bank.begins)


@pytest.mark.parametrize("causal", [True, False])
def test_causal_flag_is_forwarded(engine, causal):
    result = run(engine, FakeModel(steps=6), (512, 512), cap=1024, causal=causal)
    check_run(result, causal=causal)
    assert result.banks and all(begin.causal is causal for begin in result.banks[0].begins)


@pytest.mark.parametrize("bad", [0, 31, 33, 129, 513, -32, True, 32.0, "32"])
def test_unsupported_offers_fail_closed(engine, bad):
    model = FakeModel(steps=6)

    def source(info):
        return bad

    with pytest.raises((ValueError, TypeError)):
        engine.generate(model, DATA, plan(1024), next_budget=source)


def test_invalid_plan_and_request_fail_closed(engine):
    with pytest.raises(ValueError):
        engine.generate(FakeModel(steps=6), DATA,
                        dict(block_size=128, cap=768, flowblock_spawn_threshold=0.0),
                        next_budget=lambda info: 32)
    with pytest.raises(ValueError):
        engine.generate(FakeModel(steps=6), DATA,
                        dict(block_size=B, cap=768, flowblock_spawn_threshold=1.5),
                        next_budget=lambda info: 32)
    padded = {"input_ids": torch.tensor([[5, 6, 7]]),
              "attention_mask": torch.tensor([[1, 0, 1]])}
    with pytest.raises(ValueError):
        engine.generate(FakeModel(steps=6), padded, plan(768), next_budget=lambda info: 32)
    with pytest.raises(ValueError):
        engine.generate(FakeModel(steps=6), DATA, plan(768),
                        next_budget=lambda info: 32, causal="yes")


def test_trace_corruption_is_rejected(engine):
    """The local audit must catch the failure modes the suite targets."""
    model = FakeModel(steps=6)
    result = run(engine, model, (512, 512, 32), cap=1024)
    check_run(result)

    bad = [dict(record) for record in result.trace]
    bad[1]["q"] += 1
    with pytest.raises(AssertionError):
        check_run(Run(result.ids, bad, result.calls, result.model, result.banks))

    bad = [dict(record) for record in result.trace]
    bad[1]["ages_after"] = list(bad[1]["ages_after"])
    bad[1]["ages_after"][1] += 1
    with pytest.raises(AssertionError):
        check_run(Run(result.ids, bad, result.calls, result.model, result.banks))

    bad = [dict(record) for record in result.trace]
    bad[1]["dirty_after"] = [list(rows) for rows in bad[1]["dirty_after"]]
    bad[1]["dirty_after"][1] = [0]
    with pytest.raises(AssertionError):
        check_run(Run(result.ids, bad, result.calls, result.model, result.banks))
