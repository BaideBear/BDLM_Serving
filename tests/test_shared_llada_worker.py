"""Shard runtime: guards, prompt identity, charge audit and result mapping. No inference."""
import json
import sys
import types
from pathlib import Path

import pytest

from experiments.t2t.serving_slots import shared_llada_worker as w


class Row:
    def __init__(self, ids):
        self.ids = list(ids)

    def __getitem__(self, index):
        return self

    def tolist(self):
        return list(self.ids)


class Tok:
    eos_token_id, all_special_ids = 2, [1, 2]

    def __init__(self, ids, framed=None):
        self.ids, self.framed, self.calls = list(ids), framed, []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "CHAT:" + messages[0]["content"]

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        self.calls.append(dict(text=text, add_special_tokens=add_special_tokens,
                               return_tensors=return_tensors))
        assert add_special_tokens is False and return_tensors == "pt"
        return types.SimpleNamespace(input_ids=Row(self.ids))

    def convert_tokens_to_ids(self, token):
        return 5


def test_fingerprint_tracks_small_file_content(tmp_path):
    (tmp_path / "config.json").write_text('{"a":1}')
    (tmp_path / "weights.bin").write_bytes(b"x" * 32)
    first = w.fingerprint(tmp_path)
    assert first == w.fingerprint(tmp_path)                      # deterministic
    (tmp_path / "config.json").write_text('{"a":2}')
    assert w.fingerprint(tmp_path) != first                      # metadata is hashed
    big = tmp_path / "shard.bin"
    big.write_bytes(b"y" * (w._SMALL + 1))
    sized = w.fingerprint(tmp_path)
    big.write_bytes(b"z" * (w._SMALL + 1))
    assert w.fingerprint(tmp_path) == sized                      # weights only sized
    with pytest.raises(ValueError, match="missing"):
        w.fingerprint(tmp_path / "nope")


def test_reference_root_requires_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("BDLM_LLADA_REFERENCE", raising=False)
    with pytest.raises(ValueError, match="BDLM_LLADA_REFERENCE"):
        w.reference_root()
    monkeypatch.setenv("BDLM_LLADA_REFERENCE", str(tmp_path / "absent"))
    with pytest.raises(ValueError, match="BDLM_LLADA_REFERENCE"):
        w.reference_root()
    monkeypatch.setenv("BDLM_LLADA_REFERENCE", str(tmp_path))
    assert w.reference_root() == tmp_path


def test_load_refuses_cpu_and_unpinned_devices(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is required"):
        w.load(Path("/nonexistent"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="exactly one visible GPU"):
        w.load(Path("/nonexistent"))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        w.load(Path("/nonexistent"))


def test_canonical_prompt_uses_the_project_chat_template_rule():
    ref = [1, 7, 8, 2]
    tok = Tok(ref)
    ids, how = w.canonical_prompt(tok, dict(index=0, prompt="hi"), ref)
    assert ids == ref and how == "chat_template"
    assert tok.calls == [dict(text="CHAT:hi", add_special_tokens=False, return_tensors="pt")]
    with pytest.raises(RuntimeError, match="prompt token drift"):
        w.canonical_prompt(Tok([1, 7, 3, 2]), dict(index=3, prompt="hi"), ref)


def ev(q, rows, dirty=(), cold=(), mask=(), prefix=()):
    return dict(q=q, rows=list(rows), dirty=list(dirty), cold=list(cold), mask=list(mask),
                prefix=list(prefix), layer_physical_q=[q] * 20)


def test_roles_partition_each_forward_and_the_total():
    trace = [ev(6, range(32, 38), dirty=[32, 33], cold=[36], mask=[36, 37], prefix=[34, 35])]
    assert w.roles_of(trace) == dict(dirty_write=2, uninitialized_mask=1, resident_mask=1,
                                     clean_resolved=2)
    assert sum(w.roles_of(trace).values()) == 6
    with pytest.raises(ValueError, match="do not partition"):
        w.roles_of([ev(6, range(32, 37))])                       # q disagrees with rows


def complete_result(offer=16, failure=None):
    trace = [ev(6, range(32, 38), dirty=[32], cold=[36], mask=[36, 37], prefix=[33,34,35]),
             ev(offer, range(64, 64 + offer), mask=list(range(64, 64 + offer)))]
    # Exactly the key set _fixed returns (llada_fixed_family_eval lines 179-194).
    return dict(index=0, dataset="gsm8k", task_id="gsm8k/1", seed=1, q=offer, law=[1, 1],
                input_sha256="deadbeef", output_ids=[1, 2], score={"correct": True}, correct=True,
                failure=failure, N=7, F=len(trace), Q=sum(e["q"] for e in trace),
                TPF=7 / len(trace), efficiency=7 / 22, max_decoder_Q=max(e["q"] for e in trace),
                projection_modules=20, decoder_Q=sum(e["q"] for e in trace), encoder_Q=0,
                prefill_Q=9, row_visits={}, phase_F={}, trace=trace)


def handle():
    class Model:
        device = types.SimpleNamespace(type="cuda")
    return dict(model=Model(), tok=Tok([1, 2]), checkpoint="/ckpt", fingerprint="f",
                scorer=w.SCORER_SHA, mask_id=5, eos_id=2, prefill_F=1)


def prepare(monkeypatch, tmp_path, result):
    ref = tmp_path / "00-native32.json"
    ref.write_text(json.dumps(dict(index=0, generation_cap=w.CAP, prompt_ids=[1, 2])))
    monkeypatch.setenv("BDLM_LLADA_REFERENCE", str(tmp_path))
    monkeypatch.setattr(w, "evaluate_fixed", lambda *a, **k: result)
    return dict(index=0, dataset="gsm8k", task_id="gsm8k/1", seed=1, prompt="hi", reference="2")


def test_evaluate_maps_a_complete_result_with_diag_roles(monkeypatch, tmp_path):
    from bdlm_serving.methods.t2t.unified_rows import Params
    task = prepare(monkeypatch, tmp_path, complete_result())
    out = w.evaluate(handle(), task, 16, Params())
    assert out["status"] == w.STATUS_COMPLETE and out["correct"] is True
    assert (out["N"], out["F"], out["Q"]) == (7, 2, 22)
    assert out["decoder_F"] == 2 and out["encoder_F"] == 0 and out["prefill_Q"] == 9
    assert out["prompt_encoding"] == "chat_template" and out["forwards"] == 2
    assert sum(out["roles"].values()) == out["Q"]


def test_documented_nonconvergent_failure_is_a_valid_algorithm_failure(monkeypatch, tmp_path):
    from bdlm_serving.methods.t2t.unified_rows import Params
    task = prepare(monkeypatch, tmp_path, complete_result(failure=w.NONCONVERGENT))
    out = w.evaluate(handle(), task, 16, Params())
    assert out["status"] == w.STATUS_ALGORITHM_FAILURE and out["correct"] is False
    task = prepare(monkeypatch, tmp_path, complete_result(failure="CUDA error"))
    with pytest.raises(ValueError, match="unexpected failure"):
        w.evaluate(handle(), task, 16, Params())


def test_evaluate_refuses_bad_offer_charge_and_cap(monkeypatch, tmp_path):
    from bdlm_serving.methods.t2t.unified_rows import Params
    task = prepare(monkeypatch, tmp_path, complete_result())
    with pytest.raises(ValueError, match="offer must be one of"):
        w.evaluate(handle(), task, 12, Params())
    bad = complete_result()
    bad["trace"][1]["layer_physical_q"] = [15] * 20
    monkeypatch.setattr(w, "evaluate_fixed", lambda *a, **k: bad)
    with pytest.raises(ValueError, match="per-layer charge"):
        w.evaluate(handle(), task, 16, Params())
    over = complete_result()
    over["max_decoder_Q"] = 17
    monkeypatch.setattr(w, "evaluate_fixed", lambda *a, **k: over)
    with pytest.raises(ValueError, match="above the offer"):
        w.evaluate(handle(), task, 16, Params())
    ref = tmp_path / "00-native32.json"
    ref.write_text(json.dumps(dict(index=0, generation_cap=1024, prompt_ids=[1, 2])))
    monkeypatch.setattr(w, "evaluate_fixed", lambda *a, **k: complete_result())
    with pytest.raises(ValueError, match="capped at"):
        w.evaluate(handle(), task, 16, Params())


def make_shards(tmp_path, n=2):
    reg = {}
    for i in range(n):
        name = f"model-0000{i}-of-0000{n}.safetensors"
        data = bytes([i]) * (w._SMALL + 17)
        (tmp_path / name).write_bytes(data)
        reg[name] = w.stream_sha256(tmp_path / name)
    return reg


def test_weight_manifest_rejects_malformed_entries(tmp_path):
    bad = tmp_path / "m.json"
    bad.write_text('{"a": "not-a-digest"}')
    with pytest.raises(ValueError, match="not a sha256"):
        w.weight_manifest(bad)
    bad.write_text("[]")
    with pytest.raises(ValueError, match="non-empty object"):
        w.weight_manifest(bad)
    with pytest.raises(ValueError, match="missing weight manifest"):
        w.weight_manifest(tmp_path / "nope.json")


def test_verify_weights_streams_and_fails_closed(tmp_path):
    reg = make_shards(tmp_path)
    assert w.verify_weights(tmp_path, reg) == reg
    name = sorted(reg)[0]
    data = bytearray((tmp_path / name).read_bytes())
    data[-1] ^= 0xFF
    (tmp_path / name).write_bytes(bytes(data))
    with pytest.raises(ValueError) as err:
        w.verify_weights(tmp_path, reg)
    assert name in str(err.value) and "digest mismatch" in str(err.value)
    (tmp_path / name).unlink()
    with pytest.raises(ValueError, match="missing registered weight shards"):
        w.verify_weights(tmp_path, reg)


def test_stream_sha256_never_reads_the_whole_file(monkeypatch, tmp_path):
    shard = tmp_path / "s.bin"
    shard.write_bytes(b"q" * (w._SMALL + 3))
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(AssertionError("read_bytes used")))
    import hashlib
    assert w.stream_sha256(shard) == hashlib.sha256(b"q" * (w._SMALL + 3)).hexdigest()


def test_load_verifies_weights_before_loading_the_model(monkeypatch, tmp_path):
    import torch
    import bdlm_serving.evaluation.scoring as sc
    from hashlib import sha256 as _sha
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")
    monkeypatch.setattr(w, "SCORER_SHA", _sha(Path(sc.__file__).read_bytes()).hexdigest())
    reg = make_shards(tmp_path)
    called = []
    monkeypatch.setattr(w, "_load_gpu", lambda *a, **k: called.append(1) or (_ for _ in ()).throw(
        AssertionError("model loaded")))
    name = sorted(reg)[0]
    (tmp_path / name).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        w.load(tmp_path, manifest=reg)
    assert called == []                                   # never reached the loader
    good = make_shards(tmp_path)
    with pytest.raises(AssertionError, match="model loaded"):
        w.load(tmp_path, manifest=good)                   # weights ok -> loader reached
    assert called == [1]


def test_fingerprint_includes_the_registered_weight_digests(tmp_path):
    reg = make_shards(tmp_path)
    base = w.fingerprint(tmp_path, reg)
    other = dict(reg)
    other[sorted(reg)[0]] = "0" * 64
    assert w.fingerprint(tmp_path, other) != base


def test_realistic_fixed_row_matches_every_key_the_worker_reads(monkeypatch, tmp_path):
    from bdlm_serving.methods.t2t.unified_rows import Params
    row = complete_result()
    assert set(w.REQUIRED) <= set(row)                 # worker only reads real keys
    task = prepare(monkeypatch, tmp_path, row)
    out = w.evaluate(handle(), task, 16, Params())
    assert out["encoder_F"] == 0 and out["prefill_F"] == 1 and out["decoder_F"] == row["F"]
    broken = dict(row)
    broken.pop("prefill_Q")
    monkeypatch.setattr(w, "evaluate_fixed", lambda *a, **k: broken)
    with pytest.raises(ValueError, match="missing expected keys"):
        w.evaluate(handle(), task, 16, Params())
    enc = dict(row, encoder_Q=1)
    monkeypatch.setattr(w, "evaluate_fixed", lambda *a, **k: enc)
    with pytest.raises(ValueError, match="no encoder forward"):
        w.evaluate(handle(), task, 16, Params())
