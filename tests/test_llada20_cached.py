from types import SimpleNamespace

import pytest
import torch

from bdlm_serving.runtime.llada20_cached import cached_generate


class Trace(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.model.layers[0].attn = torch.nn.Module()
        self.model.layers[0].attn.query_key_value = torch.nn.Identity()
        self.reads = []

    def forward(self, *, input_ids, position_ids, attention_mask, past_key_values, use_cache):
        q = input_ids.shape[1]
        end = int(position_ids[0, -1])+1
        assert use_cache and past_key_values.get_seq_length() == end-q
        k = torch.arange(end)[None, :]
        assert torch.equal(attention_mask[0, 0], position_ids.T//32 >= k//32)
        v = self.model.layers[0].attn.query_key_value(input_ids[..., None].float())[:, None]
        prior, _ = past_key_values.update(v, v, 0)
        self.reads.append((q, prior[0, 0, :, 0].long().tolist()))
        return SimpleNamespace(past_key_values=past_key_values)

    def generate(self, *, inputs, mask_id, **kwargs):
        p = inputs.shape[1]
        start = p//32*32
        x = torch.full((1, start+64), mask_id)
        x[:, :p] = inputs
        pos = torch.arange(x.shape[1])[None, :]
        bias = (pos.T//32 >= pos//32).float().log()[None, None]
        for lo in (start, start+32):
            hi = lo+32
            for val in (11, 12):
                self.forward(x[:, :hi], position_ids=pos[:, :hi], attention_mask=bias[:, :, :hi, :hi])
                cur = x[:, lo:hi]
                if val == 11:
                    cur[:, -1] = val
                else:
                    cur[cur == mask_id] = val
        return x[:, p:]


@pytest.mark.parametrize("p", [1, 32, 63, 64, 75])
def test_final_values_and_actual_projection_counts(p):
    fake = Trace()
    prompt = torch.full((1, p), 7)
    facade = SimpleNamespace(calls=[])
    out, events, meta = cached_generate(fake, facade, prompt, 2048, 17, 99, 13)
    prefix = p//32*32
    assert torch.equal(out[:, :p], prompt)
    assert meta["phase_forwards"]["commit"] == 2 and meta["phase_forwards"]["denoise"] == 4
    assert meta["phase_q"].get("prefill", 0) == prefix
    expected = (["prefill"] if prefix else []) + ["denoise", "denoise", "commit"]*2
    assert [e["phase"] for e in events] == expected
    assert [c["rows"] for c in facade.calls] == [q for q, _ in fake.reads]
    for e, (_, vals) in zip(events, fake.reads):
        assert len(vals) == e["kv"]
        if e["phase"] == "commit":
            assert vals == out[0, :len(vals)].tolist()
        if len(vals) > prefix+32:
            assert vals[:prefix+32] == out[0, :prefix+32].tolist()
