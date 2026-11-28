"""Opt-in, single-request physical compaction for the pinned LLaDA21 Flow W2."""
import ast
from hashlib import sha256
from inspect import unwrap
from types import MethodType, SimpleNamespace

import torch
from . import control

PACKED = "7445d6865bfd9199795be8535546a49e1f9452f4a8b900e10317d5c6256b74ac"
B = 32


def require(ok, message):
    if not ok:
        raise ValueError(message)


class _ShortTail(control.TailCache):
    """Leave masked key columns in place, but never project or scatter filler."""
    def __init__(self, data):
        super().__init__(data, 2 * B)

    def update(self, k, v, layer, cache_kwargs=None):
        require(k.shape[-2] == v.shape[-2] == B, "unopened slot reached KV projection")
        pair = [torch.cat((self.data[layer, j, ..., :self.prefix, :], t,
                           torch.zeros_like(t)), dim=-2) for j, t in enumerate((k, v))]
        self.present[2*layer:2*layer+2] = pair
        return tuple(pair)


class _Backend(control.NativeBackend):
    def __init__(self, native, engine):
        super().__init__(native)
        self.engine = engine

    def __call__(self, unused, **kw):
        tick = self.engine.ticks[-1]
        take = tick["physical_width"]
        if take == 2*B:
            return super().__call__(unused, **kw)
        p = kw["position_ids"][:, :take].contiguous()
        mask = kw["attention_mask"][:, :take].contiguous()
        data = kw["past_key_values"]._data
        cache = _ShortTail(data)
        out = self.native(inputs_embeds=kw["inputs_embeds"][:, :take].contiguous(),
                          position_ids=p, attention_mask=mask[:, None], use_cache=True,
                          past_key_values=cache, output_attentions=True)
        self.events.append(dict(phase="flowblock_packed", q_tokens=take, physical_forwards=1,
                                max_physical_q=take, kv_length=data.shape[4],
                                query_rows_over_layers=take*len(self.native.model.layers),
                                physical_layer_attention_calls=len(self.native.model.layers),
                                absolute_positions=p[0].tolist()))
        return SimpleNamespace(logits=out.logits, past_key_values=cache.present)


class ActiveW2(control.FlowBlockControl):
    """This subclass changes no stock entry point. compact=False instruments true W2."""
    def __init__(self, native, mask_id, eos_id, config, *, compact):
        super().__init__(native, 2, mask_id, eos_id, config)
        self.ticks = []
        self.compact = compact
        self.backend = _Backend(native, self)
        self.owner.model = self.backend
        self.owner._observe_geometry = self._geometry
        old_write = self.owner._packed_kv_writeback
        old_joint = self.owner._joint_decode_slot_packed

        def scatter(data, present, dst, tail_start, n):
            t = self.ticks[-1]
            take = t["physical_width"]
            require(n == 64 and tail_start == data.shape[4]-n and
                    dst[0].tolist() == list(range(t["head"], t["head"]+n)),
                    "virtual address aliases a live write")
            old_write(data, present, dst[:, :take], tail_start, take)
            t["written_KV"] = take

        def joint(logits, tokens, pm, post, finished, active):
            t = self.ticks[-1]
            require(len(t["slots"]) < t["open"] and logits.shape[1] == B and bool(active.item()),
                    "unmeasured slot entered decoder")
            result = old_joint(logits, tokens, pm, post, finished, active)
            new, fin, pe, frontier = result
            t["slots"].append(dict(ids=tokens[0].tolist(), prompt=pm[0].tolist(), new=new[0].tolist(),
                                   fin0=bool(finished.item()), fin=bool(fin.item()),
                                   pe0=post.item(), pe=pe.item(), frontier=frontier.item()))
            return result
        self.owner._packed_kv_writeback = scatter
        self.owner._joint_decode_slot_packed = joint
        # Read the authoritative live pointers immediately before the original call.
        path = control.control_directory("FlowBlock") / "flowblock_packed.py"
        require(sha256(path.read_bytes()).hexdigest() == PACKED, "unknown FlowBlock source")
        orig = self.generate.__func__
        fn, = [n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef)
               and n.name == orig.__name__]
        loop, = [n for n in fn.body if isinstance(n, ast.While)]
        index, = [j for j, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                  and ast.unparse(n.targets[0]) == "output" and isinstance(n.value, ast.Call)
                  and ast.unparse(n.value.func) == "self.model"]
        loop.body.insert(index, ast.parse("self._observe_geometry(o_start, commit_ptr, right_ptr, "
                                          "tail_start, kvpos, attn_mask, seq_idx, total_length)").body[0])
        ns = dict(unwrap(orig).__globals__)
        exec(compile(ast.fix_missing_locations(ast.Module([fn], [])), str(path), "exec"), ns)
        self.generate = MethodType(ns[orig.__name__], self.owner)

    def _geometry(self, head, commit, right, tail, keys, mask, pos, length):
        start, c, r = head.item(), commit.item(), right.item()
        live = r-c
        require(1 <= live <= 2 and tail >= start and start % B == 0 and
                keys.shape[-1] == tail+2*B and tuple(mask.shape) == (1, 2*B, tail+2*B),
                "invalid open prefix")
        if self.ticks:
            prev = self.ticks[-1]
            require(0 <= c-prev["commit"] <= prev["open"] and r >= prev["right"] and
                    start-prev["head"] == B*(c-prev["commit"]) and
                    len(prev["slots"]) == prev["open"], "committed an unopened slot")
        for s in range(live):
            expected = (keys < start) | ((keys >= tail) & (keys < tail+(s+1)*B))
            require(bool((mask[:, s*B:(s+1)*B] == expected).all()), "unexpected dependency")
        require(pos[0, :live*B].tolist() == list(range(start, start+live*B)) and
                start+live*B <= length, "active slot alias")
        self.ticks.append(dict(head=start, commit=c, right=r, open=live,
                               physical_width=(live if self.compact else 2)*B, slots=[]))
