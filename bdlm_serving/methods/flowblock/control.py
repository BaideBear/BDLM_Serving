"""Official FlowBlock packed control on the common native backend.

SDAR can exclude immutable prompt tokens from the EOS completion guard.
This isolates request-control/TPF. It is not official fused-backend speed.
"""
import ast
import copy
import hashlib
from pathlib import Path
import time
import types

import torch
from transformers.cache_utils import DynamicCache

from bdlm_serving.upstream import control_directory
SOURCE_COMMIT = "8f730a2173140792a4324736efdcba27a2bdee75"


class PromptEosGuard(ast.NodeTransformer):
    """Change only the completion test's lower bound, never the token data."""
    def __init__(self):
        self.changes = 0

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id == "has_eos" for t in node.targets):
            candidates = [n for n in ast.walk(node.value)
                          if isinstance(n, ast.Subscript)
                          and isinstance(n.slice, ast.Tuple)
                          and len(n.slice.elts) == 2
                          and isinstance(n.slice.elts[1], ast.Slice)]
            assert len(candidates) == 1
            part = candidates[0].slice.elts[1]
            assert ast.unparse(part.lower) == "loc.start"
            assert ast.unparse(part.upper) == "loc.end"
            part.lower = ast.Call(func=ast.Name(id="max", ctx=ast.Load()),
                                  args=[part.lower, ast.Name(id="prompt_length", ctx=ast.Load())],
                                  keywords=[])
            self.changes += 1
        return self.generic_visit(node)


def extract(path, names, namespace, ignore_prompt_eos=False):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in selected} == set(names)
    if ignore_prompt_eos:
        guard = PromptEosGuard()
        selected = [guard.visit(node) for node in selected]
        assert guard.changes == 1
    module = types.ModuleType("extracted_flowblock")
    module.__dict__.update(namespace)
    code = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(code, str(path), "exec"), module.__dict__)
    return module


class TailCache(DynamicCache):
    def __init__(self, data, query):
        super().__init__()
        self.data = data
        self.prefix = data.shape[4]-query
        self.present = [None]*(data.shape[0]*2)

    def get_seq_length(self, layer_idx=0):
        return self.prefix

    def update(self, k, v, layer, cache_kwargs=None):
        keys = torch.cat((self.data[layer, 0, :, :, :self.prefix], k), dim=-2)
        values = torch.cat((self.data[layer, 1, :, :, :self.prefix], v), dim=-2)
        self.present[2*layer:2*layer+2] = [keys, values]
        return keys, values


class Buffer:
    def __init__(self, model):
        self.model = model
        self.past_key_values = types.SimpleNamespace(_data=None)

    def extend_cache(self, length):
        old = self.past_key_values._data
        assert old is not None
        if length > old.shape[4]:
            data = old.new_zeros((*old.shape[:4], length, old.shape[-1]))
            data[..., :old.shape[4], :].copy_(old)
            self.past_key_values._data = data


class NativeBackend:
    def __init__(self, native):
        self.native = native
        self.device = native.device
        self.events = []

    def __call__(self, unused, *, inputs_embeds, position_ids, use_cache,
                 past_key_values, attention_mask, replace_position):
        query = inputs_embeds.shape[1]
        data = past_key_values._data
        cache = TailCache(data, query)
        out = self.native(inputs_embeds=inputs_embeds, position_ids=position_ids,
                          attention_mask=attention_mask[:, None], use_cache=True,
                          past_key_values=cache, output_attentions=True)
        self.events.append({
            "phase": "flowblock_packed", "q_tokens": query, "max_physical_q": query,
            "physical_forwards": 1, "kv_length": data.shape[4],
            "query_rows_over_layers": query*len(self.native.model.layers),
            "physical_layer_attention_calls": len(self.native.model.layers),
            "absolute_positions": position_ids[0].tolist(),
        })
        return types.SimpleNamespace(logits=out.logits, past_key_values=cache.present)


class FlowBlockControl:
    def __init__(self, native, window, mask_id, eos_id, config, gate="acceptance_ratio",
                 ignore_prompt_eos=False, *, block_size=32):
        if type(block_size) is not int or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        self.block_size = block_size
        self.backend = NativeBackend(native)
        self.raw_tokens = None
        base = control_directory("FlowBlock")
        util = extract(base / "utils.py", ["TokenArray", "BlockIterator", "BlockLoc"],
                       {"torch": torch, "copy": copy})
        helpers = extract(base / "generate_uniform.py", ["_contiguous"], {"torch": torch})
        original = util.TokenArray
        controller = self

        class ObservedTokenArray(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                controller.raw_tokens = self

        methods = ["_joint_decode_slot_packed", "_spawn_signal_per_row_packed",
                   "_packed_kv_writeback", "flowblock_joint_generate_packed"]
        control = extract(base / "flowblock_packed.py", methods,
                          {"torch": torch, "TokenArray": ObservedTokenArray,
                           "_contiguous": helpers._contiguous},
                          ignore_prompt_eos=ignore_prompt_eos)
        self.provenance = {
            "commit": SOURCE_COMMIT,
            "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (base / n for n in ("utils.py", "generate_uniform.py", "flowblock_packed.py"))},
            "control": ("Official packed-loop AST; EOS guard excludes immutable prompt"
                        if ignore_prompt_eos else "Unchanged extracted official packed-loop AST"),
            "ignore_prompt_eos": ignore_prompt_eos,
            "backend": "Native Transformers shared-backend adapter; not official fused speed",
            "gate": gate, "gate_threshold": .8,
            "block_size": block_size, "window": window,
        }

        def prefill(model, tokens, cache, positions, mask, limit, block):
            wall = time.perf_counter()
            begin = torch.cuda.Event(enable_timing=True) if native.device.type == "cuda" else None
            end = torch.cuda.Event(enable_timing=True) if begin is not None else None
            if begin is not None:
                begin.record()
            native_cache = DynamicCache()
            native(tokens, position_ids=positions, attention_mask=mask[:, None],
                   use_cache=True, past_key_values=native_cache, output_attentions=True)
            cache.past_key_values._data = torch.stack(
                [torch.stack(pair) for pair in native_cache.to_legacy_cache()])
            if end is not None:
                end.record()
                end.synchronize()
            self.backend.events.append({
                "phase": "prefill", "q_tokens": tokens.shape[1],
                "physical_forwards": 1, "max_physical_q": tokens.shape[1],
                "wall_ms": (time.perf_counter()-wall)*1000,
                "cuda_interval_ms": begin.elapsed_time(end) if begin is not None else None,
            })

        def mask(batch, n, block):
            pos = torch.arange(n*block, device=native.device)
            return (pos[:, None]//block >= pos[None, :]//block)[None].expand(batch, -1, -1)

        def top1(logits):
            return native._sample_with_temperature_topk_topp(
                logits, temperature=0., top_k=None, top_p=None)

        self.owner = types.SimpleNamespace(
            model=self.backend, phase1_window_size=window, phase1_spawn_signal=gate,
            phase1_spawn_threshold=.8, early_stop=True, prefilling_limit=8192,
            cache_factory=types.SimpleNamespace(max_length=8192, create=lambda: Buffer(native)),
            block_runner=types.SimpleNamespace(prefill=prefill),
            iterator_factory=types.SimpleNamespace(
                create=lambda x, b: util.BlockIterator(x, b, start_block_align=True)),
            diff_iteration=types.SimpleNamespace(num_forwards=0, iter_no=0),
            decoder=types.SimpleNamespace(
                mask_id=mask_id, eos_id=eos_id, threshold=config["confidence_threshold"],
                edit_threshold=config["editing_threshold"],
                max_post_edit_steps=config["max_post_steps"],
                penalty_lambda=0., _top1_and_confidence=top1),
            _get_embedding_layer=lambda: native.model.word_embeddings,
            _get_bd_attn_mask=mask,
            _get_pos_ids=lambda batch, n, device: torch.arange(n, device=device)[None].expand(batch, -1),
            _make_block_locs=lambda first, total, b: [
                util.BlockLoc(start, min(start+b, total)) for start in range(first, total, b)],
        )
        for method in methods[:-1]:
            setattr(self.owner, method, types.MethodType(getattr(control, method), self.owner))
        self.generate = types.MethodType(control.flowblock_joint_generate_packed, self.owner)

    @torch.inference_mode()
    def run(self, prompt, gen_length):
        self.owner.cache_factory.max_length = prompt.shape[1]+gen_length
        official_return = self.generate(prompt, gen_length=gen_length, block_length=self.block_size)
        return self.raw_tokens.data, self.backend.events, {
            "official_metrics": self.owner.last_generation_metrics,
            "official_return_length": official_return.shape[1],
            "provenance": self.provenance,
        }
