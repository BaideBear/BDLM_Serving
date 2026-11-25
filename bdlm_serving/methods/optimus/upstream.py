"""Frozen Optimus prefix plan/sparse sampler on the native SDAR backend.

This is a labelled control adaptation, not the full LMDeploy implementation.
Prompt prefill, EOS and final-cache completion use the common native protocol.
All resident keys through the rightmost selected position remain visible,
including previously written MASK rows that are not selected this pass.
"""
import ast
import dataclasses
import enum
import hashlib
from pathlib import Path
import sys
import types

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from bdlm_serving.upstream import control_directory
COMMIT = "7343a37b2a69337caa452f17b59727dc6665f039"


def load_controls():
    class Strategy(enum.Enum):
        LOW_CONFIDENCE_DYNAMIC = 1
        LOW_CONFIDENCE_STATIC = 2
        SEQUENTIAL = 3

    namespace = {
        "torch": torch, "np": np, "dataclass": dataclasses.dataclass,
        "field": dataclasses.field, "DLLM_MASKED": 0, "DLLM_UNMASKED": 1,
        "DLLM_CACHED": 2, "UnmaskingStrategy": Strategy,
        "record_function": torch.profiler.record_function,
    }
    module = types.ModuleType("erqw_frozen_optimus_controls")
    module.__dict__.update(namespace)
    sys.modules[module.__name__] = module
    hashes = {}
    for name, definitions in (
            ("sequence.py", {"ProcessingPlan", "DelayedCacheState"}),
            ("unmasking.py", {"UnmaskingProcessor"})):
        path = control_directory("Optimus") / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        nodes = [n for n in ast.parse(path.read_text()).body
                 if isinstance(n, ast.ClassDef) and n.name in definitions]
        assert {n.name for n in nodes} == definitions
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
        exec(compile(tree, str(path), "exec"), module.__dict__)
    return module, Strategy, hashes


class SelectedResidentCache(DynamicCache):
    """Mirror the official rightmost-plus-one KV visibility, with absolute writes."""
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.buffers = []
        self.initialized = set()
        self.selected = None
        self.visible_end = 0
        self.last_end = 0

    def configure(self, selected, absolute_cpu):
        end = max(absolute_cpu)+1
        self.initialized.update(absolute_cpu)
        if not all(position in self.initialized for position in range(end)):
            raise RuntimeError("Official-plan adapter exposed an uninitialized KV hole")
        self.selected = selected
        self.visible_end = end

    def get_seq_length(self, layer_idx=0):
        return self.last_end

    def update(self, key, value, layer_idx, cache_kwargs=None):
        if layer_idx == len(self.buffers):
            shape = (*key.shape[:-2], self.capacity, key.shape[-1])
            self.buffers.append((key.new_empty(shape), value.new_empty(shape)))
        k, v = self.buffers[layer_idx]
        k.index_copy_(-2, self.selected, key)
        v.index_copy_(-2, self.selected, value)
        self.last_end = self.visible_end
        return k[..., :self.visible_end, :], v[..., :self.visible_end, :]
