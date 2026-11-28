"""Bounded paired prefix-dose screening on H0; never counts as formal90."""
import ast
from contextlib import redirect_stdout
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import sys
import time

from .shared_fixed_shard import DEVICES, MANIFEST, shard_cases
from .shared_gemma_worker import Handle, evaluate
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.methods.t2t.unified_rows import Params


def load_staged_h0():
    base = Path("/data1/project_data/gx-erqw-20261123")
    remaining = int(os.environ["BDLM_WINDOW_END"]) - int(time.time()) - 30
    if remaining < 60 or os.environ.get("CUDA_VISIBLE_DEVICES") != DEVICES["n164:H0"]:
        raise RuntimeError("No H0 window")
    signal.alarm(remaining)
    import torch
    from importlib.metadata import version
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from bdlm_serving.runtime.diffusiongemma import check_snapshot
    if (version("transformers"), version("math-verify")) != ("5.12.0", "0.8.0"):
        raise RuntimeError("Unregistered package versions")
    path = base / "models/diffusiongemma-cd40195"
    check_snapshot(path)
    proc = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
        experts_implementation="grouped_mm", device_map={"": "cpu"}).eval()
    helper = base/"runtime/diffusiongemma-native-20261125/dg-remote-worker.py"
    if sha256(helper.read_bytes()).hexdigest() != "1d065ca8e619dd434f09754f221bbc71514fb48477f0498958ebdfdb3f6f01d2":
        raise RuntimeError("Protected staging helper changed")
    fn, = [n for n in ast.parse(helper.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "prepare"]
    stage = {"torch": torch, "time": time, "flag": Path(os.environ["DG_SIGNAL"])}
    exec(compile(ast.Module([fn], []), str(helper), "exec"), stage)
    with redirect_stdout(sys.stderr):
        stage["prepare"](model)
    device = torch.device("cuda", 0)
    model.to(device)
    if any(t.device != device for t in (*model.parameters(), *model.buffers())):
        raise RuntimeError("Parameters or buffers not completely on H0")
    eos = model.generation_config.eos_token_id
    special = set(proc.tokenizer.all_special_ids) | set(eos if isinstance(eos, list) else [eos])
    return Handle(model, proc, device, frozenset(special | {model.generation_config.pad_token_id}))


def main():
    base = Path("/data1/project_data/gx-erqw-20261123")
    ctx = load_staged_h0()
    tasks = load_tasks(base/"runtime/t2t-native-1125/prefix90/samples.json")
    shard = shard_cases(json.loads(MANIFEST.read_bytes())["samples"], "n164:H0", "screen18")
    for q in (32, 128, 512, 64, 256):
        for j, row in enumerate(shard):
            doses = (Fraction(0), Fraction(1,8), Fraction(1,4), Fraction(1,2))
            k = j % len(doses)
            for dose in (*doses[k:], *doses[:k]):
                print(f"start index={row['index']} q={q} prefix_share={dose}", file=sys.stderr, flush=True)
                r = evaluate(ctx, tasks[row["index"]], q, Params(dose, Fraction(1,4)))
                r.update(stage="screen18", board="n164:H0", code=os.environ["BDLM_CODE_DIGEST"])
                print(json.dumps(r, separators=(",", ":"), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
