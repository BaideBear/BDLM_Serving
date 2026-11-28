"""Measure the strict parent's first real successor projection, without changing its plan."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

SAMPLES = "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed"
PROFILE = "dense_stop_joint_verified_eos"
INDICES = (7, 13, 25)


def require(value, msg):
    if not value:
        raise RuntimeError(msg)


def emit(kind, **data):
    print(json.dumps(dict(kind=kind, **data), separators=(",", ":"), allow_nan=False), flush=True)


def run(q, remaining=False):
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.methods.t2t.serving_profiles import open_strict_gemma
    from experiments.t2t.serving_slots.gemma_unified_audit import projectors
    from bdlm_serving.runtime.diffusiongemma import observe

    samples = Path("/gs/code/samples.json")
    weights = Path("/gs/models/diffusiongemma-cd40195")
    source = os.environ["PARENT_GIT_SHA"]
    uuid = os.environ["CUDA_VISIBLE_DEVICES"]
    require(q in (32, 512) and source == "7f160c54fa1fcd005aa92f9420c700f5df4cb47f", "registry")
    require(transformers.__version__ == "5.12.0" and torch.cuda.device_count() == 1 and
            str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-") == uuid[4:], "stack/board")
    require(sha256(samples.read_bytes()).hexdigest() == SAMPLES, "panel drift")
    tasks = load_tasks(samples)
    indices = tuple(i for i in range(90) if i not in INDICES) if remaining else INDICES
    emit("worker", q=q, uuid=uuid, pid=os.getpid(), source=source, runner=sha256(Path(__file__).read_bytes()).hexdigest(),
         torch=torch.__version__, transformers=transformers.__version__, indices=indices, profile=PROFILE)
    proc = AutoProcessor.from_pretrained(weights, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(weights, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", experts_implementation="grouped_mm", device_map={"":"cuda:0"}).eval()
    with torch.inference_mode():
        for i in indices:
            task = tasks[i]
            require(task["index"] == i, "index")
            torch.manual_seed(task["seed"])
            torch.cuda.manual_seed(task["seed"])
            data = proc.apply_chat_template([{"role":"user", "content":task["prompt"]}], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
            origin = data["input_ids"].shape[-1]
            with projectors(model) as (layers, positions), observe(model) as work:
                handle = open_strict_gemma(model, data, seed=task["seed"], initial_offer=q,
                                           max_new_tokens=2048, profile=PROFILE)
                require(work == [("encoder", origin)], "unexpected prefill")
                tick = 0
                exposed = False
                while handle.status == "ready":
                    e = handle.step(q)
                    pos = positions[-1]
                    width = len(pos)
                    require(len(positions) == tick+1 and pos == sorted(set(pos)) == e["positions"]
                            and 0 < width == e["decoder_Q"] <= q and
                            all(len(v) == tick+1 and v[-1] == width for v in layers.values()),
                            "per-layer physical audit")
                    tail = [p for p in pos if p >= origin+256]
                    if tail:
                        require(all(p < origin+512 for p in tail), "not the first successor")
                        emit("first_projection", index=i, seed=task["seed"], task_id=task["task_id"], q=q,
                             tick=tick, width=width, origin=origin, physical_rows=pos,
                             successor_rows=len(tail), missing_successor_rows=256-len(tail),
                             paid_F_to_event=len(work)-1, decoder_Q_to_event=sum(map(len,positions)),
                             route=e.get("route"), layer_hooks=len(layers))
                        exposed = True
                        break
                    tick += 1
                if not exposed:
                    emit("no_successor", index=i, q=q, decoder_F=tick)
                if handle.status == "ready":
                    handle.close()
    emit("finished", indices=indices, q=q)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("q", type=int, choices=[32,512])
    p.add_argument("--remaining", action="store_true", help="exact complement of the three probe cases")
    a = p.parse_args()
    run(a.q, a.remaining)
