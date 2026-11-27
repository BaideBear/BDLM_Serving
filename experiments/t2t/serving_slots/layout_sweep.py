"""Bounded single-device frozen90 worker. Weights and all request outputs stay on cluster."""
import argparse
import fcntl
import json
import os
import socket
import time
from pathlib import Path


def grid():
    cases = [j + k for j in range(10) for k in (0, 10, 20)] + list(range(30, 90))
    arms = [(layout, cold) for cold in (50, 0, 100) for layout in ("rect", "split", "halo")
            if (layout, cold) != ("split", 0)]
    return [(layout, cold, c, i) for i in cases for c in (32, 128, 64) for layout, cold in arms]


def write(path, obj):
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("samples", "checkpoint", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--deadline", required=True, type=int)
    a = ap.parse_args()
    from bdlm_serving.evaluation.frozen import sha
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.evaluation.scoring import score
    from tools.run_history_gemma import request
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    assert torch.cuda.device_count() == 1 and transformers.__version__ == "5.12.0"
    tasks = load_tasks(a.samples)
    dev = torch.cuda.get_device_name()
    scorer = sha(Path(__file__).resolve().parents[3] / "bdlm_serving/evaluation/scoring.py")
    common = dict(schema="rect-cold-v1", gpu=dev, host=socket.gethostname(),
                  samples_sha256=sha(a.samples), scorer_sha256=scorer,
                  torch=str(torch.__version__), transformers=transformers.__version__)
    print("loading", dev, os.environ["CUDA_VISIBLE_DEVICES"], flush=True)
    proc = AutoProcessor.from_pretrained(a.checkpoint, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(a.checkpoint, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa", experts_implementation="grouped_mm",
        device_map={"": "cuda:0"}).eval()
    for layout, cold, c, i in grid():
        if time.time() >= a.deadline:
            break
        key = f"{layout}-cold{cold}-C{c}-{i:02d}"
        out = a.output / (key + ".json")
        if out.exists():
            continue
        with (a.output / (key + ".lock")).open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if out.exists():
                continue
            print("START", key, flush=True)
            with torch.inference_mode():
                r = request(model, proc, tasks[i], c, 2048, dispatch=True, layout=layout, cold_share=cold)
            steps = r["trace"]
            local = [s.get("local_attention_q", s["q"]) for s in steps]
            if not all(l == {"split": s["q"], "rect": s["history_q"],
                             "halo": min(s["q"], 2*s["history_q"])}[layout]
                       and (s["history_q"] <= c or s["close"])
                       for l, s in zip(local, steps)):
                raise RuntimeError("Invalid attention geometry for " + key)
            saved = {k: r[k] for k in ("index", "task_id", "dataset", "seed", "output_ids", "output_text",
                    "output_tokens", "generation_forwards", "generation_physical_padded_rows", "truncated",
                    "events", "history_query_rows", "close_history_excess", "attention_backend", "trace")}
            saved.update(common, layout=layout, cold_share=cold, budget=c, local_attention_Q=sum(local))
            try:
                saved["correct"] = bool(score(r["output_text"], tasks[i])["correct"])
            except (RuntimeError, FileNotFoundError) as e:
                saved["score_pending"] = str(e)
            write(out, saved)
            print("DONE", key, saved.get("correct"), r["output_tokens"], r["generation_forwards"], flush=True)


if __name__ == "__main__":
    main()
