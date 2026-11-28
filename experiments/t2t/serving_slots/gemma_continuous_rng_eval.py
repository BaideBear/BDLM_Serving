"""A1 serial fixed Q256; reuse the exact verified visibility runner and controls."""
import csv
import hashlib
import json
import os
from pathlib import Path
from time import time
from unittest.mock import patch

from .gemma_cold_frontier_last_eval import board, CKPT, DEADLINE, GPU, SAMPLES, SCORER, read, sha

if __name__ == "__main__":
    board(False)  # before importing any model dependency

from . import gemma_native_visibility_eval as vis
from . import gemma_strict_request as request, gemma_strict_sampler as sampler
from .gemma_continuous_rng import sampler_class
from .gemma_joint_verify_eval import ORDER, check
from bdlm_serving.methods.t2t import serving_profiles as api

RT = Path("/gs/runtime/gemma-continuous-rng-a1-1128")
ROOT = Path("/gs/results/gemma-continuous-rng-a1-1128")
PRIOR_RT = Path("/gs/runtime/gemma-native-visibility-a1-1128")


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run(model, proc, data, task, enabled, probe):
    init, ticks = [], []
    cls = sampler_class(sampler, task["seed"], enabled, init, vis.h)
    base_step = api.StrictHandle.step

    def step(handle, q):
        req = handle._request
        s = req.sampler
        check(q == 256 and req.tail is None and isinstance(s, cls), "Non-solo sampler")
        count = len(init)
        before = vis.h(s.rng.get_state())
        e = base_step(handle, q)
        after = vis.h(s.rng.get_state())
        if e["encoded"]:
            check(len(init) == count + 1 and e["route"] in ("verify", "verify_paid"), "Wrong opening")
            if enabled:
                check(init[-1]["before"] == before and init[-1]["after"] == after, "Broken continuation")
            else:
                check(after == before, "Verify consumed block-private stream")
        else:
            check(len(init) == count, "Unexpected new canvas")
        sampled = len(e["parts"][0]["sampled"])
        end = before if e["encoded"] else after
        check(sampled in (0, 256) and (end != before) == bool(sampled), "Clock/RNG mismatch")
        ticks.append(dict(ordinal=e["ordinal"], route=e["route"], before=before, end=end,
                          event=digest(e), ids=vis.h(s.ids), accepted=vis.h(s.accepted)))
        return e

    with patch.object(request, "SelectedRowSampler", cls), patch.object(api.StrictHandle, "step", step):
        r = vis.run(model, proc, data, task, "on", probe)
    check(len(init) == r["encoder_F"] + 1 and 256 * len(ticks) == r["decoder_Q"]
          and r["mask_layer_calls"] == 30 * len(ticks), "Incomplete audit")
    for j, x in enumerate(init):
        block = [t for t in ticks if t["ordinal"] == j]
        check(x["carried"] == bool(j and enabled) and x["after"] == block[0]["before"] and
              block[-1]["route"] in ("verify", "verify_paid") and
              all(a["end"] == b["before"] for a, b in zip(block, block[1:])), "Block ledger mismatch")
    return r, dict(init=init, ticks=ticks)


def main():
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as native

    board(True)
    old = read(vis.ROOT / "registration.json")
    with (RT / "validated_visibility/pairs.csv").open() as f:
        refs = {int(r["index"]): r["sha256"] for r in csv.DictReader(f)}
    tasks = vis.load_tasks(SAMPLES)
    reg = dict(pid=os.getpid(), start=time(), start_ticks=Path("/proc/self/stat").read_text().split(") ")[1].split()[19],
               gpu=GPU, deadline=DEADLINE, boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
               inputs=sha(SAMPLES), manifest=sha(Path(api.__file__).resolve().parents[3] / "benchmarks/frozen90/manifest.json"),
               scorer=sha(Path(vis.scoring.__file__)), native=sha(Path(native.__file__)),
               checkpoint=str(CKPT), checkpoint_configs={n:sha(CKPT/n) for n in
                   ("config.json","generation_config.json","model.safetensors.index.json")},
               torch=str(torch.__version__), tf=transformers.__version__, ref=digest(refs),
               policy=sha(RT / "registration.md"), launcher=sha(RT / "launch.sh"),
               sources={p.name:sha(p) for p in (Path(__file__), Path(__file__).with_name("gemma_continuous_rng.py"),
                        Path(request.__file__), Path(sampler.__file__), Path(api.__file__), Path(vis.__file__),
                        Path(vis.__file__).with_name("gemma_native_visibility.py"))})
    check(all(reg[k] == old[k] for k in ("inputs", "scorer", "native", "torch", "tf", "checkpoint")) and
          reg["inputs"] == vis.SAMPLES_SHA and reg["scorer"] == SCORER and
          reg["sources"][Path(sampler.__file__).name] == "f927477dda44859036ef138ed9d51ad80bb4f648f9960acb3ff916aea009f293" and
          all(reg["sources"][k] == v for k,v in old["sources"].items()) and
          len(tasks) == len(ORDER) == 90 and set(refs) == set(ORDER) == set(range(90)) and
          all(sha(vis.ROOT / f"pair-{i:02}.json") == h for i,h in refs.items()) and torch.cuda.device_count() == 1,
          "Bad stack, inputs, source, or references")
    ROOT.mkdir()
    vis.save(ROOT / "registration.json", reg)
    print("REGISTERED", reg["pid"], reg["policy"], flush=True)
    proc = AutoProcessor.from_pretrained(CKPT, local_files_only=True)
    board(True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(CKPT, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", experts_implementation="grouped_mm", device_map={"":"cuda:0"}).eval()
    for j,i in enumerate(ORDER, 1):
        task = tasks[i]
        p = vis.ROOT / f"pair-{i:02}.json"
        prev = read(p)
        hist = read(vis.HIST / f"C256-dense-stop1-{i:02}.json")
        check(sha(p) == refs[i] and prev["seed"] == task["seed"] and prev["index"] == i and
              all(hist[k]==task[k] for k in ("seed", "task_id", "dataset", "index")), "Bad reference row")
        data = proc.apply_chat_template([{"role":"user","content":task["prompt"]}], tokenize=True,
               add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
        inp = vis.input_digest(data)
        check(inp == hist["input_sha256"], "Bad prompt")
        row = dict(index=i, seed=task["seed"], dataset=task["dataset"], input_sha=inp,
                   reference=refs[i], pid=reg["pid"], registration=digest(reg), audit={})
        for enabled in ((False,True) if j % 2 else (True,False)):
            board(True)
            arm = "on" if enabled else "off"
            result, track = run(model, proc, data, task, enabled, j <= 3)
            if not enabled:
                check(result == prev["on"], f"#{i}: full off row mismatch")
            row[arm], row["audit"][arm] = result, track
        a,b = (row["audit"][k] for k in ("off", "on"))
        heads = [[t for t in v["ticks"] if not t["ordinal"]] for v in (a,b)]
        check(heads[0] == heads[1] and a["init"][0] == b["init"][0] and
              row["off"]["ids"][:256] == row["on"]["ids"][:256], "First block changed")
        exposed = len(a["init"]) > 1
        check(exposed == (len(b["init"]) > 1), "Inconsistent initial exposure")
        if exposed:
            check(a["init"][1]["ids"] != b["init"][1]["ids"], "No second-canvas intervention")
        else:
            check(row["off"] == row["on"] and a == b, "Nonidentical zero-exposure row")
        if j <= 3:
            check(exposed == (i in (0,10)), "Wrong anchor exposure")
            if exposed:
                pos = next(t for t in prev["native"]["tape"][1:] if t.get("clock") == 48)
                for k,x in (("off",a),("on",b)):
                    tick = row[k]["tape"][len(heads[0])]
                    check(tick["clock"] == 48 and tick["pos"] == pos["pos"] and tick["ids"] == x["init"][1]["ids"]
                          and (tick["ids"] == pos["ids"]) == (k == "on"), "Anchor next canvas not repaired")
        vis.save(ROOT / f"pair-{i:02}.json", row)
        if j in (3,30,90):
            stamp = dict(n=j, pid=reg["pid"], registration=digest(reg))
            vis.save(ROOT / f"checkpoint-{j}.json", stamp)
            print("CHECKPOINT", j, flush=True)
    vis.save(ROOT / "complete.json", stamp)


if __name__ == "__main__":
    main()
