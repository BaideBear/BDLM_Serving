"""Same-device paired GPU evaluation. The original IBS decoder and scorer are not forked."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
from types import SimpleNamespace

from experiments.t2t.serving_slots import gemma_ibs_prefix_bridge as bridge
from experiments.t2t.serving_slots import gemma_ibs_prefix_bridge_eval as pilot


def registered_cases(panel):
    if panel["protocol"] == "gemma-ibs-bridge-q32-frozen90-v1":
        bridge.require(panel["indices"] == list(range(90)) and panel["tiers"] == [32]
                       and panel["prior_six"] == [1,11,21,31,51,71],
                       "Full Q32 registration changed")
    else:
        bridge.require(panel["protocol"] == "gemma-ibs-bridge-replay-v1" and
                       panel["indices"] == [1,11,21,31,51,71] and
                       panel["tiers"] == list(bridge.TIERS), "Unknown qualification panel")
    bridge.require(panel["old_pilot"] == list(bridge.CASES)
                   and panel["arms"] == ["plain", "off", "on"], "Pilot or arms changed")
    return panel["indices"], panel["tiers"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    device = parser.add_mutually_exclusive_group(required=True)
    device.add_argument("--staged-h0", action="store_true")
    device.add_argument("--checkpoint", type=Path, help="On a GPU exclusively allocated by the caller")
    a = parser.parse_args()
    import torch
    from bdlm_serving.evaluation import frozen, frozen90, scoring
    from bdlm_serving.methods.optimus import upstream
    from bdlm_serving.runtime import diffusiongemma_optimus as rt, optimus_limits
    from experiments.t2t.serving_slots.mask_pair_gemma import input_digest
    from experiments.t2t.serving_slots import shared_gemma_worker, shared_h0_screen

    panel = pilot.read(a.panel)
    indices, tiers = registered_cases(panel)
    tasks = frozen90.load_tasks(a.samples)
    paths = dict(runtime=rt.__file__, limits=optimus_limits.__file__, scorer=scoring.__file__,
                 samples=a.samples, model_manifest=frozen.ROOT/"benchmarks/checkpoints/diffusiongemma.json",
                 native=frozen.ROOT/"archive/diffusiongemma-native/config.json", controls_module=upstream.__file__)
    bridge.require({k: pilot.sha(Path(v)) for k,v in paths.items()} == pilot.PINS, "Inputs changed")
    ctl, strategy, hashes = upstream.load_controls()
    bridge.require(hashes == pilot.CONTROL_HASHES and upstream.COMMIT == "7343a37b2a69337caa452f17b59727dc6665f039"
                   and version("transformers") == "5.12.0" and version("math-verify") == "0.8.0",
                   "Unregistered controller or scorer")
    sampler = ctl.UnmaskingProcessor(SimpleNamespace(confidence_threshold=.9, block_length=256,
                  denoising_steps=256, unmasking_strategy=strategy.LOW_CONFIDENCE_DYNAMIC))
    if a.staged_h0:
        bridge.require(os.environ["BDLM_CODE_DIGEST"] == pilot.sha(a.panel), "Panel hash differs")
    a.out.mkdir(exist_ok=True, parents=True)
    # Exclusive child directory; never overwrite a partial run with a repeat.
    raw = a.out/"pairs"
    raw.mkdir(mode=0o700)
    handle = shared_h0_screen.load_staged_h0() if a.staged_h0 else shared_gemma_worker.load(a.checkpoint)
    bridge.require(handle.device.type == "cuda" and torch.cuda.device_count() == 1, "Select one GPU")
    proof = dict(panel=panel, panel_sha=pilot.sha(a.panel), torch=str(torch.__version__),
                 gpu=torch.cuda.get_device_name(), uuid=str(torch.cuda.get_device_properties(0).uuid),
                 pins=pilot.PINS, controls=hashes, pid=os.getpid(), sources={
                     Path(p).name:pilot.sha(Path(p)) for p in
                     (__file__, pilot.__file__, bridge.__file__, shared_h0_screen.__file__)})
    pilot.write(raw/"registration.json", proof)
    provenance = pilot.digest(proof)
    for j, i in enumerate(indices):
        task = tasks[i]
        data = handle.processor.apply_chat_template([{"role":"user", "content":task["prompt"]}],
                tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt").to(handle.device)
        input_sha = input_digest(data)
        for k, q in enumerate(tiers):
            arms = {}
            for arm in (("plain", "off", "on") if (j+k)%2 == 0 else ("on", "plain", "off")):
                bridge.require(input_digest(data) == input_sha, "Input mutated between arms")
                arms[arm] = pilot.run(handle.model, handle.processor, data, sampler, ctl, task, q, arm)
            plain, off, on = (arms[key] for key in ("plain", "off", "on"))
            bridge.require(plain == {key:value for key,value in off.items() if key not in ("notes","bill")},
                           f"Unmodified IBS replay mismatch for {i=} {q=}")
            first = bridge.match(off, on)
            receipt = dict(index=i, task_id=task["task_id"], dataset=task["dataset"], seed=task["seed"],
                           q=q, reg=provenance, input_sha=input_sha, first=first, plain_sha=pilot.digest(plain))
            pilot.write(raw/f"{i:02}-q{q}.json", dict(**receipt, off=off, on=on))
            for key, arm in (("off",off), ("on",on)):
                receipt[key] = {field:arm[field] for field in ("N","F","decoder_Q","encoder_F","correct")}
                receipt[key].update(extra_Q=arm["bill"]["extra_Q"], peak=arm["bill"]["peak"],
                                    ids_sha=pilot.digest(arm["output_ids"]))
            print(json.dumps(receipt, separators=(",",":"), allow_nan=False), flush=True)
    pilot.write(raw/"complete.json", dict(reg=provenance, pairs=len(indices)*len(tiers)))


if __name__ == "__main__":
    main()
