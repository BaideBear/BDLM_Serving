"""A0-only exact-row warm-refresh replay probe for fixed Live Q128."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from collections import Counter
from contextlib import ExitStack

GPU = "GPU-d3c3fc9c-b15d-9776-9094-7d3b85b705c1"
CASES = (0, 10, 20, 30, 40, 50)
Q = 128


def check(ok, why):
    if not ok:
        raise RuntimeError(why)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard(*, own):
    rows = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    check(rows == ([f"{os.getpid()}, {GPU}"] if own else []), f"A0 compute apps: {rows}")


def main():
    check(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU, "Wrong visible GPU")
    guard(own=False)
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from transformers.models.diffusion_gemma import generation_diffusion_gemma
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.runtime.diffusiongemma import observe
    from experiments.t2t.serving_slots.gemma_joint_verify_eval import digest
    from experiments.t2t.serving_slots.gemma_strict_eval import encoder_tape
    from experiments.t2t.serving_slots.gemma_unified_audit import projectors
    from experiments.t2t.serving_slots.gemma_verified_eos import VerifiedEOSRequest
    from experiments.t2t.serving_slots.mask_pair_gemma import input_digest

    check(torch.cuda.device_count() == 1 and str(torch.__version__) == "2.10.0+cu128"
          and transformers.__version__ == "5.12.0", "Wrong environment")
    check(sha(generation_diffusion_gemma.__file__) ==
          "cded57b7e365f0265b1c088e993c358b5f70acdf48055fce396540b4d9d691bc", "Sampler pin")
    samples = Path("/gs/code/samples.json")
    check(sha(samples) == "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed"
          and sha(scoring.__file__) ==
          "a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692", "Evaluation pins")

    class Probe(VerifiedEOSRequest):
        def __init__(self, model, data, *, seed, reuse):
            self.reuse = reuse
            self.previous = None
            self.opportunities = 0
            self.saved = 0
            self.warm_drains = 0
            self.tape = []
            self.rng_tape = []
            self.close_drains = self.cold_drains = 0
            super().__init__(model, data, seed=seed, q=Q, early=True)

        def _apply(self, states, pos, sample, prefix, cold, offer, route):
            head, = states
            local = [p - self.origin - 256 * head.ordinal for p in pos]
            positions = tuple(pos)
            key = (head.ordinal, self.cache.get_seq_length(), positions)
            warm = head.can_sample and len(head.rows.initialized) == 256
            candidate = self.previous
            self.previous = None
            take = (candidate is not None and candidate[0] == key and warm and
                    route == "sparse" and set(sample) == set(pos) and not head.rows.dirty
                    and not cold and offer == len(pos) == Q)
            if route == "drain":
                check(not sample, "Drain has sampler action")
                if warm:
                    self.warm_drains += 1
                    check(len(pos) == Q and head.rows.dirty == set(local), "Invalid warm drain")
                elif head.can_sample:
                    self.cold_drains += 1
                else:
                    self.close_drains += 1
            saved_project = self._project
            top = []

            def project(project_states, project_positions):
                if take:
                    self.opportunities += 1
                    check(self.bank is candidate[1] and tuple(self.bank.live) == candidate[2]
                          and self.bank.positions == list(pos) and head.sampler is candidate[3]
                          and head.sampler.age == candidate[4]
                          and torch.equal(head.sampler.ids[:, local], candidate[5])
                          and all(head.sampler.sc[r] is signal for r, signal in zip(local, candidate[6], strict=True)),
                          "Not the same resident bank or selected inputs")
                logits = candidate[7] if take and self.reuse else saved_project(project_states, project_positions)
                if take:
                    if self.reuse:
                        self.saved += 1
                    else:
                        check(torch.equal(logits, candidate[7]), "Repeated shape did not give bit-identical logits")
                if route == "drain" and warm:
                    self.previous = (key, self.bank, tuple(self.bank.live), head.sampler, head.sampler.age,
                                     head.sampler.ids[:, local], tuple(head.sampler.sc[r] for r in local), logits)
                top.append(logits[0].argmax(-1).tolist())
                return logits

            self._project = project
            try:
                event = super()._apply(states, pos, sample, prefix, cold, offer, route)
            finally:
                del self._project
            self.tape.append((event, top[0]))
            self.rng_tape.append(digest(head.sampler.rng.get_state().tolist()))
            return event

    def run(model, proc, data, task, reuse):
        with ExitStack() as stack:
            physical = stack.enter_context(observe(model))
            widths, positions = stack.enter_context(projectors(model))
            enc = stack.enter_context(encoder_tape(model))
            req = Probe(model, data, seed=task["seed"], reuse=reuse)
            while req.status == "ready":
                check(len(req.tape) < 4096, "Unbounded Live request")
                req.step()
            check(req.status == "done", "Live did not finish")
            ids = req.tokens
            special = set(proc.tokenizer.all_special_ids) | set(model.generation_config.eos_token_id)
            dec = [w for kind, w in physical if kind == "decoder"]
            check(len(dec) == len(req.tape) - req.saved and len(positions) == len(dec)
                  and len(widths) == 85 and all(v == dec for v in widths.values())
                  and all(0 < w <= Q for w in dec), "Wrong physical decoder bill")
            check(sum(e["decoder_Q"] for e, _ in req.tape) - sum(dec) == Q * req.saved, "Wrong saved rows")
            score = bool(scoring.score(proc.tokenizer.decode(ids, skip_special_tokens=True), task)["correct"])
            out = dict(index=task["index"], F=len(physical)-1, D=len(dec), E=len(enc)-1, Q=sum(dec),
                       N=sum(x not in special for x in ids), correct=score, opportunities=req.opportunities,
                       saved=req.saved, warm_drains=req.warm_drains, cold_drains=req.cold_drains,
                       closing_drains=req.close_drains, ids_sha256=digest(ids),
                       encoder_digest=digest(enc), rng_digest=digest(req.rng_tape))
            check(out["F"] == out["D"] + out["E"], "Wrong physical forwards")
            # _advance updates these same event objects after _apply.
            out["tape"] = [digest([{k: v for k, v in e.items() if k != "verified_eos_end"}, top])
                           for e, top in req.tape]
            out["input_sha256"] = input_digest(data)
            return out

    tasks = load_tasks(samples)
    checkpoint = "/gs/models/diffusiongemma-cd40195"
    proc = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    guard(own=False)
    torch.zeros(1, device="cuda:0")
    guard(own=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
        experts_implementation="grouped_mm", device_map={"": "cuda:0"},
    ).eval()
    check(all(p.is_cuda and p.dtype == torch.bfloat16 for p in model.parameters()), "Wrong placement/dtype")
    print(json.dumps(dict(kind="start", pid=os.getpid(), gpu=GPU, q=Q, cases=CASES, source=sha(__file__))),
          flush=True)
    totals = Counter()
    with torch.inference_mode():
        for i in CASES:
            guard(own=True)
            task = tasks[i]
            check(task["index"] == i, "Wrong case")
            data = proc.apply_chat_template([{"role": "user", "content": task["prompt"]}], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda:0")
            off = run(model, proc, data, task, False)
            on = None
            if off["opportunities"]:
                on = run(model, proc, data, task, True)
                check(all(on[k] == off[k] for k in off if k not in {"D", "F", "Q", "saved"})
                      and on["saved"] == off["opportunities"] == off["F"]-on["F"] == off["D"]-on["D"]
                      and off["Q"]-on["Q"] == Q * on["saved"], "Physical replay changed behavior")
            totals.update({k: off[k] for k in ("opportunities", "warm_drains", "cold_drains", "closing_drains")})
            print(json.dumps(dict(kind="case", off=off, on=on)), flush=True)
    print(json.dumps(dict(kind="complete", totals=totals)), flush=True)


if __name__ == "__main__":
    main()
