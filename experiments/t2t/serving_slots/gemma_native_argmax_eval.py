"""Same-H0 native.generate / request-step / paid-verify paired qualification."""
from argparse import ArgumentParser
from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path
from time import time
from unittest.mock import patch


B, CAP = 256, 2048
SIX = (0, 10, 20, 1, 11, 21)
ORDER = SIX + tuple(i for i in range(90) if i not in SIX)
ARMS = ("native", "step", "paid")
GPU = "GPU-ceadb3cd-e63f-1446-a2d5-a291144f8efb"
BASE = Path("/data1/project_data/gx-erqw-20261123")
SAMPLES = BASE / "runtime/t2t-native-1125/prefix90/samples.json"
SAMPLES_SHA = "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed"
NATIVE_SHA = "cded57b7e365f0265b1c088e993c358b5f70acdf48055fce396540b4d9d691bc"
SCORER_SHA = "a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix(".writing")
    tmp.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def run(ctx, data, task, arm):
    import torch
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import StableAndConfidentStoppingCriteria
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.runtime.diffusiongemma import account, observe
    from .gemma_native_argmax_request import NativeArgmaxRequest
    from .gemma_unified_audit import projectors

    model = ctx.model
    eos = set(model.generation_config.eos_token_id)
    origin = data["input_ids"].shape[1]
    enc, inputs, stops, hits, steps = [], [], [], [], []
    criterion = StableAndConfidentStoppingCriteria.__call__

    def record_criterion(obj, draft, scores, **kwargs):
        require(draft.shape == (1, B) and scores.shape[:2] == (1, B)
                and scores.dtype == torch.float32
                and (obj.stability_threshold, obj.confidence_threshold) == (1, .005),
                "Unexpected criterion input")
        result = criterion(obj, draft, scores, **kwargs)
        ids = draft[0].tolist()
        hit = bool(result.item())
        stops.append(dict(d=len(inputs)-1, draft=digest(ids), hit=hit))
        if hit:
            hits.append(ids)
        return result

    def encoder(_, args, kw):
        enc.append(kw["input_ids"][0].tolist())

    def decoder(_, args, kw):
        inputs.append(digest(kw["decoder_input_ids"][0].tolist()))

    torch.manual_seed(task["seed"])
    with torch.inference_mode(), ExitStack() as stack:
        physical = stack.enter_context(observe(model))
        projections, positions = stack.enter_context(projectors(model))
        stack.enter_context(patch.object(StableAndConfidentStoppingCriteria, "__call__", record_criterion))
        stack.callback(model.model.encoder.register_forward_pre_hook(encoder, with_kwargs=True).remove)
        stack.callback(model.model.decoder.register_forward_pre_hook(decoder, with_kwargs=True).remove)
        if arm == "native":
            result = model.generate(**data, max_new_tokens=CAP)
            require(torch.equal(result.sequences[:, :origin], data["input_ids"]), "Changed native prompt")
            suffix = result.sequences[0, origin:].tolist()
            ids, _ = account(physical, suffix, origin, B, CAP, eos, ctx.special_ids,
                             model.generation_config.pad_token_id, result.tokens_per_forward.item())
            inits = []
        else:
            req = NativeArgmaxRequest(model, data, seed=task["seed"], max_new_tokens=CAP,
                                      paid_verify=arm == "paid")
            while req.status == "ready":
                at = len(physical)
                prev = len(stops)
                e = req.step(B)
                part, = e["parts"]
                sampled = part["sampled"] == list(range(B))
                use = [("decoder", B)] + [("encoder", B)] * e["encoder_F"]
                require(physical[at:] == use and e["encoded"] == bool(e["encoder_F"])
                        and (e["generation_F"], e["physical_generation_Q"]) == (len(use), B * len(use))
                        and e["decoder_Q"] == B and e["positions"] == positions[-1]
                        and len(stops)-prev == sampled and e["dirty_before_paid"] == part["dirty_before"],
                        "Step did not match physical decoder/encoder/criterion bill")
                if e["native_draft"] is not None:
                    require(sampled and stops[-1]["hit"] and list(e["native_draft"]) == hits[-1],
                            "Published an argument other than the actual criterion draft")
                if e["encoded"]:
                    require(list(e["delivered"]) == enc[-1], "Encoder did not receive delivery")
                steps.append(e)
            ids = list(req.result())
            inits = req.canvas_inits

    dec = len(inputs)
    require(physical[0] == ("encoder", origin) and enc[0] == data["input_ids"][0].tolist()
            and len(projections) == 85 and all(x == [B] * dec for x in projections.values()),
            "Incomplete prefill or actual Q/K/V projection tape")
    epochs = [[]]
    for kind, width in physical[1:]:
        require(width == B and kind in ("decoder", "encoder"), "Not physical Q256")
        if kind == "decoder":
            epochs[-1].append(sum(map(len, epochs)))
        else:
            epochs.append([])
    require(len(epochs) == len(enc) == (len(ids)+B-1)//B, "Invalid encoder/block count")
    verify, unread, paid = 0, 0, 0
    sampled_trace = []
    for j, turns in enumerate(epochs):
        c = [v for v in stops if v["d"] in turns]
        stop_mask = [v["hit"] for v in c]
        block = ids[j*B:(j+1)*B]
        require(turns and c and stop_mask == [False]*(len(c)-1)+[True] and
                [v["d"] for v in c] == turns[:len(c)], f"Block {j} lacks an actual terminal hit")
        require(all(positions[d] == list(range(origin + j*B, origin+(j+1)*B)) for d in turns),
                "Noncontiguous physical positions")
        sampled_trace.append([(inputs[v["d"]], v["draft"]) for v in c])
        if j + 1 < len(epochs):
            require(len(block) == B and not eos.intersection(block) and enc[j+1] == block,
                    "Nonterminal encoder/delivery mismatch")
        if arm == "native":
            require(len(turns) == len(c) and suffix[j*B:j*B+len(block)] == block
                    and hits[j][:len(block)] == block, "Native output did not come from hit")
        else:
            es = [steps[d] for d in turns]
            parts = [e["parts"][0] for e in es]
            paid += sum(len(e["dirty_before_paid"]) for e in es)
            require([e["ordinal"] for e in es] == [j]*len(es) and
                    [p["dirty_before"] for p in parts] == [[]]+[p["dirty_after"] for p in parts[:-1]] and
                    all(p["age_after"] == k+1 for k, p in enumerate(parts[:len(c)])) and
                    list(es[-1]["delivered"]) == block and
                    all(e["delivered"] is None and not e["encoded"] for e in es[:-1]),
                    "Missing carry, age, or actual delivery boundary")
            if arm == "step":
                require(len(turns) == len(c) and hits[j][:len(block)] == block
                        and es[-1]["stop_reason"] == "native_hit"
                        and es[-1]["unread_speculative_rows"] == parts[-1]["dirty_after"],
                        "Wrong same-turn output or unread canvas")
                unread += len(es[-1]["unread_speculative_rows"])
            else:
                require(len(turns) == len(c)+1 and es[-1]["stop_reason"] == "paid_verify"
                        and not parts[-1]["sampled"] and not parts[-1]["dirty_after"],
                        "Strict arm did not pay ordered verify")
                verify += 1
    require(len(hits) == len(epochs) and dec == len(stops)+verify, "Incomplete stop tape")
    if arm != "native":
        require(len(inits) == len(epochs) and [x["carried"] for x in inits] ==
                [False]+[True]*(len(inits)-1), "Wrong RNG lineage")
    require(len(ids) == CAP or ids[-1] in eos, "Neither EOS nor cap")
    enc_f = len(enc)-1
    metrics = dict(N=sum(i not in ctx.special_ids for i in ids), prefill_Q=origin,
                   D=dec, D_Q=B*dec, sample_D=len(stops), verify_D=verify,
                   E=enc_f, E_Q=B*enc_f, F=dec+enc_f, Q=B*(dec+enc_f),
                   blocks=len(epochs), hits=len(hits), eos=ids[-1] in eos,
                   correct=bool(scoring.score(ctx.processor.tokenizer.decode(
                       ids, skip_special_tokens=True), task)["correct"]),
                   write_rows_read=paid if arm!="native" else None,
                   unread_speculative_rows=unread if arm!="native" else None)
    return dict(metrics=metrics, ids=ids, encoder=enc[1:], hits=hits,
                sample_trace=sampled_trace, steps=steps, inits=inits)


def pair(arms):
    a, b, c = (arms[k] for k in ARMS)
    fields = ("N", "prefill_Q", "D", "D_Q", "sample_D", "verify_D", "E", "E_Q", "F", "Q",
              "blocks", "hits", "eos", "correct")
    equality = {k: a[k] == b[k] for k in ("ids", "encoder", "hits", "sample_trace")}
    equality["ledger"] = all(a["metrics"][k] == b["metrics"][k] for k in fields)
    first = next((j for j, ids in enumerate(c["hits"])
                  if ids[:len(c["ids"][j*B:(j+1)*B])] != c["ids"][j*B:(j+1)*B]), None)
    # The paid arm has the same state up to its first changed commitment.
    equal_prefix = len(c["hits"]) if first is None else first+1
    paid_prefix = (c["sample_trace"][:equal_prefix] == b["sample_trace"][:equal_prefix] and
                   c["encoder"][:max(0, equal_prefix-1)] == b["encoder"][:max(0, equal_prefix-1)])
    return dict(matches=equality, paid_prefix=paid_prefix, paid_first_changed_block=first,
                paid_ids_equal=a["ids"]==c["ids"],
                paid_changed_deliveries=sum(block[:len(c["ids"][j*B:(j+1)*B])] != c["ids"][j*B:(j+1)*B]
                                           for j, block in enumerate(c["hits"])),
                paid_changed_encoder=sum(x != y for x, y in zip(c["hits"], c["encoder"])))


def summarize(rows, proof):
    return dict(reg=digest(proof), indices=[r["index"] for r in rows], pairs=len(rows),
        endpoint_exact=sum(all(r["matches"].values()) for r in rows),
        matching_paid_prefix=sum(r["paid_prefix"] for r in rows),
        paid_ids_equal=sum(r["paid_ids_equal"] for r in rows),
        paid_changed_deliveries=sum(r["paid_changed_deliveries"] for r in rows),
        paid_changed_encoder=sum(r["paid_changed_encoder"] for r in rows),
        multiblock=sum(r["arms"]["step"]["blocks"] > 1 for r in rows),
        arms={arm: {k: sum(r["arms"][arm][k] for r in rows) for k in
                    ("N", "prefill_Q", "D", "D_Q", "sample_D", "verify_D", "E", "E_Q", "F", "Q", "blocks",
                     "hits", "eos", "correct", "write_rows_read", "unread_speculative_rows")
                    if rows[0]["arms"][arm][k] is not None} for arm in ARMS})


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ["CUDA_VISIBLE_DEVICES"] == GPU and Path(os.environ["DG_SIGNAL"]) == args.out,
            "Require exclusive staged n164:H0 launch")
    import transformers
    import torch
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from transformers.models.diffusion_gemma import generation_diffusion_gemma
    from . import gemma_native_argmax_request, shared_h0_screen
    from .mask_pair_gemma import input_digest

    tasks = load_tasks(SAMPLES)
    proof = dict(pid=os.getpid(), uid=os.getuid(), start=time(), gpu=GPU,
                 boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                 ticks=int(Path("/proc/self/stat").read_text().split(") ")[1].split()[19]),
                 source_tree=os.environ["BDLM_SOURCE_TREE"],
                 sources={Path(p).name:file_sha(p) for p in
                     (__file__, gemma_native_argmax_request.__file__, shared_h0_screen.__file__)},
                 samples=file_sha(SAMPLES), scorer=file_sha(scoring.__file__),
                 native=file_sha(generation_diffusion_gemma.__file__), torch=str(torch.__version__),
                 transformers=transformers.__version__, panel=SIX, order=ORDER, arms=ARMS,
                 checkpoint=str(BASE/"models/diffusiongemma-cd40195"))
    require((proof["samples"], proof["scorer"], proof["native"], len(tasks), proof["transformers"]) ==
            (SAMPLES_SHA, SCORER_SHA, NATIVE_SHA, 90, "5.12.0"), "Input/stack pin mismatch")
    raw = args.out/"pairs"
    raw.mkdir(mode=0o700)
    save(args.out/"registration.json", proof)
    ctx = shared_h0_screen.load_staged_h0()
    require(torch.cuda.device_count() == 1 and
            str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-") == GPU[4:],
            "Physical GPU UUID mismatch")
    rows = []
    for n, i in enumerate(ORDER, 1):
        task = tasks[i]
        data = ctx.processor.apply_chat_template([{"role":"user", "content":task["prompt"]}],
                    tokenize=True, add_generation_prompt=True, return_dict=True,
                    return_tensors="pt").to(ctx.device)
        fingerprint = input_digest(data)
        arms = {}
        order = ARMS if n % 2 == 1 else tuple(reversed(ARMS))
        for arm in order:
            require(input_digest(data) == fingerprint, "Paired arm changed inputs")
            arms[arm] = run(ctx, data, task, arm)
        common = dict(index=i, seed=task["seed"], task_id=task["task_id"], dataset=task["dataset"],
                      reg=digest(proof), input_sha=fingerprint, arm_order=order, **pair(arms))
        path = raw/f"{i:02}.json"
        save(path, dict(common, arms=arms))
        receipt = dict(common, raw_sha=file_sha(path),
            ids_sha={k:digest(v["ids"]) for k,v in arms.items()},
            encoder_sha={k:digest(v["encoder"]) for k,v in arms.items()},
            arms={k:v["metrics"] for k,v in arms.items()})
        print(json.dumps(receipt, sort_keys=True), flush=True)
        rows.append(receipt)
        require(all(common["matches"].values()) and common["paid_prefix"],
                f"Pair {i} failed strict agreement; no expansion")
        if n == len(SIX):
            gate = summarize(rows, proof)
            require(gate["multiblock"] >= 2, "Six-case gate did not exercise multiple encoder boundaries")
            save(args.out/"qualified-six.json", gate)
        if n % 10 == 0:
            save(args.out/"progress.json", summarize(rows, proof))
    save(args.out/"complete.json", summarize(rows, proof))


if __name__ == "__main__":
    main()
