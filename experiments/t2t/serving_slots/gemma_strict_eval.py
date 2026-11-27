"""One frozen90 request and one strict B256 arm per invocation. No GPU scheduling."""
import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path

from bdlm_serving.evaluation import scoring
from bdlm_serving.evaluation.frozen import ROOT, sha
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.runtime.diffusiongemma import STUDY, check_snapshot, observe
from .gemma_fixed_caps import DATA, SAMPLES_SHA
from .gemma_unified_audit import projectors
from .mask_pair_gemma import generation_settings, input_digest

B, CAP = 256, 2048
TIERS, CASES = (32, 64, 128, 256), (0, 10, 20)
SCORER_SHA = "a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692"


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def check_arm(q, k, index):
    require(type(q) is int and q in TIERS and type(k) is int and 0 <= k <= q
            and type(index) is int and index in CASES, "Not a registered case/decoder arm")


def local(e, key):
    xs = e[key]
    require(isinstance(xs, list) and all(type(x) is int and 0 <= x < B for x in xs)
            and len(xs) == len(set(xs)), f"Invalid block-local {key}")
    return set(xs)


def audit(prompt, tokens, trace, events, positions, projections, encoders, *, q, k, eos, special, pad):
    """Check the independent physical tape before scoring. N excludes special tokens."""
    require(0 < len(tokens) <= CAP and all(type(x) is int and x >= 0 for x in tokens),
            "Invalid returned suffix")
    stop = next((i for i, t in enumerate(tokens) if t in eos), None)
    require(stop == len(tokens)-1 if stop is not None else len(tokens) == CAP, "Not first EOS or cap")
    widths = [e["q"] for e in trace]
    require(trace and len(positions) == len(trace) and projections and
            all(t == widths for t in projections.values()), "Layer projection/call mismatch")
    origin = len(prompt)
    expected = [("encoder", origin)]
    snapshots = [(prompt, list(range(origin)))]
    full = set(range(B))
    accepted, seen, verified = set(), set(), set()
    phase_f, phase_q = Counter(), Counter()
    changes = Counter()
    block = prefix_calls = opportunities = available = prefix_rows = 0
    for i, (e, pos) in enumerate(zip(trace, positions, strict=True)):
        phase, w = e["phase"], e["q"]
        require(type(e["offer"]) is int and e["offer"] == q and type(w) is int and 0 < w <= q
                and type(e["ordinal"]) is int and e["ordinal"] == block and type(e["encoded"]) is bool
                and phase in ("sample", "drain", "verify"), f"Invalid call {i}")
        rows, pref, sampled, dirty, add, edit, revoke = (
            local(e, key) for key in ("rows", "prefix", "sampled", "dirty_before", "admitted",
                                      "edited", "revoked"))
        base = origin + B*block
        require(len(rows) == w and e["rows"] == sorted(rows)
                and pos == [base+r for r in sorted(rows)], "Decoder position mismatch")
        require(pref <= sampled <= rows and add | edit | revoke <= sampled
                and not add & accepted and edit | revoke <= accepted and not edit & revoke
                and bool(sampled) == (phase == "sample"), "Invalid sampler ledger")
        if phase == "verify":
            require(seen == full and not dirty and not verified & rows, "Invalid terminal sweep")
            verified.update(rows)
        else:
            require(not verified and set(sorted(dirty)[:q]) <= rows, "Missing debt or sampling after verify")
        if phase == "sample" and seen == full:
            # Seats before optional spending. Dirty includes renoise, not just admission flips.
            end = min(full-accepted, default=B)
            pool = sorted(set(range(end)) - dirty)
            mandatory = rows - pref
            require(len(mandatory) == min(q, len(dirty | (full-accepted))) and
                    mandatory <= dirty | (full-accepted), "Unexpected mandatory warm rows")
            seats = min(q-len(mandatory), len(pool))
            take = min(k, seats)
            require(pref == set(pool[len(pool)-take:]), "Not the rightmost eligible prefix")
            opportunities += seats > 0
            available += seats
        else:
            require(not pref, "Prefix outside warm sample")
        prefix_calls += bool(pref)
        prefix_rows += len(pref)
        changes.update(admitted=len(add), edited=len(edit), revoked=len(revoke),
                       prefix_edited=len(edit & pref), prefix_revoked=len(revoke & pref))
        accepted = (accepted | add) - revoke
        seen.update(sampled)
        phase_f[phase] += 1
        phase_q[phase] += w
        expected.append(("decoder", w))
        if verified == full:
            require(e["encoded"] == (i+1 < len(trace)), "Wrong terminal encoder flag")
        else:
            require(not e["encoded"] and i+1 < len(trace), "Premature end/encode")
        if e["encoded"]:
            expected.append(("encoder", B))
            snapshots.append((tokens[B*block:B*(block+1)], list(range(base, base+B))))
            block += 1
            accepted, seen, verified = set(), set(), set()
    require(block*B < len(tokens) <= (block+1)*B and expected == events,
            "Output blocks or physical encoder/decoder order mismatch")
    require(snapshots == encoders, "Encoder input IDs/absolute positions disagree with committed output")
    n = sum(t not in special and t not in eos and t != pad for t in tokens)
    f, paid = len(events)-1, sum(width for _, width in events[1:])
    return dict(N=n, F=f, Q=paid, N_per_F=n/f, N_per_Q=n/paid, prefill_rows=origin,
                decoder_F=len(trace), decoder_Q=sum(widths), decoder_q_max=max(widths),
                encoder_F=block, encoder_Q=block*B, phase_F=dict(phase_f), phase_Q=dict(phase_q),
                eos=stop is not None, truncated=stop is None, output_ids_count=len(tokens),
                prefix_calls=prefix_calls, prefix_rows=prefix_rows, eligible_warm_calls=opportunities,
                eligible_warm_seats=available, **changes)


@contextmanager
def encoder_tape(model):
    tape = []

    def record(_, args, kw):
        ids, pos = kw["input_ids"], kw["position_ids"]
        require(ids.ndim == 2 and ids.shape[0] == 1 and pos.shape == ids.shape, "Encoder shape")
        tape.append((ids[0].tolist(), pos[0].tolist()))

    with model.model.encoder.register_forward_pre_hook(record, with_kwargs=True):
        yield tape


def evaluate(model, processor, task, *, q=128, k=0):
    """For one already-loaded worker; returns (compact metrics, raw evidence). Serial use."""
    import torch
    from .gemma_strict_request import generate

    check_arm(q, k, task["index"])
    require(not model.training and model.config.canvas_length == B and
            model.generation_config.max_denoising_steps == 48, "Unexpected native configuration")
    data = processor.apply_chat_template([{"role": "user", "content": task["prompt"]}], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    prompt = data["input_ids"][0].tolist()
    digest = input_digest(data)
    with torch.inference_mode(), observe(model) as events, projectors(model) as (proj, pos), \
            encoder_tape(model) as enc:
        tokens, trace = generate(model, data, seed=task["seed"], offers=(q,), prefix_quota=k,
                                 prefix_mode="right", max_new_tokens=CAP)
    cfg, tok = model.generation_config, processor.tokenizer
    eos = cfg.eos_token_id
    m = audit(prompt, tokens, trace, events, pos, proj, enc, q=q, k=k,
              eos={eos} if isinstance(eos, int) else set(eos or []),
              special=set(tok.all_special_ids), pad=cfg.pad_token_id)
    text = tok.decode(tokens, skip_special_tokens=True)
    score = scoring.score(text, task)
    compact = {key: task[key] for key in ("index", "seed", "task_id", "dataset")}
    compact.update(q=q, k=k, prefix_mode="right", cap=CAP, input_sha256=digest,
                   device=str(model.device), correct=bool(score["correct"]), metric=score["metric"],
                   layers=len(model.model.decoder.layers), projection_modules=len(proj), **m)
    return compact, dict(output_ids=tokens, output_text=text, score=score, trace=trace, events=events,
                         decoder_position_ids=pos, encoder_inputs=enc,
                         projection_widths={f"{i}.{key}": t for (i, key), t in proj.items()})


def write(path, value):
    tmp = path.with_suffix(".tmp")
    with tmp.open("x") as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--q", type=int, choices=TIERS, default=128)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--index", type=int, choices=CASES, default=0)
    ap.add_argument("--output", type=Path, required=True, help="Outside checkout; raw/ and compact/ below")
    a = ap.parse_args()
    check_arm(a.q, a.k, a.index)
    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if a.device == "cpu":
        require(visible == "" and os.sched_getaffinity(0) == set(range(32, 36)),
                "CPU requires hidden GPUs and taskset -c 32-35")
    else:
        require(visible and visible != "-1" and "," not in visible, "Worker needs one assigned visible GPU")
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    if a.device == "cpu":
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
    native = json.loads((STUDY/"config.json").read_text())
    require(native["transformers"] == transformers.__version__ and
            (native["canvas"], native["cap"], native["attention"], native["dtype"], native["experts"])
            == (B, CAP, "sdpa", "bfloat16", "grouped_mm"), "Wrong runtime or frozen settings")
    samples, ckpt = DATA/"samples.json", ROOT.parent.parent/"models/diffusiongemma-cd40195"
    require(sha(samples) == SAMPLES_SHA and sha(Path(scoring.__file__)) == SCORER_SHA,
            "Frozen samples/scorer changed")
    task = load_tasks(samples)[a.index]
    tag = f"{a.device}-q{a.q}-k{a.k}-{a.index:02d}"
    root = a.output.resolve()
    require(not root.is_relative_to(ROOT.resolve()), "Raw artifacts must stay outside checkout")
    raw, compact = root/"raw"/tag, root/"compact"/(tag+".json")
    compact.parent.mkdir(exist_ok=True, parents=True)
    require(not compact.exists(), "Already scored")
    raw.mkdir(exist_ok=False, parents=True)
    files = ("gemma_strict_eval", "gemma_strict_request", "gemma_strict_sampler", "gemma_strict_rows",
             "gemma_strict_project", "gemma_wave_kv", "gemma_unified", "gemma_unified_audit")
    meta = dict(checkpoint=str(ckpt), native=native, samples_sha256=SAMPLES_SHA, scorer_sha256=SCORER_SHA,
                torch=str(torch.__version__), transformers=transformers.__version__, pid=os.getpid(),
                affinity=sorted(os.sched_getaffinity(0)), case=task["task_id"], seed=task["seed"],
                q=a.q, k=a.k, device=a.device, index=a.index,
                sources={s: sha(Path(__file__).with_name(s+".py")) for s in files})
    write(raw/"run.json", meta)
    print(f"{tag}: verifying checkpoint; pid={os.getpid()}", flush=True)
    check_snapshot(ckpt)
    print(f"{tag}: loading verified weights", flush=True)
    processor = AutoProcessor.from_pretrained(ckpt, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(ckpt, local_files_only=True,
        dtype=torch.bfloat16, device_map={"": a.device}, attn_implementation=native["attention"],
        experts_implementation=native["experts"]).eval()
    print(f"{tag}: generating, cap={CAP}", flush=True)
    metrics, evidence = evaluate(model, processor, task, q=a.q, k=a.k)
    write(raw/"evidence.json", dict(**evidence, generation_config=generation_settings(model.generation_config)))
    write(compact, dict(**metrics, revision=native["revision"], samples_sha256=SAMPLES_SHA,
                        scorer_sha256=SCORER_SHA, run_sha256=sha(raw/"run.json")))
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
