"""Physical A0 semantic acceptance for fixed-Q64 full-MASK admission."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from .gemma_full_admission import FullAdmissionRequest, open_full_admission

A0_UUID = "GPU-d3c3fc9c-b15d-9776-9094-7d3b85b705c1"
SIX = (7, 13, 25, 33, 55, 71)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard(stage):
    if sys.platform == "darwin":
        return
    apps = subprocess.check_output(["nvidia-smi", "-i", "0",
        "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
        text=True).strip()
    uuid = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid",
        "--format=csv,noheader"], text=True).strip()
    if uuid != A0_UUID or __import__("os").environ.get("CUDA_VISIBLE_DEVICES") != A0_UUID:
        raise RuntimeError(f"A0 UUID mismatch: {uuid!r}")
    if stage == "before":
        memory, utilization = subprocess.check_output(["nvidia-smi", "-i", "0",
            "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True).strip().split(",")
        if memory.strip() != "0" or utilization.strip() != "0":
            raise RuntimeError(f"A0 not idle: {memory.strip()} MiB / {utilization.strip()}%")
    expected = "" if stage == "before" else f"{__import__('os').getpid()}, {A0_UUID}"
    if apps != expected:
        raise RuntimeError(f"Unexpected A0 apps at {stage}: {apps!r}")


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def run_full_admission(model, data, task, *, q=64, early=True):
    from bdlm_serving.runtime.diffusiongemma import observe
    from .gemma_strict_eval import encoder_tape
    from .gemma_unified_audit import projectors
    from .mask_pair_gemma import input_digest
    prompt = data["input_ids"][0].tolist()
    origin = len(prompt)
    route_f, route_q = Counter(), Counter()
    tape, ledger = [], [("encoder", origin)]
    enc_expected = [(prompt, list(range(origin)))]
    tokens, sealed, debt = [], [], set()
    top = []
    age = 0
    admission = None
    with ExitStack() as stack:
        physical = stack.enter_context(observe(model))
        projection, positions = stack.enter_context(projectors(model))
        encoder = stack.enter_context(encoder_tape(model))
        stack.enter_context(model.register_forward_hook(
            lambda _, arguments, output: top.append(output.logits[0].argmax(-1).tolist())))
        request = FullAdmissionRequest(model, data, seed=task["seed"],
                                       initial_offer=q, early=early)
        while request.status == 'ready':
            tick = len(tape)
            if tick >= 4096:
                raise RuntimeError("Unbounded request")
            event = request.step()
            part, = event['parts']
            route = event['route']
            actual_rows = positions[-1]
            width = len(actual_rows)
            nominal = event['nominal_q']
            ordinal = len(enc_expected) - 1
            base = origin + 256 * ordinal
            if tick != event['tick'] or len(positions) != len(top) != tick + 1:
                raise RuntimeError("Tick/top mismatch")
            if event['offer'] != q or event['nominal_q'] != q:
                raise RuntimeError("Offer changed")
            if event['decoder_Q'] != width or event['decoder_F'] != 1:
                raise RuntimeError("Forward bill mismatch")
            if event['ordinal'] != part['ordinal'] != ordinal:
                raise RuntimeError("Canvas ordinal mismatch")
            if event['positions'] != actual_rows or actual_rows != [base + row for row in part['rows']]:
                raise RuntimeError("Position mismatch")
            if actual_rows != sorted(set(actual_rows)):
                raise RuntimeError("Duplicate physical rows")
            if len(projection) != 90 or any(value[-1] != width for value in projection.values()):
                raise RuntimeError("Per-layer projection width mismatch")
            if width > nominal:
                certificate = event.get('full_admission') or {}
                if not (event.get('full_admission') and ordinal > 0
                        and certificate.get('explainable') and certificate.get('physical_q') == width
                        and certificate.get('valid_rows') == width
                        and part['rows'] == list(range(256))):
                    raise RuntimeError("Uncertified over-offer projection")
            sampled = event['sampled_absolute']
            writes = {base + row for key in ('admitted', 'revoked', 'input_changed')
                      for row in part[key]}
            if event['dirty_before_absolute'] != sorted(debt):
                raise RuntimeError("Dirty-before mismatch")
            if set(sorted(debt)[:q]) - set(actual_rows):
                raise RuntimeError("Unpaid debt omitted")
            if not set(writes) <= set(sampled) <= set(actual_rows):
                raise RuntimeError("Unmeasured write or sampling")
            if part['age_before'] != age or part['age_after'] != age + bool(sampled):
                raise RuntimeError("Sampler clock mismatch")
            debt = (debt - set(actual_rows)) | writes
            age = part['age_after']
            if event['dirty_after_absolute'] != sorted(debt):
                raise RuntimeError("Lost write")
            ledger.append(('decoder', width))
            if event['encoder_F'] != int(event['encoded']) or event['encoder_Q'] != 256 * event['encoder_F']:
                raise RuntimeError("Encoder bill mismatch")
            route_f[route] += 1
            route_q[route] += width
            tape.append(digest([{key: value for key, value in event.items()
                                 if key != 'verified_eos_end'}, top[-1]]))
            if route not in ('verify', 'verify_paid'):
                if sealed or event['encoded'] or 'verified_eos_end' in event:
                    raise RuntimeError("Draft short circuit")
                continue
            start = len(sealed)
            if part['phase'] != 'verify' or sampled or debt or width != q:
                raise RuntimeError("Unordered or paid verification")
            if actual_rows != list(range(base + start, base + start + q)):
                raise RuntimeError("Noncontiguous verification")
            sealed.extend(top[-1])
            local = next((row for row in range(start, start + q)
                          if sealed[row] in request.eos), None)
            if local is not None and admission is None:
                admission = dict(tick=tick, ordinal=ordinal, eos_row=local,
                                 chunk_start=start, chunk_end=start + q)
            cut = bool(early and admission is not None and len(sealed) < 256)
            if ('verified_eos_end' in event) != cut:
                raise RuntimeError("Missed or spurious early EOS")
            if cut or len(sealed) == 256:
                tokens.extend(sealed if admission is None else sealed[:admission['eos_row'] + 1])
                terminal = admission is not None or len(tokens) == 2048
                if event['encoded'] != (not terminal) or (request.status == 'done') != terminal:
                    raise RuntimeError("Encoder or EOS boundary mismatch")
                if event['encoded']:
                    if len(sealed) != 256 or admission is not None:
                        raise RuntimeError("Incomplete nonterminal commit")
                    enc_expected.append((sealed, list(range(base, base + 256))))
                    ledger.append(('encoder', 256))
                    age = 0
                sealed = []
        if debt or sealed or request.tokens != tokens:
            raise RuntimeError("Final tokens or debt mismatch")
        if encoder != enc_expected or physical != ledger:
            raise RuntimeError("Encoder or physical ledger mismatch")
        if any(len(values) != len(tape) for values in projection.values()):
            raise RuntimeError("Projection ledger length mismatch")
    return dict(output_ids=tokens, tape=tape, input_sha256=input_digest(data),
                hit=admission, encoder_digest=digest(encoder), prefix_order='full_admission',
                decoder_F=len(tape), decoder_Q=sum(route_q.values()),
                encoder_F=len(encoder) - 1, encoder_Q=256 * (len(encoder) - 1),
                F=len(tape) + len(encoder) - 1, route_F=dict(route_f),
                route_Q=dict(route_q), admission_events=[])


def run_parent(model, data, task, *, q=64, early=True):
    from .gemma_verified_eos_eval import run
    return run(model, data, task, q, early)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('/gs/models/diffusiongemma-cd40195'))
    parser.add_argument('--samples', type=Path, default=Path('/gs/code/samples.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--deadline', type=int, required=True)
    args = parser.parse_args()
    guard('before')
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.runtime.diffusiongemma import check_snapshot
    if transformers.__version__ != '5.12.0':
        raise RuntimeError('Wrong transformers version')
    if sha(args.samples) != 'bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed':
        raise RuntimeError('Wrong frozen samples')
    check_snapshot(args.checkpoint)
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.checkpoint, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa', experts_implementation='grouped_mm',
        device_map={'': 'cuda:0'}).eval()
    tasks = load_tasks(args.samples)
    special = (set(processor.tokenizer.all_special_ids) |
               set(model.generation_config.eos_token_id) |
               {model.generation_config.pad_token_id})
    args.out.mkdir(parents=True, exist_ok=False)
    cases = []
    with torch.inference_mode():
        for index in SIX:
            if time.time() >= args.deadline:
                raise RuntimeError('Deadline')
            guard('request')
            task = next(task for task in tasks if task['index'] == index)
            data = processor.apply_chat_template(
                [{'role': 'user', 'content': task['prompt']}], tokenize=True,
                add_generation_prompt=True, return_dict=True,
                return_tensors='pt').to('cuda:0')
            parent = run_parent(model, data, task, q=64, early=True)
            parent['N'] = sum(token not in special for token in parent['output_ids'])
            parent['text'] = processor.tokenizer.decode(parent['output_ids'], skip_special_tokens=True)
            parent['correct'] = bool(scoring.score(parent['text'], task)['correct'])
            candidate = run_full_admission(model, data, task, q=64, early=True)
            candidate['N'] = sum(token not in special for token in candidate['output_ids'])
            candidate['text'] = processor.tokenizer.decode(candidate['output_ids'], skip_special_tokens=True)
            candidate['correct'] = bool(scoring.score(candidate['text'], task)['correct'])
            first_canvas_parent = [event for event in
                                   # Parent eval does not retain raw events; use tape until candidate fork.
                                   range(len(candidate['tape']))]
            case = dict(index=index, task_id=task['task_id'], dataset=task['dataset'],
                        seed=task['seed'], parent=parent, candidate=candidate,
                        delta={key: candidate[key] - parent[key] for key in
                               ('N', 'F', 'decoder_F', 'decoder_Q', 'encoder_F', 'encoder_Q')},
                        correct_delta=int(candidate['correct']) - int(parent['correct']))
            cases.append(case)
            compact = {key: case[key] for key in
                       ('index', 'delta', 'correct_delta')}
            compact['parent'] = {key: parent[key] for key in
                                 ('N', 'correct', 'F', 'decoder_F', 'decoder_Q',
                                  'encoder_F', 'encoder_Q', 'route_F', 'route_Q')}
            compact['candidate'] = {key: candidate[key] for key in
                                    ('N', 'correct', 'F', 'decoder_F', 'decoder_Q',
                                     'encoder_F', 'encoder_Q', 'route_F', 'route_Q',
                                     'prefix_order')}
            (args.out / f'case-{index:02}.json').write_text(json.dumps(
                compact, separators=(',', ':'), allow_nan=False) + '\n')
            print('CASE ' + json.dumps(compact, separators=(',', ':'), allow_nan=False), flush=True)
    summary = dict(schema='gemma-full-admission-q64-a0-semantic-v1',
                   gpu=A0_UUID, q=64, cases=list(SIX), cases_measured=len(cases),
                   transformers=transformers.__version__)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('SUMMARY ' + json.dumps(summary, separators=(',', ':')), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
