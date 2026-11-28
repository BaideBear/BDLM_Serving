"""Separate Gemma response-block admission executor; not a public serving entry.

Small-offer phase admits a full-MASK successor only through ordinary spare seats.
After that successor is opened, the registered 2B offer may pay both canvases.
The class keeps native row sampling, T2T, dirty writeback, retirement/promotion,
rebuild, verify, measured EOS and actual per-step decoder-history Q.  It does not
relax a step cap: admission uses Q64 spare seats, then the explicit 2B call.
"""
import hashlib
import json

from .gemma_fixed_dual import AblationRequest, B, TIERS
from .gemma_verified_eos import SwitchableVerifiedEOSRequest
from bdlm_serving.methods.t2t.budget_family import Law


class GemmaResponseRequest(SwitchableVerifiedEOSRequest):
    """Strict verified-EOS Live semantics plus one full-MASK response block."""

    fixed_fixture = True

    def __init__(self, model, data, *, seed, admission_offer=64, response_offer=512,
                 early=True, max_new_tokens=2048):
        if admission_offer not in (32, 64, 128) or response_offer != 2 * B:
            raise ValueError("Response executor requires a small admission offer and exact 2B")
        if type(early) is not bool:
            raise ValueError("early must be explicit")
        # Construct the dual-canvas engine at 2B so WaveKV is present, while all
        # later step widths come only from response_offer/admission_offer.
        AblationRequest.__init__(
            self, model, data, q=response_offer, seed=seed,
            max_new_tokens=max_new_tokens, ablation='10', law=Law(1, 8),
            top_gate='quarter', head_first=True, late_den=8,
            prefix_order='right', reuse_paid_masks=False,
            fuse_first_verify=True, receipt_tail_ids=True)
        self.admission_offer = int(admission_offer)
        self.response_offer = int(response_offer)
        self.early = bool(early)
        self.response_admission_tick = None
        self.response_admission_ordinal = None
        self.response_schedule = []
        self.route_F = {}
        self.route_Q = {}

    def response_offer_for(self, tick=None):
        """Small offer until the response block is open, then exact 2B."""
        if self.status != 'ready':
            raise ValueError('Completed/failed request has no next offer')
        if self.admission_offer not in (32, 64, 128) or self.response_offer != 2 * B:
            raise ValueError('Response executor has an unregistered offer schedule')
        if self.tail is None and self.response_admission_tick is None:
            return self.admission_offer
        return self.response_offer

    def step(self, q=None):
        if q is not None:
            raise ValueError('Response executor owns its two-phase offer schedule')
        offer = self.response_offer_for(self.tick)
        before_tail = self.tail
        before_ordinal = self.ordinal
        event = AblationRequest.step(self, offer)
        if before_tail is None and self.tail is not None and self.response_admission_tick is None:
            self.response_admission_tick = self.tick - 1
            self.response_admission_ordinal = self.tail.ordinal
        if event.get('decoder_Q', 0) > offer:
            raise RuntimeError('Response executor exceeded its registered step cap')
        event['response_schedule_offer'] = offer
        event['history_Q'] = event['decoder_Q']
        self.route_F[event['route']] = self.route_F.get(event['route'], 0) + 1
        self.route_Q[event['route']] = self.route_Q.get(event['route'], 0) + event['decoder_Q']
        event['response_block_opened'] = (before_tail is None and self.tail is not None)
        self.response_schedule.append(offer)
        return event


class _ResponseHandle:
    def __init__(self, request):
        self._request = request
        self.opened_prefill_q = request.origin

    @property
    def status(self):
        return self._request.status

    def step(self, _registered_offer):
        offer = self._request.response_offer_for(self._request.tick)
        event = self._request.step()
        if event.get('decoder_Q', 0) > offer:
            raise RuntimeError('Response executor exceeded its registered step cap')
        return event

    def result(self):
        if self._request.status != 'done':
            raise RuntimeError(f'Response request is {self._request.status}')
        return tuple(self._request.tokens)

    def close(self):
        pass


def run_candidate(model, data, task, *, deadline):
    """Audit one response request through the established frozen90 physical drive."""
    import json
    from types import SimpleNamespace
    from experiments.t2t.serving_slots import strict_handle_qualify as qualify
    from bdlm_serving.evaluation import scoring
    handle = None
    old_open = qualify.open_strict_gemma

    def open_response(model, data, **_kwargs):
        nonlocal handle
        request = GemmaResponseRequest(
            model, data, seed=task['seed'], admission_offer=64,
            response_offer=512, early=True)
        handle = _ResponseHandle(request)
        return handle

    setattr(qualify, 'open_strict_gemma', open_response)
    try:
        row, ids = qualify.drive(
            model, data, task, (64, 512), deadline=deadline,
            profile='fixed90_candidate')
    finally:
        setattr(qualify, 'open_strict_gemma', old_open)
    row["route_F"]=dict(request.route_F)
    row["route_Q"]=dict(request.route_Q)
    return row, ids, handle._request


def _physical_summary(row, request):
    events = row['events']
    opened = [e for e in events if e.get('response_block_opened')]
    promoted = [e for e in events if e.get('encoded')]
    rebuild = [e for e in events if e.get('route') == 'rebuild']
    verify = [e for e in events if e.get('route') in ('verify', 'verify_paid')]
    eos = request.response_admission_tick is not None
    return dict(admission_tick=request.response_admission_tick,
                admission_ordinal=request.response_admission_ordinal,
                opened_events=opened, promoted_events=promoted,
                rebuild_events=rebuild, verify_events=verify,
                response_schedule=list(request.response_schedule),
                max_decoder_Q=max(e['decoder_Q'] for e in events),
                total_decoder_Q=sum(e['decoder_Q'] for e in events),
                total_encoder_Q=sum(e['encoder_Q'] for e in events))


def run_reference(model, data, task, *, deadline):
    from .gemma_verified_eos_eval import run
    return run(model, data, task, 64, True)


def run_gpu_gate(*, checkpoint, samples, references, out, deadline):
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from bdlm_serving.evaluation import frozen90, scoring
    from bdlm_serving.evaluation.frozen import sha as file_sha
    from bdlm_serving.runtime.diffusiongemma import check_snapshot
    from pathlib import Path
    import subprocess
    import os
    expected_gpu = 'GPU-d3c3fc9c-b15d-9776-9094-7d3b85b705c1'
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected_gpu:
        raise RuntimeError('GPU gate is authorized only for physical A0 with one visible UUID')
    apps = subprocess.check_output(['nvidia-smi', '-i', '0',
        '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], text=True).strip()
    require = lambda ok, message: (_ for _ in ()).throw(RuntimeError(message)) if not ok else None
    require(transformers.__version__ == '5.12.0', 'Wrong transformers')
    require(file_sha(samples) == 'bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed',
            'Wrong frozen samples')
    check_snapshot(checkpoint)
    proc = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa', experts_implementation='grouped_mm',
        device_map={'': 'cuda:0'}).eval()
    tasks = frozen90.load_tasks(samples)
    special = (set(proc.tokenizer.all_special_ids) |
               set(model.generation_config.eos_token_id) |
               {model.generation_config.pad_token_id})
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    with torch.inference_mode():
        for index in (7, 13, 25, 33, 55, 71):
            require(torch.cuda.is_available(), 'GPU disappeared')
            task = next(t for t in tasks if t['index'] == index)
            data = proc.apply_chat_template([{'role': 'user', 'content': task['prompt']}],
                tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors='pt').to('cuda:0')
            reference_path = references / f'C64-fuse1-{index:02}.json'
            sealed = json.loads(reference_path.read_bytes())
            reference = run_reference(model, data, task, deadline=deadline)
            reference['N'] = sum(x not in special for x in reference['output_ids'])
            reference['text'] = proc.tokenizer.decode(reference['output_ids'], skip_special_tokens=True)
            reference['correct'] = bool(scoring.score(reference['text'], task)['correct'])
            replay = {k: reference[k] == sealed[k] for k in
                      ('output_ids', 'input_sha256',
                       'N', 'correct', 'decoder_F', 'decoder_Q', 'encoder_F',
                       'encoder_Q', 'F', 'route_F', 'route_Q')}
            if not all(replay.values()):
                raise RuntimeError(f'Strict Live replay mismatch at {index}: {replay}')
            apps = subprocess.check_output(['nvidia-smi', '-i', '0',
                '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], text=True).strip()
            if apps != f"{os.getpid()}, {expected_gpu}":
                raise RuntimeError(f'Unauthorized A0 state during gate: {apps!r}')
            candidate_row, ids, request = run_candidate(
                model, data, task, deadline=deadline)
            candidate = dict(N=sum(x not in special for x in ids),
                             correct=bool(scoring.score(
                                 proc.tokenizer.decode(ids, skip_special_tokens=True), task)['correct']),
                             decoder_F=candidate_row['decoder_F'], decoder_Q=candidate_row['decoder_Q'],
                             encoder_F=candidate_row['encoder_F'], encoder_Q=candidate_row['encoder_Q'],
                             F=candidate_row['F'], route_F=candidate_row['route_F'], route_Q=candidate_row['route_Q'],
                             output_sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest())
            physical = _physical_summary(candidate_row, request)
            case = dict(index=index, task_id=task['task_id'], dataset=task['dataset'],
                        seed=task['seed'], strict_live_replay=replay, reference=reference,
                        response=candidate, response_physical=physical,
                        response_raw=row, delta={k: candidate[k] - reference[k] for k in
                                                 ('N', 'F', 'decoder_F', 'decoder_Q',
                                                  'encoder_F', 'encoder_Q')},
                        correct_delta=int(candidate['correct']) - int(reference['correct']))
            cases.append(case)
            (out / f'case-{index:02}.json').write_text(json.dumps(
                case, separators=(',', ':'), allow_nan=False) + '\n')
            print('CASE ' + json.dumps({k: case[k] for k in
                  ('index', 'strict_live_replay', 'response_physical', 'delta', 'correct_delta')},
                  separators=(',', ':'), allow_nan=False), flush=True)
    summary = dict(schema='gemma-response-executor-a0-gate-v1', gpu=expected_gpu,
                   cases=list(range(7, 72)), admission_offers=[64], response_offer=512,
                   strict_live_profile='dense_stop_joint_verified_eos_q64',
                   cases_measured=len(cases), case_indices=[c["index"] for c in cases], transformers=transformers.__version__)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main() -> int:
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    gate = sub.add_parser('gpu-gate')
    for key in ('checkpoint', 'samples', 'references', 'out'):
        gate.add_argument('--' + key, type=Path, required=True)
    gate.add_argument('--deadline', type=int, required=True)
    args = ap.parse_args()
    try:
        summary = run_gpu_gate(checkpoint=args.checkpoint, samples=args.samples,
                               references=args.references, out=args.out,
                               deadline=args.deadline)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print('FAILURE ' + json.dumps(dict(error=repr(exc)), separators=(',', ':')),
              flush=True)
        return 5
    print('SUMMARY ' + json.dumps({k: summary[k] for k in
          ('schema', 'gpu', 'cases_measured', 'case_indices')},
          separators=(',', ':')), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
