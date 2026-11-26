from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch

from bdlm_serving.methods.t2t import admission_gate as ag, embedding_context as ec, online
from bdlm_serving.methods.t2t.llada_sparse import LLaDASparseKernel
from bdlm_serving.methods.t2t.protocol import Arm
from bdlm_serving.methods.t2t.seat_audit import SeatAudit
from test_t2t_llada_sparse import _Layer, _Toy, _state
from tools import run_t2t as runner


@pytest.mark.parametrize('mode', ag.MODES)
def test_only_registered_arms(mode):
    assert Arm.parse(f'8:0:7:{mode}', 32).edit == 1
    for spec, width in [('8:0:7', 256), ('4:0:3', 32), ('8:1:6', 32), ('8:0:6', 32)]:
        with pytest.raises(ValueError):
            Arm.parse(f'{spec}:{mode}', width)
    with pytest.raises(ValueError, match='fit'):
        LLaDASparseKernel(_Toy().eval(), 8, mask_slots=7, policy=mode, context_ranker=object())


@pytest.mark.parametrize('mode', ag.MODES)
def test_first_approved_only_same_rank_and_victim_twenty_layer_work(monkeypatch, mode):
    model = _Toy(edit=False).eval()
    model.model.layers.extend(_Layer() for _ in range(18))
    calls, fit = [], object()

    def decide(move, loaded, name, seed, block, clock):
        assert loaded is fit and name == mode and move['reason'] == 'swap_mask'
        assert seed == 29 and len(model.embedding_rows) == clock - 1  # pre-forward
        calls.append((block, clock))
        return len(calls) == 2, .25

    monkeypatch.setattr(ag, 'admission_decision', decide)
    ranker = SimpleNamespace(scores=lambda before, fresh, old, pos: old)
    kernel = LLaDASparseKernel(model, 8, mask_slots=7, policy=mode, context_ranker=ranker, admission_fit=fit)
    s = _state(kernel, gen_length=63)
    kernel.step(s)
    s.x[0, 52:62] = 3  # stable outside candidates alongside unresolved MASK rows
    kernel.run(s)
    anchor = None
    assert len(calls) == 2
    for e in s.events:
        q = e['physical_q']
        assert e['layer_physical_q'] == [q]*20 and q == len(e['absolute_positions']) and e['F'] == 1
        screen = e.get('crowded_screen')
        assert ('crowded_screen' in e) == (e['stage'] == 'selected' and anchor is None)
        assert ('admission_decision' in e) == bool(screen and screen['reason'] == 'swap_mask')
        keep = e.get('admission_decision', {}).get('keep', False)
        assert ('intervention' in e) == keep
        if not keep:
            assert not e['reserved_extra_edit_positions']
        if screen:
            base = set(screen['base'])
            expected = base - {screen['victim']} | {screen['rank']} if keep else base
            assert {i - 32*e['block'] for i in e['absolute_positions']} == expected
        if 'admission_decision' in e:
            assert (e['block'],e['hard_clock']) == calls[int(keep)]
            assert e['admission_decision']['risk_probability'] == .25
        if keep:
            anchor = e
            p = e['intervention']
            assert p['executed'] == p['targeted'] and p['candidate'] == screen['rank'] == screen['entrant']
            assert p['victim'] == screen['victim'] and not p['victim_queried']
            assert len(p['pre_sha256']) == len(p['post_rng_sha256']) == len(e['accepted_sha256']) == 64
            assert 'resident_l15' in p and 'beneficiary' in e
    assert anchor and s.events[-1]['block'] > anchor['block']
    assert [e['physical_q'] for e in s.events] == model.embedding_rows
    assert {'prefill','bootstrap','selected','verify','commit'} <= {e['stage'] for e in s.events}


def test_slack_offer_is_not_a_timing_decision(monkeypatch):
    monkeypatch.setattr(ag, 'admission_decision', lambda *_: pytest.fail('Called on slack'))
    mode = ag.MODES[0]
    kernel = LLaDASparseKernel(_Toy().eval(), 8, mask_slots=7, policy=mode,
                               admission_fit={}, context_ranker=SimpleNamespace(scores=lambda _, f, old, pos: old))
    s = SimpleNamespace(history=SimpleNamespace(fresh=set()), events=[], seat_audit=SeatAudit(mode),
                        selection_seed=11, previous_strict=set(), last_write=[-1]*32, block=2)
    picked, reserve = kernel._hard_pick(s, range(32), [i > 1 for i in range(32)], [0]*32)
    assert s.context_move['reason'] == 'swap_old' and not s.seat_audit.used
    assert picked == s.context_move['base'] and not reserve


def test_runner_checks_fit_before_backend_and_hashes_loaded_artifacts(tmp_path, monkeypatch):
    weights, fit = tmp_path/'weights.json', tmp_path/'fit.json'
    weights.write_text(json.dumps(dict(**ec.DESIGN, weights=[0.]*28, calibration_sha256=ag.CAL_SHA)))
    monkeypatch.setattr(ag, 'RANK_SHA', runner.sha(weights))
    expected = dict(schema=ag.SCHEMA, calibration_sha256=ag.CAL_SHA, ranker_sha256=ag.RANK_SHA,
                    features=list(ag.FEATURES), weights=[0.]*8, cutoff=.5, keep_probability=.4, fit_requests=24)
    fit.write_text(json.dumps(expected))
    samples = tmp_path/'samples.json'
    samples.write_text(json.dumps({'samples': []}))
    args = SimpleNamespace(model='llada21', samples=samples, reference=None, checkpoint=None, context_weights=weights,
                           arm=['8:0:7:'+mode for mode in ag.MODES], output=tmp_path/'run')
    monkeypatch.setattr(runner, 'load_tasks', lambda _: [])
    monkeypatch.setattr(ec, 'input_table', lambda _: nullcontext(None))
    monkeypatch.setattr(ec, 'Projector', lambda _: SimpleNamespace(unit=None))
    monkeypatch.delenv('T2T_ADMISSION_FIT', raising=False)
    with pytest.raises(ValueError, match='require --admission-fit'):
        runner.main(args)
    monkeypatch.setenv('T2T_ADMISSION_FIT', str(fit))
    args.admission_fit = tmp_path/'bad.json'
    args.admission_fit.write_text('{}')
    with pytest.raises(ValueError, match='Invalid admission fit'):
        runner.main(args)  # explicit path wins
    del args.admission_fit
    assert not args.output.exists()
    def backend(args, prepare, *, context_ranker, admission_fit):
        assert admission_fit == expected and context_ranker is not None
        raise StopIteration  # inspect run.json without even initializing a model
    monkeypatch.setattr(online, 'load_backend', backend)
    with pytest.raises(StopIteration):
        runner.main(args)
    meta = json.loads((args.output/'run.json').read_text())
    assert meta['suite'] == 'frozen30' and meta['samples_sha256'] == runner.sha(samples)
    assert meta['admission_fit_sha256'] == runner.sha(fit) and meta['context_weights_sha256'] == runner.sha(weights)
    assert meta['policies'] == ['t2t_q8_f0_m7_'+m for m in ag.MODES]
    weights.write_text(weights.read_text() + '\n')
    with pytest.raises(ValueError, match='frozen gate ranker'):
        runner.main(args)
    args.arm, args.admission_fit = ['8:0:8:hard_noedit'], fit
    with pytest.raises(ValueError, match='requires an admission arm'):
        runner.main(args)
