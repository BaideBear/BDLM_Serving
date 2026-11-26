import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tools import native_kv_drift as drift


def test_full_width_same_canvas_capture_and_default(tmp_path, monkeypatch):
    rms = Mock(wraps=drift.relative_rms)
    monkeypatch.setattr(drift, 'relative_rms', rms)
    task = dict(index=3, dataset='math', task_id='test', seed=7)
    outputs = []
    for mode in ({}, {'lag_joint': True}):
        state = drift.DriftState(2, 15, **mode)
        rms.reset_mock()
        out = tmp_path / str(len(outputs))
        out.mkdir()
        # The first transition in each canvas has no old rows: both were just admitted.
        for block, nsteps in [(8, 3), (10, 2)]:
            before = [0, 0]
            for j in range(nsteps):
                state.begin(block, 3-j, before)
                k = torch.tensor([2**j, 1.0]).reshape(1, 1, 2, 1)
                v = torch.tensor([1.0, 3**j]).reshape(1, 1, 2, 1)
                before = [1, 2]
                state.end(before, [True, True], k, v)
        drift.write_task(out, task, state)
        assert len(state.records) == 1 and state.empty_pool_steps == 2
        # One K and one V calculation per required transition, never twice.
        assert rms.call_count == (6 if mode else 2)
        outputs.append((out / '03-kvdrift-layer15.json').read_bytes())
        if not mode:
            assert sorted(p.name for p in out.iterdir()) == ['03-kvdrift-layer15.json']
        else:
            payload = json.loads((out / '03-lag-joint-layer15.json').read_text())
            assert payload == dict(**task, canvas=2, layer=15, metric='max_dk_dv', frames=[
                dict(block=b, clock=c, joint=[1., 2.]) for b,c in ((8,2),(8,1),(10,2))])
            assert all(type(x) is float for f in payload['frames'] for x in f['joint'])
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize('cli,env,enabled', [(False,None,False),(False,'0',False),
                                              (False,'1',True),(True,None,True)])
def test_main_opt_in(cli, env, enabled, monkeypatch, tmp_path):
    monkeypatch.delenv('DG_KV_LAG_JOINT', raising=False)
    if env is not None:
        monkeypatch.setenv('DG_KV_LAG_JOINT', env)
    args = {k:tmp_path for k in ('checkpoint','reference','samples','output')}
    if cli:
        monkeypatch.setattr(sys, 'argv', ['drift', '--lag-joint', *[v for k in args for v in ('--'+k,str(tmp_path))]])
        args = drift.parse_args()
    else:
        args = SimpleNamespace(**args)  # Protected launcher has no lag_joint attribute.
    factory = Mock()
    monkeypatch.setattr(drift, 'kv_drift_trace', factory)
    monkeypatch.setattr(drift, 'load_tasks', lambda _: list(range(30)))
    class AtNativeBoundary(Exception):
        pass
    monkeypatch.setattr(drift, 'run_native', Mock(side_effect=AtNativeBoundary))
    with pytest.raises(AtNativeBoundary):
        drift.main(args)
    assert factory.call_args.kwargs['lag_joint'] is enabled
