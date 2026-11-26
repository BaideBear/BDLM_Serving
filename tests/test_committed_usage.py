import gzip
import json
from math import log1p

import numpy as np
import pytest

from bdlm_serving.methods.t2t.committed_usage import CommittedUsage
from tools.edit_opportunity import load


def test_missing_masks_match_and_alternative_counts():
    h=CommittedUsage()
    assert h.features([1],[2,3,4],0)==[[0.]*6]
    h.commit([2,3,4,2,5,4])
    assert h.features([1],[2,3,4],0)==[[log1p(1)]*6]
    assert h.features([1],[0,3,4],0)==[[0.,0.,log1p(1),log1p(1),0.,0.]]
    assert h.features([0],[6],0)[0][:2]==[0.,log1p(1)]
    h.commit([9]*256)
    assert h.features([1],[2,3,4],0)==[[0.]*6]
    assert h.features([1],[9]*3,0)==[[log1p(16),0.]*3]


def test_only_previous_committed_output_is_counted(tmp_path):
    last=[33]*28+[1,2,3,4]
    b0=last[:28]+[0]*4
    c0=[0]*32
    c1=[1,2,3,0]+[0]*28
    task=dict(index=0,task_id="toy",dataset="toy",seed=42)
    def trace(end,before=last):
        p=tmp_path/"trace.gz"
        with gzip.open(p,"wt") as s:
            for r in [task,dict(prompt_length=28,mask_id=0)]+[
                dict(block=z,before=a,after=b) for z,a,b in
                [(0,b0,before),(32,c0,c1),(32,c1,c1),(32,c1,end)]]:
                s.write(json.dumps(r)+"\n")
        return load(p,task,prefix_factory=CommittedUsage)
    r=trace(c1)[-1]
    np.testing.assert_allclose(r[-2][1,-6:],[log1p(1),0.]*3)
    assert trace([8]+c1[1:])[-1][-1]=={0}
    np.testing.assert_array_equal(trace([8]+c1[1:])[-1][-2],r[-2])
    with pytest.raises(ValueError,match="not committed"):
        trace(c1,b0)


def test_own_commits_before_live_ranking():
    from test_t2t_llada_sparse import _GateToy, _state
    from bdlm_serving.methods.t2t.llada_sparse import LLaDASparseKernel
    from bdlm_serving.methods.t2t.embedding_context import ContextRanker
    from bdlm_serving.methods.t2t.committed_usage import UsageRanker
    from bdlm_serving.methods.t2t.protocol import Arm

    model = _GateToy().eval()
    calls = []
    w = np.arange(28)
    unit = lambda _: np.ones(16)/4
    base = ContextRanker(unit, w)
    class Recorder(UsageRanker):
        def scores(self, before, fresh, old, position, *, usage, mask):
            scores = super().scores(before, fresh, old, position, usage=usage, mask=mask)
            np.testing.assert_allclose(scores, base.scores(before, fresh, old, position))
            calls.append((list(usage.tail), before.copy(), fresh.copy(), old.copy()))
            return scores
    kernel = LLaDASparseKernel(model, 4, mask_slots=3, policy="hard_committed_usage",
                               context_ranker=Recorder(unit, np.r_[w, np.zeros(6)]))
    for q,e in ((4,1),(8,1),(16,2)):
        assert Arm.parse(f"{q}:0:{q-e}:hard_committed_usage",32).edit == e
    with pytest.raises(ValueError):
        Arm.parse("8:0:6:hard_committed_usage",32)
    s = _state(kernel, gen_length=62)
    other = _state(kernel, gen_length=62)
    assert s.usage_history is not other.usage_history
    kernel.step(s)
    s.x[0,34:40] = 1
    committed = []
    with_prefix = 0
    while s.phase != "done":
        block = s.block
        before = s.x[0,32*block:32*block+32].tolist()
        count = len(calls)
        action = kernel.step(s)
        if count < len(calls):
            hist, canvas, fresh, old = calls[-1]
            assert hist == committed and canvas == before
            assert not fresh.intersection(old)
            e = s.events[-1]
            assert set(i+32*block for i in fresh) <= set(e["absolute_positions"])
            assert e["layer_physical_q"] == [len(e["absolute_positions"])]*2
            with_prefix += bool(hist)
        if action == "commit" and s.phase != "done":
            committed.extend(before[max(0,s.prompt_length-block*32):])
            assert list(s.usage_history.tail) == committed
            assert not other.usage_history.tail
    assert len(committed) == 30 and with_prefix > 0


def test_live_weights_match_only_fitted_schema(tmp_path):
    from bdlm_serving.evaluation.frozen import sha
    from bdlm_serving.methods.t2t.embedding_context import CALIBRATION
    from bdlm_serving.methods.t2t.committed_usage import read_usage_weights
    p = tmp_path / "weights.json"
    good = dict(schema="llada-committed-usage-v1", calibration_sha256=sha(CALIBRATION), weights=[0]*34)
    p.write_text(json.dumps(good))
    assert len(read_usage_weights(p)) == 34
    for bad in (dict(weights=[0]*28),dict(weights=[np.nan]*34),dict(calibration_sha256="wrong")):
        p.write_text(json.dumps({**good, **bad}))
        with pytest.raises(ValueError):
            read_usage_weights(p)
