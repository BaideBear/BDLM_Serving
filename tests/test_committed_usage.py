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
