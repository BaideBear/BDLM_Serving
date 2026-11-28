from pathlib import Path
import pytest


def test_work_partition_with_a_revocation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"tools"))
    from ibs_boundary_headroom import bill_work, bound
    # Admit row 0; pay it and revoke it; re-admit it on its second MASK visit.
    rows = [(0,[0,1],[0],[],0,[True,True],[]),
            (0,[0],[],[0],0,[False],[0]),
            (0,[0,1],[0],[],0,[True,True],[])]
    w = bill_work(rows,"llada21",0)
    assert (w["unresolved_first_Q"],w["unresolved_revisit_Q"],w["debt_Q"]) == (2,2,1)
    assert (w["first_view_admits"],w["revisit_admits"]) == (1,1)
    assert bound(12,4,1) == 4
    with pytest.raises(ValueError):
        bound(12,4,4)
