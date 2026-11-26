from experiments.t2t.calibrated.embedding_context.report import endpoint_envelope


def test_envelope_keeps_whole_endpoints_and_maximizes_quality_before_tpf():
    arms = {mode:(None,None,rows) for mode,rows in {
        "noedit":[[0,40,2,50],[1,21,3,30],[0,5,2,18]],
        "context_once_rank":[[1,7,7,42],[0,100,2,110],[0,5,2,15]],
        "admit_low":[[0,10,1,10],[1,28,4,26],[0,5,2,15]],
        "admit_coin":[[0,40,2,50],[1,2,1,20],[0,5,2,18]]
    }.items()}
    groups,rows=endpoint_envelope([(0,"x"),(1,"x"),(2,"y")],arms)
    assert [r["path"] for r in rows]==["context_once_rank","admit_low","context_once_rank"]
    g=groups[0]
    assert (g["base_correct"],g["any_correct"],g["all_correct"],g["rescued"],g["correct_varies"])==(1,2,0,1,2)
    assert (g["N"],g["F"],g["physical_Q"])==(40,13,83)
    assert g["macro_tpf"]==(1+7+2.5)/3
