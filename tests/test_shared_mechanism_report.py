import json

import pytest

from experiments.t2t.serving_slots.shared_fixed_shard import identity, shard_cases
from tools.frozen90_six_shards import MANIFEST
from tools.shared_mechanism_report import ARMS, audit, registration


def make_shard(root):
    base = dict(stage="formal90", cap=2048)
    samples = json.loads(MANIFEST.read_bytes())["samples"]
    path = root / "H0" / "llada21.ndjson"
    path.parent.mkdir()
    rows = []
    for case in shard_cases(samples,"n164:H0","screen18"):
        for q in (4,8):
            for arm in ARMS:
                rows.append(dict(case, model="llada21",board="n164:H0",version=identity(registration(base)),
                                 stage="mechanism18", offer=q, arm=arm, status="complete",N=3,F=2,decoder_F=2,
                                 encoder_F=0,encoder_Q=0,Q=2*q,decoder_Q=2*q,max_decoder_Q=q,correct=True,
                                 roles=dict(dirty_write=q, uninitialized_mask=q, resident_mask=0, clean_resolved=0),
                                 writes=dict(prefix_Q=0,prefix_t2t=0,prefix_revocations=0),
                                 probe=dict(first=None,unexposed_tape="t",exposed_calls=0,reserved_seats=0,
                                            avoided_mask_rows=0),output_ids_sha256="x"))
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    return base,path,rows


def test_single_shard_is_qualification_not_whole_panel(tmp_path):
    base,_,_ = make_shard(tmp_path)
    a = audit([tmp_path],base,"n164:H0")
    assert a["target"]==36 and a["seen"]==18 and a["distinct_cases"]==3
    cell=a["results"]["llada21"][4]
    assert cell["arms"]["all_mask"]["requests"]==3
    assert cell["arms"]["prefix"]["pooled_TE"]==3/8
    contrast=cell["contrasts"]["reserve_idle_minus_all_mask"]
    assert contrast["macro_TPF"]==0 and contrast["same_final_ids_on_complete_pairs"]==3
    assert audit([tmp_path],base)["results"]=={}


@pytest.mark.parametrize("defect", ["duplicate","wrong_seed","floor","pre_treatment", "missing_arm"])
def test_corrupt_or_partial_triplet_not_scored(tmp_path,defect):
    base,path,rows=make_shard(tmp_path)
    if defect=="duplicate":
        rows.append(rows[0])
    elif defect=="wrong_seed":
        rows[0]["seed"]+=1
    elif defect=="floor":
        rows[0]["probe"]["first"]={"tick":2}
    elif defect=="pre_treatment":
        rows[1]["probe"]["unexposed_tape"]="y"
    else:
        rows.pop()
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    if defect=="missing_arm":
        assert audit([tmp_path],base,"n164:H0")["results"]=={}
    else:
        with pytest.raises(ValueError):
            audit([tmp_path],base,"n164:H0")
