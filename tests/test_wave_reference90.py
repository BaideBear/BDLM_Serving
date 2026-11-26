import copy
import json
from types import SimpleNamespace

import pytest

from bdlm_serving.runtime.diffusiongemma_wave import reference90
from tools.prefix_cohorts import check_gemma_wave


def test_reference_settings_and_all_original_record_pins(tmp_path):
    cfg = dict(cap=2048, policies=["flow_w1", "flow_w2"], flowblock_windows=[1, 2], include_b3=False)
    old = dict(cfg, policies=cfg["policies"]+["b3"], include_b3=True)
    (tmp_path/"run.json").write_text(json.dumps(old))
    for i in range(30):
        for p in cfg["policies"]:
            (tmp_path/f"{i:02d}-{p}.json").write_text("{}")
    args = SimpleNamespace(reference30_root=tmp_path, native_reference=tmp_path)
    refs = reference90(args, cfg)
    assert all(len(v)==30 for v in refs.values()) and len(refs)==2
    with pytest.raises(ValueError):
        reference90(args, dict(cfg, cap=512))
    with pytest.raises(FileNotFoundError):
        (tmp_path/"07-flow_w2.json").unlink()
        reference90(args, cfg)


def test_idle_slots_and_reencoding_are_both_charged():
    row = dict(policy="flow_w2", events=[["encoder",7],["decoder",512],["encoder",256],["decoder",512]],
               trace=[dict(width=2, active=a, q=512) for a in (1,2)],
               max_q=512, generation_forwards=3, generation_query_rows=1280)
    check_gemma_wave(row, 256)
    for key, bad in (("max_q",256), ("generation_query_rows",1024), ("generation_forwards",2)):
        with pytest.raises(ValueError):
            check_gemma_wave(dict(row, **{key:bad}), 256)
    corrupt = copy.deepcopy(row)
    corrupt["events"][1][1] = corrupt["trace"][0]["q"] = 256
    with pytest.raises(ValueError):
        check_gemma_wave(corrupt, 256)
