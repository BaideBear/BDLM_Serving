import json
from pathlib import Path

import pytest

from tools.gemma_parent_exposure_summary import summarize


MANIFEST = Path(__file__).resolve().parents[1]/"benchmarks/frozen90/manifest.json"


def fixture(root, indices):
    root.mkdir()
    samples = json.loads(MANIFEST.read_bytes())["samples"]
    common = dict(kind="worker", q=512, source="7f160c54fa1fcd005aa92f9420c700f5df4cb47f",
                  profile="dense_stop_joint_verified_eos", uuid="test", torch="test", transformers="5.12.0",
                  indices=indices, runner="test")
    out = [common]
    for i in indices:
        if i < 30:
            out.append(dict(kind="no_successor", index=i, q=512, decoder_F=10))
            continue
        new = list(range(288, 544 if i < 80 else 352))
        out.append(dict(kind="first_projection", q=512, origin=32, index=i, task_id=samples[i]["task_id"],
                        seed=samples[i]["seed"], tick=3, width=len(new), physical_rows=new, layer_hooks=85,
                        successor_rows=len(new), missing_successor_rows=256-len(new)))
    out.append(dict(kind="finished", q=512, indices=indices))
    (root/"events.ndjson").write_text("".join(json.dumps(r)+"\n" for r in out))
    (root/"pipeline.exit").write_text("0 0 0\n")
    return root


def test_measurements_are_disjoint_and_exhaustive(tmp_path):
    paths = [fixture(tmp_path/"p",list(range(3))),fixture(tmp_path/"r",list(range(3,90)))]
    r = summarize(paths,MANIFEST)
    assert r["cohorts"]["frozen90"] == dict(n=90,no_successor=30,first_already_full=50,partial_first=10)
    with pytest.raises(ValueError,match="overlap"):
        summarize([*paths,paths[0]],MANIFEST)
    with pytest.raises(ValueError,match="complete"):
        summarize(paths[:1],MANIFEST)
    (paths[0]/"pipeline.exit").write_text("0 1 0\n")
    with pytest.raises(ValueError,match="pipeline"):
        summarize(paths,MANIFEST)
