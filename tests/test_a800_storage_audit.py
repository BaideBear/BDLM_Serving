import json
from pathlib import Path
import pytest
from tools.a800_storage_audit import inventory, prune, regular_unlinked, role


@pytest.mark.parametrize("name",[
    "../results/test.log", "runtime/../logs/file.log", "/gs/results/foo.json",
    "models-persistent/m/weights.bin","env/deps/configs.json",
    "runtime/repo/benchmarks/frozen90/manifest.json","history-repo/.git/config",
    "runtime/repo/figures/plot.py","work/copy/helpers.sh"])
def test_scope_and_dependencies_kept(name):
    assert role(name) is None


def test_inventory_no_links_and_no_unmatched_source_cleanup(tmp_path,capsys):
    root=(tmp_path/"source").resolve()
    directory=root/"results"
    directory.mkdir(parents=True)
    (directory/"ok.json").write_bytes(b"foo")
    (directory/"elsewhere.json").symlink_to(tmp_path)
    (directory/"link.json").symlink_to(directory/"ok.json")
    inventory(root)
    (r,)=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert r[0]=="results/ok.json" and r[1]==3 and r[-1]=="results"
    assert regular_unlinked(root,"results/link.json") is None
    assert role("runtime/block-history-1127/triton-g4/sub/kernel.ptx")=="cache"
    with pytest.raises(ValueError,match="source"):
        prune(root,[],apply=True)
    assert (directory/"ok.json").is_file()
