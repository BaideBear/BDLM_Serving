import io
import json
import sys

import pytest

from tools import fixed90_partition as p


def setup_run(root):
    (root/"source.sha256").write_bytes(b"test inventory")
    reg = {"uuids": ["gpu-a", "gpu-b", "gpu-c"], "offer": 32}
    (root/"registration.json").write_text(json.dumps(reg))
    (root/"cases").mkdir()
    for rank in range(3):
        (root/f"part-{rank}").mkdir()
    return reg, p.digest(b"test inventory")


def events(rank, reg, source):
    indices = p.partition(90, rank, 3)
    for i in indices:
        raw = f"tick for {i}\n".encode()
        seal = json.dumps(dict(index=i, raw=f"part-{rank}/case-{i:02}.ndjson",
                               raw_sha256=p.digest(raw), uuid=reg["uuids"][rank],
                               offer=reg["offer"], source_sha256=source, status="complete")).encode()
        yield dict(kind="pair", index=i, payload=p.encode(raw), seal=p.encode(seal),
                   seal_sha=p.digest(seal))
    yield dict(kind="finished", indices=indices, rank=rank)


def receive(root, rank, rows, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("".join(json.dumps(e)+"\n" for e in rows)))
    p.receive(root, rank)


def test_three_partitions_close_exactly_once(tmp_path, monkeypatch):
    reg, source = setup_run(tmp_path)
    for rank in (2, 0, 1):
        receive(tmp_path, rank, events(rank, reg, source), monkeypatch)
    p.close(tmp_path)
    blob = json.loads((tmp_path/"complete.json").read_bytes())
    assert blob["completed"] == len(blob["case_sha256"]) == 90
    with pytest.raises(FileExistsError):
        p.close(tmp_path)


@pytest.mark.parametrize("fault", ["overlap", "gap", "digest", "disconnect"])
def test_bad_pipeline_never_closes(tmp_path, monkeypatch, fault):
    reg, source = setup_run(tmp_path)
    data = list(events(1, reg, source))
    if fault == "overlap":
        data.insert(1, data[0])
    elif fault == "gap":
        del data[-2]
    elif fault == "digest":
        data[0]["payload"] = p.encode(b"corrupt")
    else:
        data = data[:-1]
    with pytest.raises(ValueError):
        receive(tmp_path, 1, data, monkeypatch)
    assert not (tmp_path/"part-1"/"complete.json").exists()
    assert not (tmp_path/"complete.json").exists()
