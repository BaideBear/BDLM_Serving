"""Distribute independent, fixed-offer requests; keep the qualified per-case loop."""
import argparse
import ast
import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time
import zlib


CASE_DRIVER = "4ece0e2598be8eb705a438eb696376599d89b00e70e771c3137335996f323feb"
EXECUTOR = "7c584c5426d77fad3eec4aa397c1f58a86b7ea093ccded2574ee8b76fa44c993"
SAMPLES = "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed"
CKPT = "d4b48682fb49b315db221a96e288fb01ea20499f6aa4e4ef705ec9a2b328a918"


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def digest(blob):
    return sha256(blob).hexdigest()


def partition(n, rank, count):
    require(0 <= rank < count <= n, "invalid partition")
    return list(range(rank, n, count))


def bind_case(module, offer):
    """Bind only the per-case nameplate, step argument and non-admission guard."""
    blob = module.read_bytes()
    require(digest(blob) == CASE_DRIVER, "unknown qualified case driver")
    case = next(x for x in ast.parse(blob).body if isinstance(x, ast.FunctionDef) and x.name == "case")
    refs = [x for x in case.body if isinstance(x, ast.Try)]
    anchor, = [x for x in refs[0].body if isinstance(x, ast.If) and ast.unparse(x.test) == "ix == 7"]
    # That post-run comparison is an H4-only Q16 replay, not an A800 reference.
    require("llada-response-admission-h4-gate-v2/complete.json" in ast.unparse(anchor), "anchor changed")
    refs[0].body.remove(anchor)
    sites = [x for x in ast.walk(case) if isinstance(x, ast.Constant) and type(x.value) is int and x.value == 16]
    require(len(sites) == 3, "unexpected Q16 binding sites")
    for x in sites:
        x.value = offer
    return compile(ast.Module([case], []), str(module), "exec")


def send(kind, **data):
    print(json.dumps(dict(kind=kind, **data), separators=(",", ":")), flush=True)


def encode(data):
    return base64.b64encode(zlib.compress(data, 1)).decode("ascii")


def decode(text):
    return zlib.decompress(base64.b64decode(text, validate=True))


def write_new(path, data):
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def run(root, rank):
    import runpy

    reg = json.loads((root/"registration.json").read_bytes())
    indices = partition(90, rank, len(reg["uuids"]))
    q, uuid = reg["offer"], reg["uuids"][rank]
    require(reg["mode"] == "full_block" and reg["cap"] == 2048 and q in (4,8,16,32,64), "bad campaign")
    require(os.environ["CUDA_VISIBLE_DEVICES"] == uuid, "GPU binding")
    sys.path[:0] = [str(root/"src"), str(root/"helpers")]
    import torch
    import forks
    require(torch.cuda.device_count() == 1 and str(torch.cuda.get_device_properties(0).uuid).lower()
            .removeprefix("gpu-") == uuid[4:], "GPU identity")
    require(digest((root/"samples.json").read_bytes()) == SAMPLES and
            digest(Path(forks.candidate.__file__).read_bytes()) == EXECUTOR and
            Path(forks.candidate.__file__).resolve().is_relative_to(root/"src"), "source or tasks drift")
    forks.REG = dict(forks.REG, cutoff=reg["cutoff"])
    forks.CKPT = Path("/gs/models/LLaDA2.1-mini")
    forks.DATA = root
    forks.meter.emit = send
    path = root/"fixed90.py"
    code = bind_case(path, q)
    g = runpy.run_path(str(path), run_name="pinned_driver")
    g["reg"] = dict(reg, uuid=uuid)
    exec(code, g)
    source = digest((root/"source.sha256").read_bytes())
    tasks = forks.load_tasks(root/"samples.json")
    require([t["index"] for t in tasks] == list(range(90)), "wrong frozen panel")
    attempt = root/f"part-{rank}"
    attempt.mkdir()
    (root/"cases").mkdir()
    start_ticks = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
    send("worker", pid=os.getpid(), start_ticks=start_ticks, uuid=uuid, source=source,
         indices=indices, time=time.time())
    model = forks.worker.load(forks.CKPT)
    require(model["fingerprint"] == CKPT, "unexpected weight fingerprint")
    with torch.inference_mode():
        for i in indices:
            raw = attempt/f"case-{i:02}.ndjson"
            try:
                g["case"](model, tasks[i], attempt, source)
            except Exception:
                if raw.exists():
                    send("partial", index=i, payload=encode(raw.read_bytes()))
                raise
            seal = (root/"cases"/f"{i:02}.json").read_bytes()
            send("pair", index=i, seal=encode(seal), payload=encode(raw.read_bytes()), seal_sha=digest(seal))
            raw.unlink()
    send("finished", rank=rank, indices=indices)


def receive(root, rank):
    reg = json.loads((root/"registration.json").read_bytes())
    expected = partition(90, rank, len(reg["uuids"]))
    dest = root/f"part-{rank}"
    source = digest((root/"source.sha256").read_bytes())
    got, finished = [], False
    with (dest/"worker.ndjson").open("x", buffering=1) as ledger:
        for line in sys.stdin:
            e = json.loads(line)
            kind = e["kind"]
            require(not finished, "unexpected event after worker exit")
            if kind in ("pair","partial"):
                i = e["index"]
                require(len(got) < len(expected) and expected[len(got)] == i, "overlapping or misordered indices")
                raw = decode(e.pop("payload"))
                if kind == "partial":
                    write_new(dest/f"case-{i:02}.partial.ndjson", raw)
                else:
                    seal = decode(e.pop("seal"))
                    r = json.loads(seal)
                    require(digest(raw) == r["raw_sha256"] and digest(seal) == e["seal_sha"] and
                            r["index"] == i and r["raw"] == f"part-{rank}/case-{i:02}.ndjson" and
                            r["uuid"] == reg["uuids"][rank] and r["source_sha256"] == source
                            and r["offer"] == reg["offer"] and r["status"] == "complete", "bad case receipt")
                    write_new(dest/f"case-{i:02}.ndjson", raw)
                    write_new(root/"cases"/f"{i:02}.json", seal)
                    got.append(i)
            if kind == "finished":
                require(e["rank"] == rank and got == expected == e["indices"], "missing cases")
                finished = True
            ledger.write(json.dumps(e, separators=(",", ":"))+"\n")
    require(finished, "worker disconnected before all assigned cases were sealed")
    write_new(dest/"complete.json", json.dumps({"indices":got, "source_sha256":source}).encode())


def close(root):
    reg = json.loads((root/"registration.json").read_bytes())
    source = digest((root/"source.sha256").read_bytes())
    keys = {}
    for rank, uuid in enumerate(reg["uuids"]):
        p = root/f"part-{rank}"
        d = json.loads((p/"complete.json").read_bytes())
        require(d == {"indices": partition(90,rank,len(reg["uuids"])), "source_sha256":source}, "bad shard")
        for i in d["indices"]:
            path = root/"cases"/f"{i:02}.json"
            blob = path.read_bytes()
            r = json.loads(blob)
            require(r["uuid"] == uuid and r["raw"] == f"part-{rank}/case-{i:02}.ndjson" and
                    digest((root/r["raw"]).read_bytes()) == r["raw_sha256"], "bad artifact")
            keys[path.name] = digest(blob)
    require(set(keys) == {f"{i:02}.json" for i in range(90)}, "incomplete")
    write_new(root/"complete.json", json.dumps(dict(completed=90,source_sha256=source,
                                                    case_sha256=keys), sort_keys=True).encode())


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("action",choices=("run","receive","close"))
    a.add_argument("root",type=Path)
    a.add_argument("--rank",type=int)
    args = a.parse_args()
    root = args.root.resolve()
    if args.action == "close":
        close(root)
    else:
        {"run":run,"receive":receive}[args.action](root,args.rank)
