"""Manifest-based offload for this project's A800 experiment products. No inference."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat
import sys


SCOPE = {"runtime", "work", "logs", "results", "history-repo", "code"}
DATA = {".csv", ".tsv", ".jsonl", ".gz", ".zip", ".pdf", ".png", ".svg", ".pyc",
        ".log", ".out", ".pid", ".go", ".ready"}


def role(rel):
    p = Path(rel)
    a = p.parts
    if len(a) < 2 or a[0] not in SCOPE or p.is_absolute() or ".." in a or ".git" in a:
        return None
    if a[0] in ("results", "logs"):
        return a[0]
    if any(s in {".pytest_cache", "__pycache__", "triton", "triton-rl"} or
           (s.startswith("triton-g") and s[8:].isdigit()) for s in a[:-1]):
        return "cache"
    # Keep code even when a historic launcher put code inside an output directory.
    if p.suffix.lower() in {".py",".sh",".cu",".cuh",".h",".c",".cpp",".toml",".yaml",".yml",".ini",".md"}:
        return None
    # Keep tracked files in the actual source checkout (not in copied experiment runtimes).
    if a[0] == "history-repo":
        return None
    # Benchmark inputs, model/config manifests and tokenizer assets are reproducibility inputs.
    if any(s in {"benchmarks","configs","checkpoints","third_party"} for s in a):
        return None
    if p.suffix.lower() in DATA:
        return "product"
    if p.suffix == ".json" and (
            any(s in {"results","logs","reports","compact","rows","archive","controls","figures",
                      "pairs","llada-refs"} or s.startswith(("first-step-","gate-g","dispatch-",
                      "packed-control-")) for s in a[:-1]) or
            any(s in p.name.lower() for s in ("report","summary","score","pilot","audit","result"))):
        return "product"
    return None


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""):
            h.update(b)
    return h.hexdigest()


def regular_unlinked(root, name):
    path = root / name
    return path if path.resolve()==path and path.is_file() else None


def inventory(root):
    for scope in sorted(SCOPE):
        for d, _, fs in os.walk(root / scope,followlinks=False):
            for f in sorted(fs):
                path=Path(d)/f
                name=str(path.relative_to(root))
                what=role(name)
                if what and regular_unlinked(root,name):
                    print(json.dumps([name,path.stat().st_size,digest(path),what],separators=(",",":")))


def opened():
    paths=set()
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            for fd in (proc/"fd").iterdir():
                try:
                    paths.add(os.readlink(fd))
                except OSError:
                    pass
        except OSError:
            pass
    return paths


def prune(root, keep, *, apply):
    if root != Path("/gs") or not (root/"models-persistent").is_dir():
        raise ValueError("pruning is only available for the project source on A800")
    held = opened()
    counts, sizes = Counter(),Counter()
    for line in sys.stdin:
        name,size,sha,group=json.loads(line)
        if not role(name) or role(name)!=group or not isinstance(sha,str) or len(sha)!=64:
            raise ValueError("invalid destination inventory")
        p=regular_unlinked(root,name)
        if not p:
            status="not_regular_or_absent"
        elif str(p) in held or any(name==k or name.startswith(k.rstrip("/")+"/") for k in keep):
            status="protected"
        else:
            before=p.stat()
            if (not stat.S_ISREG(before.st_mode) or before.st_size!=size or digest(p)!=sha or
                    (before.st_ino,before.st_mtime_ns)!=(p.stat().st_ino,p.stat().st_mtime_ns)):
                status="changed"
            else:
                status=group
                if apply:
                    p.unlink()
        counts[status]+=1
        sizes[status]+=size
    print(json.dumps(dict(applied=apply,counts=counts,bytes=sizes),sort_keys=True))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["inventory","prune"])
    ap.add_argument("root",type=Path)
    ap.add_argument("--keep",action="append",default=[])
    ap.add_argument("--apply",action="store_true")
    args=ap.parse_args()
    if args.action=="inventory":
        inventory(args.root.resolve())
    else:
        prune(args.root.resolve(),args.keep,apply=args.apply)
