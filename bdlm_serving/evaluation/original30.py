"""Pin the original 30 records and require exact reproduction before extending a run."""
import json
from .frozen import sha


def pin_rows(folder, policy):
    paths = [folder / f"{i:02d}-{policy}.json" for i in range(30)]
    return [(path, sha(path)) for path in paths]


def check_reproduction(reference, row):
    path, digest = reference
    if sha(path) != digest:
        raise ValueError(f"{path.name}: historical reference changed during generation")
    old = json.loads(path.read_text())
    new = json.loads(json.dumps(row))
    if old != new:
        fields = sorted(k for k in old.keys() | new.keys() if k not in old or k not in new
                        or old[k] != new[k])
        raise ValueError(f"{path.name}: original30 reproduction failed: {fields}")


def save_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(path)
