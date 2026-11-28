"""Extract just the fixed six from hash-pinned n164 originals for the A3 worker."""
from hashlib import sha256
import json
from pathlib import Path
import sys

from seat import check

CASES = (7, 13, 25, 33, 55, 71)


def prepare(root):
    raw = (root/"raw.log").read_bytes()
    ref = (root/"reference.json").read_bytes()
    check(sha256(raw).hexdigest() == "a7d85ee3e36cdc59e100ca76794da35752cab1cebc059f693c64fb4d55542ae4"
          and sha256(ref).hexdigest() == "c7b450237fdf3dd7abc905200e704ea89d9dbee4bd6dacd8806dd03a4a67f93a",
          "Wrong sealed A3 reference")
    cases = {}
    for line in raw.splitlines():
        if line.startswith(b"RECEIPT "):
            p = json.loads(line[8:])
            if p["kind"] == "case" and p["index"] in CASES:
                cases[p["index"]] = p
    refs = json.loads(ref)["rows"]
    return [dict(case=cases[i], prompt=refs[i]) for i in CASES]


if __name__ == "__main__":
    print(json.dumps(prepare(Path(sys.argv[1])), separators=(",", ":")))
