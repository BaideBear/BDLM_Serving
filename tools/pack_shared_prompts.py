"""Stream only native prompt IDs and identity metadata to a temporary scorer input directory."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
import tarfile

from bdlm_serving.evaluation.frozen90 import MANIFEST


def pack(root, dest):
    cases = json.loads(MANIFEST.read_bytes())["samples"]
    with tarfile.open(fileobj=dest, mode="w|gz") as tar:
        for task in cases:
            name = f"{task['index']:02d}-native32.json"
            original = json.loads((root/name).read_bytes())
            keys = ("index", "task_id", "dataset", "seed")
            if (any(original[k] != task[k] for k in keys)
                    or original["generation_cap"] != 2048 or not original["prompt_ids"]):
                raise ValueError(f"Unregistered native case: {name}")
            payload = {k: original[k] for k in (*keys, "prompt_ids", "generation_cap")}
            data = json.dumps(payload, separators=(",", ":")).encode()
            info = tarfile.TarInfo(name)
            info.mode, info.size = 0o600, len(data)
            tar.addfile(info, BytesIO(data))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference_root", type=Path)
    pack(p.parse_args().reference_root, sys.stdout.buffer)
