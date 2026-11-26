"""Rebuild the 90-case prefix study without bundling dataset texts or answers."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from . import frozen

MANIFEST = frozen.ROOT / "benchmarks/frozen90/manifest.json"
DATASETS = ("gsm8k", "math500", "humaneval")
SALT = "block-prefix-90-v1:20261126"


def encoded(x):
    return json.dumps(x, ensure_ascii=False, indent=2).encode()


def digest(x):
    return hashlib.sha256(x).hexdigest()


def register(row):
    return dict(index=row["index"], source_index=row["source_index"], dataset=row["dataset"],
                task_id=row["task_id"], seed=row["seed"], prompt_sha256=digest(row["prompt"].encode()))


def validate(rows):
    if [r["index"] for r in rows] != list(range(90)):
        raise ValueError("Noncanonical 90-case ordering")
    if any(Counter(r["dataset"] for r in cohort) != dict.fromkeys(DATASETS, n)
           for cohort, n in ((rows[:30], 10), (rows[30:], 20))):
        raise ValueError("Unbalanced frozen90 cohorts")
    for key in ("source_index", "task_id", "prompt_sha256"):
        if len({(r["dataset"], r[key]) for r in rows}) != 90:
            raise ValueError(f"Repeated {key}")


def load_tasks(path):
    manifest = json.loads(MANIFEST.read_text())
    if frozen.sha(path) != manifest["samples_sha256"]:
        raise ValueError("Not the registered frozen90 snapshot")
    rows = json.loads(path.read_text())["samples"]
    entries = list(map(register, rows))
    validate(entries)
    original = json.loads(frozen.MANIFEST.read_text())
    if entries != manifest["samples"] or entries[:30] != original["samples"]:
        raise ValueError("Case identity or original 30 changed")
    return rows


def freeze(raw, old_samples, excluded):
    original = json.loads(frozen.MANIFEST.read_text())
    rows = list(frozen.load_tasks(old_samples))
    data = frozen.source_rows(raw, original)
    seen = {digest(r["prompt"].encode()) for r in rows}
    for ds in DATASETS:
        banned = {r["source_index"] for r in rows if r["dataset"] == ds}
        for source in excluded.values():
            banned.update(source["indices"][ds])
        candidates = sorted((i for i in range(160 if ds == "gsm8k" else 0, len(data[ds]))
                             if i not in banned),
                            key=lambda i: digest(f"{SALT}:{ds}:{i}".encode()))
        selected = []
        for i in candidates:
            row = data[ds][i]
            prompt, ref = frozen.prompt_and_reference(ds, row)
            checksum = digest(prompt.encode())
            if checksum in seen:
                continue
            seen.add(checksum)
            selected.append((i, row, prompt, ref))
            if len(selected) == 20:
                break
        for i, row, prompt, ref in sorted(selected):
            task = f"gsm8k/{i}" if ds == "gsm8k" else row["task_id" if ds == "humaneval" else "unique_id"]
            rows.append(dict(index=len(rows), source_index=i, dataset=ds, task_id=task,
                             prompt=prompt, reference=ref, source=row, seed=17000+len(rows)))
    entries = list(map(register, rows))
    validate(entries)
    if entries[:30] != original["samples"]:
        raise ValueError("Original cases changed")
    payload = dict(selection_salt=SALT, samples=rows, source_sha256=original["source_sha256"],
                   scorer_version=original["scorer_version"])
    manifest = dict(selection_salt=SALT, samples_sha256=digest(encoded(payload)),
                    original_samples_sha256=original["samples_sha256"],
                    source_sha256=original["source_sha256"], scorer_version=original["scorer_version"],
                    exclusions=excluded, samples=entries)
    return manifest, payload


def exclusions(calibration, confirmation):
    result = {}
    for key, path, n in (("calibration24", calibration, 8), ("confirmation300", confirmation, 100)):
        rows = json.loads(path.read_text())["samples"]
        indices = {kind: sorted({r["source_index"] for r in rows if r["dataset"] == kind})
                   for kind in DATASETS}
        if len(rows) != n*3 or any(len(v) != n for v in indices.values()):
            raise ValueError(f"Invalid exclusions: {key}")
        result[key] = dict(file_sha256=frozen.sha(path), indices=indices)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "original", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--calibration", type=Path)
    p.add_argument("--confirmation", type=Path)
    p.add_argument("--register-manifest", type=Path)
    args = p.parse_args()
    if args.register_manifest:
        if not args.calibration or not args.confirmation:
            p.error("initial registration requires both exclusion snapshots")
        blocked = exclusions(args.calibration, args.confirmation)
    else:
        blocked = json.loads(MANIFEST.read_text())["exclusions"]
    manifest, payload = freeze(args.raw, args.original, blocked)
    if args.register_manifest:
        args.register_manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.register_manifest.open("xb") as stream:
            stream.write(encoded(manifest))
    elif manifest != json.loads(MANIFEST.read_text()):
        raise ValueError("Rebuilt manifest differs from frozen90")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as stream:
        stream.write(encoded(payload))


if __name__ == "__main__":
    main()
