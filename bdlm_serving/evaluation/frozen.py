"""Rebuild the original UTF-8 JSON from locally supplied dataset snapshots."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmarks/frozen30/manifest.json"

# Predeclared additive offset for a repeat run over the identical task set.
# Applying the same offset to every request changes only the seed, never the
# prompt, reference, identifier, ordering, or any other task field.
REPEAT_SEED_OFFSET = 1_000_003


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tasks(path, *, seed_offset=0):
    """Load the frozen task list with an optional same-task seed offset.

    ``seed_offset`` is added to every request's ``seed`` after the archived
    samples snapshot has been verified.  The task set is otherwise unchanged:
    prompts, references, source rows, identifiers, and ordering are preserved.
    The default of zero reproduces the archived seeds exactly.
    """
    if isinstance(seed_offset, bool) or not isinstance(seed_offset, int):
        raise TypeError("seed_offset must be an int")
    if seed_offset < 0:
        raise ValueError("seed_offset must be non-negative")
    if sha(path) != json.loads(MANIFEST.read_text())["samples_sha256"]:
        raise ValueError("samples.json does not match the archived 30-request manifest")
    tasks = json.loads(path.read_text())["samples"]
    if seed_offset == 0:
        return tasks
    return [{**task, "seed": task["seed"] + seed_offset} for task in tasks]


def source_rows(raw_dir, manifest):
    import pyarrow.parquet as pq
    for logical, expected in manifest["source_sha256"].items():
        path = raw_dir / Path(logical).name
        if sha(path) != expected:
            raise ValueError(f"Wrong dataset snapshot: {path.name}")
    datasets = {
        "gsm8k": pq.read_table(raw_dir / "gsm8k-test.parquet").to_pylist(),
        "math500": [json.loads(line) for line in (raw_dir / "math500-test.jsonl").read_text().splitlines()],
    }
    with gzip.open(raw_dir / "HumanEval.jsonl.gz", "rt", encoding="utf-8") as stream:
        datasets["humaneval"] = [json.loads(line) for line in stream]
    return datasets


def prompt_and_reference(kind, row):
    if kind == "humaneval":
        return ("Complete the following Python function. Return the full function "
                "implementation in one Python code block.\n\n"+row["prompt"]), row["canonical_solution"]
    prompt = row["question" if kind == "gsm8k" else "problem"] + "\nPut your final answer in \\boxed{}."
    return prompt, row["answer"].split("####")[-1].strip() if kind == "gsm8k" else row["answer"]


def restore(raw_dir, output):
    manifest = json.loads(MANIFEST.read_text())
    datasets = source_rows(raw_dir, manifest)
    rows = []
    for entry in manifest["samples"]:
        kind = entry["dataset"]
        row = datasets[kind][entry["source_index"]]
        prompt, ref = prompt_and_reference(kind, row)
        if hashlib.sha256(prompt.encode()).hexdigest() != entry["prompt_sha256"]:
            raise ValueError(f"Prompt mismatch: {entry['task_id']}")
        rows.append({**{k: entry[k] for k in ("index", "source_index", "dataset", "task_id")},
                     "prompt": prompt, "reference": ref, "source": row, "seed": entry["seed"]})
    result = {"selection_seed": manifest["selection_seed"], "samples": rows,
              "source_sha256": manifest["source_sha256"], "scorer_version": manifest["scorer_version"]}
    encoded = json.dumps(result, ensure_ascii=False, indent=2).encode()
    if hashlib.sha256(encoded).hexdigest() != manifest["samples_sha256"]:
        raise ValueError("Reconstructed dataset does not match the original bytes")
    output.parent.mkdir(exist_ok=True, parents=True)
    with output.open("xb") as stream:
        stream.write(encoded)
