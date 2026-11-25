"""Rebuild the original UTF-8 JSON from locally supplied dataset snapshots."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmarks/frozen30/manifest.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tasks(path):
    if sha(path) != json.loads(MANIFEST.read_text())["samples_sha256"]:
        raise ValueError("samples.json does not match the archived 30-request manifest")
    return json.loads(path.read_text())["samples"]


def restore(raw_dir, output):
    import pyarrow.parquet as pq
    manifest = json.loads(MANIFEST.read_text())
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
    rows = []
    for entry in manifest["samples"]:
        kind = entry["dataset"]
        row = datasets[kind][entry["source_index"]]
        if kind == "humaneval":
            prompt = ("Complete the following Python function. Return the full function "
                      "implementation in one Python code block.\n\n"+row["prompt"])
            ref = row["canonical_solution"]
        else:
            prompt = row["question" if kind == "gsm8k" else "problem"] + "\nPut your final answer in \\boxed{}."
            ref = row["answer"].split("####")[-1].strip() if kind == "gsm8k" else row["answer"]
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
