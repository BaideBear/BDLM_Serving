"""Fit one positional hazard on 24 deterministic requests outside frozen30."""
import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path

from bdlm_serving.evaluation.frozen import MANIFEST, load_tasks, sha
from tools.edit_opportunity import fit, load
from tools.trace_native import llada


def select(raw):
    import pyarrow.parquet as pq
    manifest = json.loads(MANIFEST.read_text())
    for filename, expected in manifest["source_sha256"].items():
        if sha(raw/Path(filename).name) != expected:
            raise ValueError("Wrong calibration source")
    with gzip.open(raw/"HumanEval.jsonl.gz","rt") as f:
        human = [json.loads(line) for line in f]
    datasets = {
        "gsm8k": pq.read_table(raw/"gsm8k-test.parquet").to_pylist(),
        "math500": [json.loads(s) for s in (raw/"math500-test.jsonl").read_text().splitlines()],
        "humaneval": human}
    used = {s["prompt_sha256"] for s in manifest["samples"]}
    train = []
    for kind, data in datasets.items():
        avoid = {s["source_index"] for s in manifest["samples"] if s["dataset"] == kind}
        order = sorted(set(range(len(data)))-avoid,
            key=lambda j:sha256(f"calibration-v1:{kind}:{j}".encode()).digest())
        count = 0
        for j in order:
            row = data[j]
            prompt = ("Complete the following Python function. Return the full function implementation "
                      "in one Python code block.\n\n" + row["prompt"] if kind == "humaneval" else
                      row["question" if kind == "gsm8k" else "problem"] + "\nPut your final answer in \\boxed{}.")
            ident = sha256(prompt.encode()).hexdigest()
            if ident in used:
                continue
            used.add(ident)
            idx = len(train)
            train.append(dict(index=idx,source_index=j,dataset=kind,task_id=f"calibration/{kind}/{j}",
                              seed=18100+idx,prompt=prompt,prompt_sha256=ident))
            count += 1
            if count == 8:
                break
        assert count == 8
    return train


def main(args, prepare=None):
    load_tasks(args.samples)
    train = select(args.raw)
    llada(args, prepare, calibration_tasks=train)
    paths = [load(args.output / f"{t['index']:02d}.trace.jsonl.gz", t) for t in train]
    positive = sum(len(labels.intersection(old)) for path in paths for stage,_,_,old,*rest,labels in path
                   if stage == "selected")
    if not positive:
        raise ValueError("No old edit labels on calibration prompts")
    report = dict(schema="llada-positional-hazard-v1", weights=fit(paths).tolist(),
        samples=[{k:v for k,v in t.items() if k != "prompt"} for t in train],
        calibration_old_positives=positive)
    (args.output/"weights.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("samples","raw","checkpoint","output"):
        p.add_argument("--"+k,required=True,type=Path)
    main(p.parse_args())
