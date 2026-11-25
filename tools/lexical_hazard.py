"""Calibration24-only lexical ablation on teacher-forced LLaDA native paths."""
import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import json
from pathlib import Path
import string
import unicodedata

import numpy as np
from tokenizers import Tokenizer

from bdlm_serving.evaluation.frozen import MANIFEST, ROOT, sha
from tools.edit_opportunity import evaluate, fit, load, shortlist


CLASSES = ("whitespace", "digit", "alpha", "punct_operator", "mixed", "other")
BUDGETS = ((4, 1), (8, 1), (16, 2))
TOP = (1, 4, 8)


def lexical_class(text):
    if not text or "\ufffd" in text:
        return "other"
    stripped = text.strip()
    if not stripped:
        return "whitespace"
    kinds = {"digit" if c.isdigit() else "alpha" if c.isalpha()
             else "punct_operator" if c in string.punctuation or
             unicodedata.category(c).startswith("P") or unicodedata.category(c) == "Sm"
             else "whitespace" if c.isspace() else "other" for c in stripped}
    return next(iter(kinds)) if len(kinds) == 1 else "mixed"


def token_encoder(path):
    checkpoint = json.loads((ROOT/"benchmarks/checkpoints/llada21.json").read_text())
    expected = next(r["sha256"] for r in checkpoint if r["name"] == "tokenizer.json")
    if sha(path) != expected:
        raise ValueError("Tokenizer does not match archived LLaDA checkpoint")
    tokenizer = Tokenizer.from_file(str(path))
    special = {t["id"] for t in json.loads(path.read_text())["added_tokens"] if t["special"]}

    @lru_cache(maxsize=None)
    def encode(token_id):
        kind = "other" if token_id in special else lexical_class(
            tokenizer.decode([token_id], skip_special_tokens=False))
        return tuple(float(kind == c) for c in CLASSES)

    return encode


def check_split(train, test):
    for tasks, n in ((train, 8), (test, 10)):
        if (Counter(t["dataset"] for t in tasks) != dict.fromkeys(("gsm8k", "math500", "humaneval"), n)
                or [t["index"] for t in tasks] != list(range(3*n))):
            raise ValueError("Unexpected request count, order or composition")
    for fields in (("dataset", "source_index"), ("prompt_sha256",), ("task_id",)):
        keys = [tuple(t[k] for k in fields) for t in train + test]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicated request identity: {fields}")


def measurements(path, weights):
    curve = shortlist(path, weights)
    for k in TOP:
        c = curve[k]
        yield f"top-{k}", Counter(seats=c["seats"], targets=c["positives"], hits=c["hits"])
    for q, cap in BUDGETS:
        c = evaluate(path, weights, q, cap, "hazard")
        yield f"Q{q}/{cap}", Counter(seats=c["sparse_seats"], targets=c["selected_old"],
                                      hits=c["sparse_reserved_hits"])


def write_rows(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main(cal, native, tokenizer, out):
    archived = ROOT/"experiments/t2t/calibrated/weights.json"
    if sha(cal/"weights.json") != sha(archived):
        raise ValueError("Unexpected calibration manifest")
    if json.loads((native/"verified.json").read_text()) != {"model":"llada21", "matched_requests":30}:
        raise ValueError("Original verified native30 required")
    artifact = json.loads(archived.read_text())
    train, test = artifact["samples"], json.loads(MANIFEST.read_text())["samples"]
    check_split(train, test)
    arms = (("position", None), ("position_lexical", token_encoder(tokenizer)))
    weights = {}
    for name, encode in arms:
        paths = [load(cal/f"{t['index']:02d}.trace.jsonl.gz", t, old_token_features=encode) for t in train]
        weights[name] = fit(paths)
    np.testing.assert_allclose(weights["position"], artifact["weights"], rtol=1e-5, atol=1e-6)

    totals, rows = defaultdict(Counter), []
    for t in test:
        counts = {}
        for name, encode in arms:
            path = load(native/f"{t['index']:02d}.trace.jsonl.gz", t, old_token_features=encode)
            for metric, c in measurements(path, weights[name]):
                counts[name, metric] = c
                totals[name, metric].update(c)
                rows.append(dict(index=t["index"], task_id=t["task_id"], model=name, metric=metric, **c))
        for metric in (key for arm, key in counts if arm == "position"):
            a, b = counts["position", metric], counts["position_lexical", metric]
            assert (a["targets"], a["seats"]) == (b["targets"], b["seats"])
    summary = [dict(model=name, metric=metric, **c, precision=c["hits"]/c["seats"],
                    recall=c["hits"]/c["targets"]) for (name, metric), c in totals.items()]
    if any(r["targets"] != 191 for r in summary):
        raise ValueError("Residual target count differs from native archive")
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out/"requests.csv", rows)
    write_rows(out/"summary.csv", summary)
    (out/"coefficients.json").write_text(json.dumps(
        dict(classes=CLASSES, weights={k:v.tolist() for k,v in weights.items()}), indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for opt in ("cal", "native", "tokenizer", "out"):
        p.add_argument("--"+opt, required=True, type=Path)
    main(**vars(p.parse_args()))
