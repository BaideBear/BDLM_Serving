"""Emit the same paired six-board split for every fixed model/offer/arm."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


BOARDS = ("n164:H0", "n164:H4", "A800:0", "A800:1", "A800:2", "A800:3")
OFFERS = {"llada21": (4, 8, 16, 32, 64), "gemma": (32, 64, 128, 256, 512)}
MANIFEST = Path(__file__).resolve().parents[1]/"benchmarks/frozen90/manifest.json"


def split(samples):
    rows = sorted(samples, key=lambda r: r["index"])
    if [r["index"] for r in rows] != list(range(90)) or len({r["task_id"] for r in rows}) != 90:
        raise ValueError("Missing or duplicated frozen identity")
    ds = sorted({r["dataset"] for r in rows})
    if len(ds) != 3:
        raise ValueError("Expected three frozen datasets")
    result = {board: [] for board in BOARDS}
    for d, dataset in enumerate(ds):
        subset = [r for r in rows if r["dataset"] == dataset]
        if len(subset) != 30 or sum(r["index"] < 30 for r in subset) != 10:
            raise ValueError("Expected 10 original and 20 new rows per dataset")
        for i, row in enumerate(subset):
            result[BOARDS[(i+2*d) % 6]].append(row)
    for shard in result.values():
        shard.sort(key=lambda r: r["index"])
        assert len(shard) == 15 and sum(r["index"] < 30 for r in shard) == 5
        assert set(Counter(r["dataset"] for r in shard).values()) == {5}
    return result


def verify_complete(samples, receipts, *, model, offer, version):
    """Verify the identity ledger of one frozen point, not the validity of its numerical bills."""
    expected = {r["index"]: (board, r) for board, shard in split(samples).items() for r in shard}
    seen = set()
    for rec in receipts:
        index = rec["index"]
        if index not in expected or index in seen:
            raise ValueError(f"Unknown or duplicate frozen index {index}")
        board, row = expected[index]
        if (not version or (rec["board"], rec["model"], rec["offer"], rec["version"]) !=
                (board, model, offer, version) or
                any(rec[k] != row[k] for k in ("task_id", "seed", "prompt_sha256")) or
                rec["status"] not in ("complete", "valid_algorithm_failure")):
            raise ValueError(f"Unusable or mismatched frozen receipt {index}")
        seen.add(index)
    missing = sorted(expected.keys()-seen)
    if missing:
        raise ValueError(f"Missing {len(missing)} frozen receipts: {missing}")
    return len(seen)


def verify_matrix(samples, receipts, *, version):
    """Require all ten cells under one registration (it may contain separate model parameters)."""
    by_cell = defaultdict(list)
    for rec in receipts:
        model, offer = rec["model"], rec["offer"]
        if model not in OFFERS or type(offer) is not int or offer not in OFFERS[model]:
            raise ValueError(f"Unregistered fixed offer: {model}/{offer}")
        by_cell[model, offer].append(rec)
    result = {}
    for model, offers in OFFERS.items():
        for offer in offers:
            try:
                result[f"{model}/Q{offer}"] = verify_complete(
                    samples, by_cell[model, offer], model=model, offer=offer, version=version)
            except ValueError as err:
                raise ValueError(f"{model}/Q{offer}: {err}") from err
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, help="Check all cells in a compact NDJSON ledger")
    parser.add_argument("--version", help="Pinned registration ID for the full matrix")
    args = parser.parse_args()
    if bool(args.receipts) != bool(args.version):
        parser.error("--receipts and --version must be supplied together")
    samples = json.loads(MANIFEST.read_text())["samples"]
    if args.receipts:
        with args.receipts.open() as source:
            result = verify_matrix(samples, (json.loads(line) for line in source), version=args.version)
    else:
        result = split(samples)
    print(json.dumps(result, indent=2))
