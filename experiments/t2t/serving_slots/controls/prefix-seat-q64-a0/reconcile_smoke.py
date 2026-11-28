"""Replay one retained physical case through the production reconcile interface.

This is a CPU-only interface smoke.  It does not infer, rescore, or create a
prefix-seat result.  It proves that a real raw ``RECEIPT`` record can still be
reduced to N/F/Q and the paid writeback ledger before another A0 launch.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


def find_case(raw_log: Path, index: int):
    matches = []
    with raw_log.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.startswith("RECEIPT "):
                continue
            try:
                record = json.loads(line[8:])
            except Exception:
                continue
            if record.get("kind") == "case" and record.get("index") == index:
                matches.append((line_number, record))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one case receipt for {index}, found {len(matches)}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-log", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--q", type=int, default=64)
    parser.add_argument("--eos", default="1,50,106")
    parser.add_argument("--special", default="0,1,50,100,106")
    parser.add_argument("--tokenizer-json", type=Path,
                        default=Path("/data1/project_data/gx-erqw-20261123/models/"
                                     "diffusiongemma-cd40195/tokenizer.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[5])
    args = parser.parse_args()

    if str(args.source_root) not in sys.path:
        sys.path.insert(0, str(args.source_root))
    production = importlib.import_module("tools.run_gemma_ibs_certified_eos")
    line_number, receipt = find_case(args.raw_log, args.index)
    arm = receipt["arms"]["stock"]
    require_keys = {"ids", "trace", "decoder", "encoder", "events", "bill"}
    if not require_keys <= arm.keys():
        raise RuntimeError(f"Raw stock arm lacks production fields: {sorted(require_keys-arm.keys())}")
    prompt = arm["encoder"][0][0]
    eos = {int(x) for x in args.eos.split(",") if x}
    special = {int(x) for x in args.special.split(",") if x}
    if args.tokenizer_json.is_file():
        tokenizer = json.loads(args.tokenizer_json.read_bytes())
        special |= {entry["id"] for entry in tokenizer.get("added_tokens", [])
                    if entry.get("special") is True and isinstance(entry.get("id"), int)}
    bill, states = production.reconcile(
        arm["trace"], arm["decoder"], arm["encoder"], arm["events"], arm["ids"],
        prompt, eos, arm.get("trigger"), arm.get("payment"), args.q)
    if bill != arm["bill"]:
        raise RuntimeError("Production bill differs from sealed raw bill")
    if len(states) != len(arm["trace"]):
        raise RuntimeError("State ledger length differs from trace")
    if bill["admissions"] != bill["paid_rows"] != arm["bill"]["admissions"]:
        raise RuntimeError("Admissions and paid writeback rows disagree")
    expected_n = sum(token not in special for token in arm["ids"])
    if expected_n != arm["N"]:
        raise RuntimeError(f"N mismatch: recomputed {expected_n}, sealed {arm['N']}")
    output = dict(
        interface="RECEIPT(kind=case).arms.stock -> tools.run_gemma_ibs_certified_eos.reconcile",
        raw_log=args.raw_log.name, raw_line=line_number, index=args.index,
        task_id=receipt.get("task_id"), dataset=receipt.get("dataset"), seed=receipt.get("seed"),
        source_records_sha256=hashlib.sha256(args.raw_log.read_bytes()).hexdigest(),
        raw_line_sha256=hashlib.sha256(line.encode()).hexdigest(),
        N=arm["N"], correct=arm.get("correct"), scoring=arm.get("scoring"),
        bill=bill, states=len(states), writeback_validated=True,
        output_sha256=hashlib.sha256(json.dumps(
            arm["ids"], separators=(",", ":")).encode()).hexdigest())
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
