"""Check shard completion and independently decode/rescore sealed output without a model."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


def require(ok, why):
    if not ok:
        raise ValueError(why)


def load(path):
    return json.loads(path.read_bytes())


def reconcile(root, ckpt, missing_exit):
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.evaluation.scoring import score
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    tasks = load_tasks(root / "samples.json")
    reg = load(root / "registration.json")
    source = sha256((root / "source.sha256").read_bytes()).hexdigest()
    require(len(tasks) == 90 and [t["index"] for t in tasks] == list(range(90)), "panel")
    summary = {}
    for rank, uuid in enumerate(reg["uuids"]):
        part = root / f"part-{rank}"
        idx = list(range(rank, 90, len(reg["uuids"])))
        ledger = [json.loads(line) for line in (part/"worker.ndjson").read_bytes().splitlines()]
        worker, = [e for e in ledger if e["kind"] == "worker"]
        exit_path = part/"pipeline.exit"
        statuses = exit_path.read_text().split() if exit_path.exists() else None
        require(statuses == ["0", "0", "0"] or (rank == missing_exit and statuses is None),
                "unexpected exit; never infer a missing code")
        require(worker["uuid"] == uuid and worker["indices"] == idx and worker["source"] == source
                and ledger[-1] == {"kind":"finished","rank":rank,"indices":idx}
                and load(part/"complete.json") == {"indices":idx,"source_sha256":source},
                "worker/receiver completion")
        for kind in ("case_start", "sealed", "pair"):
            require([e["index"] for e in ledger if e["kind"] == kind] == idx, kind)
        require(not any(e["kind"] in ("partial","case_error") for e in ledger), "partial run")
        by_id = {e["index"]: e for e in ledger if e["kind"] == "pair"}
        counts = Counter()
        for i in idx:
            seal = (root/"cases"/f"{i:02}.json").read_bytes()
            r = json.loads(seal)
            require(sha256(seal).hexdigest() == by_id[i]["seal_sha"]
                    and r["raw"] == f"part-{rank}/case-{i:02}.ndjson" and r["uuid"] == uuid
                    and r["status"] == "complete" and r["offer"] == reg["offer"]
                    and r["source_sha256"] == source
                    and all(r[k] == tasks[i][k] for k in ("index", "task_id", "seed")), "case seal")
            raw = (root/r["raw"]).read_bytes()
            require(sha256(raw).hexdigest() == r["raw_sha256"], "tape hash")
            positions = {}
            for line in raw.splitlines():
                tick = json.loads(line)
                positions.update((int(k), v) for k, v in tick["written"].items())
            origin = r["prefill"]["decoder_q"]
            ids = [positions[p] for p in range(origin, origin+r["output_ids"])]
            require(sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest() == r["ids_sha"]
                    and sum(t not in tokenizer.all_special_ids for t in ids) == r["N"]
                    and bool(score(tokenizer.decode(ids,skip_special_tokens=True),tasks[i])["correct"])
                        == r["correct"], f"independent score or token digest {i}")
            counts["reconstructed_and_rescored"] += 1
            counts["correct"] += r["correct"]
        summary[rank] = dict(counts, pipeline_exit=statuses, producer_finished=True, receiver_finished=True)
    return dict(source_sha256=source, cases=sum(v["reconstructed_and_rescored"] for v in summary.values()),
                shards=summary, missing_pipeline_exit=missing_exit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--missing-exit", type=int, help="allow only an absent post-run wrapper record for this rank")
    a = parser.parse_args()
    print(json.dumps(reconcile(a.root, a.checkpoint, a.missing_exit), indent=2, allow_nan=False))
