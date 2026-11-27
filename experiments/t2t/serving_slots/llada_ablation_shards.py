"""Resume a registered n164 LLaDA block ablation on two protected GPUs."""
import fcntl
import gzip
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.t2t.serving_slots import block_ablation as base
from experiments.t2t.serving_slots import block_ablation_gpu as protected

RUN = protected.SITE / "prefix90/block-ablation-llada21-v1"
MANIFEST = RUN / "split-v1.json"
CARDS = ("0", "4")


def key(i, q, arm):
    return f"{i:02}-q{q}-{arm}.json.gz"


def all_names():
    return [key(i, q, a) for i in range(90) for q in base.TIERS["llada21"] for a in base.ARMS]


def make_manifest():
    done = {p.name for p in RUN.glob("*.json.gz")}
    expected = set(all_names())
    if not done < expected or list(RUN.glob("*.partial")):
        raise RuntimeError("Unexpected or incomplete files; inspect before splitting")
    # Alternate within each (tier, arm) first, keeping counts balanced globally.
    todo = [key(i, q, a) for q in base.TIERS["llada21"] for a in base.ARMS
            for i in range(90) if key(i, q, a) not in done]
    shards = {card: sorted(todo[j::2]) for j, card in enumerate(CARDS)}
    base.put(MANIFEST, dict(registration=base.sha(RUN/"run.json"), already_done=sorted(done),
                            tasks=shards, launcher=base.sha(Path(__file__)),
                            gpu_uuid=protected.UUID))
    print("checkpoint", len(done), "assigned", {c: len(v) for c, v in shards.items()}, flush=True)


def seal_if_complete():
    names = all_names()
    with (RUN/"seal.lock").open("a") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if (RUN/"complete.json").exists() or not all((RUN/n).is_file() for n in names):
            return
        split = json.loads(MANIFEST.read_bytes())
        subsets = {n: c for c, files in split["tasks"].items() for n in files}
        assert set(subsets).isdisjoint(split["already_done"])
        assert set(subsets).union(split["already_done"]) == set(names)
        sums = {}
        for i in range(90):
            sig = None
            for q in base.TIERS["llada21"]:
                for a in base.ARMS:
                    n = key(i, q, a)
                    with gzip.open(RUN/n) as f:
                        r = json.load(f)
                    assert (r["index"], r["q"], r["arm"]) == (i, q, a)
                    assert r["registration"] == split["registration"] and r["TPF"] == r["N"]/r["F"]
                    if n in subsets:
                        assert r["shard_uuid"] == protected.UUID[subsets[n]]
                    if q == 64:
                        s = base.top_signature(r)
                        assert sig is None or sig == s
                        sig = s
                    sums[n] = {k:r[k] for k in ("N","F","Q","correct","TPF","efficiency")}
        base.put(RUN/"complete.json", dict(registration=split["registration"], rows=len(sums),
                                           split=base.sha(MANIFEST), cases=sums))


def run_shard(model, out, samples):
    import torch
    import transformers
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen90 import load_tasks
    split = json.loads(MANIFEST.read_bytes())
    cfg = json.loads((RUN/"run.json").read_bytes())
    uuid = os.environ["CUDA_VISIBLE_DEVICES"]
    card, = [c for c in CARDS if protected.UUID[c] == uuid]
    assert out == RUN.parent/f"{RUN.name}-shard{card}" and model == "llada21"
    assert split["registration"] == base.sha(RUN/"run.json")
    assert split["launcher"] == base.sha(Path(__file__))
    assert cfg["runner"] == base.sha(Path(base.__file__))
    assert base.sha(samples) == cfg["samples"] and base.sha(Path(scoring.__file__)) == cfg["scorer"]
    assert torch.cuda.device_count() == 1 and torch.cuda.is_available()
    assert transformers.__version__ == cfg["transformers"] and str(torch.__version__) == cfg["torch"]
    work = split["tasks"][card]
    out.mkdir(exist_ok=False)
    base.put(out/"run.json", dict(card=card, uuid=uuid, split=base.sha(MANIFEST), tasks=work))
    tasks = load_tasks(samples)
    case = base.build(model, samples)
    with torch.inference_mode():
        for count, name in enumerate(work, 1):
            assert all(base.sha(ROOT/k) == v for k,v in cfg["sources"].items())
            assert not (RUN/name).exists()
            idx, q, a = name.split(".")[0].split("-")
            i, q = int(idx), int(q[1:])
            t = time.monotonic()
            r = case(tasks[i], q, a)
            torch.cuda.synchronize()
            assert all(r[k] == tasks[i][k] for k in ("index","dataset","task_id","seed"))
            assert r["q"] == q and r["F"] > 0 and r["Q"] >= r["F"] and r["TPF"] == r["N"]/r["F"]
            assert r["efficiency"] == r["N"]/r["Q"] and type(r["correct"]) is bool
            assert r["correct"] == bool(r["score"]["correct"])
            r.update(arm=a, registration=split["registration"], elapsed=time.monotonic()-t,
                     shard_uuid=uuid, shard_manifest=base.sha(MANIFEST))
            base.put(RUN/name,r)
            print(f"gpu={card} {count}/{len(work)} {name} F={r['F']} Q={r['Q']}", flush=True)
    base.put(out/"finished.json", dict(count=len(work), uuid=uuid))
    seal_if_complete()


if __name__ == "__main__":
    if sys.argv[1:] == ["plan"]:
        make_manifest()
    else:
        protected.SELF = Path(__file__).resolve()
        base.run = run_shard
        protected.main()
