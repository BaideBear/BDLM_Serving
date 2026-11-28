"""One real full request through the unified adapter. No control arms, no holders."""
import argparse
import json
import os
from hashlib import sha256
from pathlib import Path

from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.methods.t2t.budget_family import Law
from . import unified_llada
from .cross_race_llada import load_gpu, save
from .llada_fixed_family_eval import CAP, KEYS, evaluate_fixed, require


def sha(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def run(a):
    tasks = load_tasks(a.samples)
    a.out.mkdir(parents=True, exist_ok=False)
    save(a.out / "run.json", dict(protocol="unified-llada-smoke-v1", q=a.q, index=a.ix,
         device=os.environ["CUDA_VISIBLE_DEVICES"], checkpoint=str(a.checkpoint),
         samples=sha(a.samples), scorer=sha(__import__("bdlm_serving.evaluation.scoring",
                                                       fromlist=["x"]).__file__),
         sources={p.name: sha(p) for p in (Path(__file__), Path(unified_llada.__file__))}))
    tok, model = load_gpu(a, lambda: None)
    ref = json.loads((a.reference / f"{a.ix:02d}-native32.json").read_bytes())
    require(ref["index"] == a.ix and ref["generation_cap"] == CAP, "wrong native prompt")
    receipts = []
    r = evaluate_fixed(model, tok, tasks[a.ix], ref["prompt_ids"], a.q, Law(),
                       request_cls=unified_llada.factory(on_step=receipts.append), audit_admission=True)
    require(r["max_decoder_Q"] <= a.q, "physical Q exceeds the offer")
    require(all(e["layer_physical_q"] == [e["q"]] * 20 for e in r["trace"]), "per-layer bill mismatch")
    row = dict(**{k: tasks[a.ix][k] for k in KEYS}, q=a.q, correct=r["correct"], N=r["N"], F=r["F"],
               Q=r["Q"], failure=r["failure"], forwards=len(r["trace"]), planned=len(receipts),
               rows=evaluate_head(r["trace"]), sampled=[e["sampled"] for e in r["trace"]][:6])
    save(a.out / f"{a.ix:02d}.json", row)
    print(json.dumps(dict(index=a.ix, q=a.q, correct=r["correct"], N=r["N"], F=r["F"], Q=r["Q"],
                          forwards=len(r["trace"]), planned=len(receipts)), sort_keys=True))


def evaluate_head(trace, n=6):
    return [dict(q=e["q"], rows=len(e["rows"]), sampled=len(e["sampled"]), admitted=len(e["admitted"]))
            for e in trace[:n]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("checkpoint", "samples", "reference", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    ap.add_argument("--q", type=int, required=True)
    ap.add_argument("--ix", type=int, default=0)
    run(ap.parse_args())
