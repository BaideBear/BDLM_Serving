"""Fixed-offer paired pilot: one measured-MASK acceptance rule at all five Q."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from bdlm_serving.evaluation import scoring
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.methods.t2t.budget_family import Law
from .cross_race_llada import load_gpu, save
from .llada_close_factor import SAMPLES
from .llada_fixed_family_eval import CAP, KEYS, SCORER_SHA, TIERS, evaluate_fixed, require
from .llada_frontier_accept_eval import compact


ANCHORS = (0, 10, 20, 30, 40, 50, 60, 70, 80)
RISK = .35


def select(pending, measured, native):
    """Traverse actual unresolved positions, bridging resolved interior only."""
    extra = []
    spent = 0.
    for pos in pending:
        v = measured.get(pos)
        if v is None or not v[0]:
            break
        if pos not in native:
            spent += 1. - v[1]
            if spent > RISK:
                break
            extra.append(pos)
    return extra


def ribbon(base):
    class Ribbon(base):
        def _step(self, q):
            self._ribbon_extra = ()
            lo, end = max(self.head, self.origin), min(self.head + 32, self.end)
            self._open_masks = [lo+i for i, tok in enumerate(self.x[0, lo:end].tolist())
                                if tok == self.mask_id] if not self.stopping and self.bank.last_end > lo else []
            e = super()._step(q)
            e["ribbon_additions"] = self._ribbon_extra
            return e

        def _additional_admissions(self, idx, old, draft, conf, accepted):
            import torch
            positions = idx.tolist()
            vals = {p: (v != self.mask_id, float(c)) for p, v, c in
                    zip(positions, draft[0].tolist(), conf[0].tolist())}
            native = {p for p,k in zip(positions, accepted[0].tolist()) if k}
            self._ribbon_extra = tuple(select(self._open_masks, vals, native))
            result = torch.zeros_like(accepted)
            if self._ribbon_extra:
                result[0] = torch.isin(idx, idx.new_tensor(self._ribbon_extra))
            return result
    return Ribbon


def arms(q):
    from .llada_agreement import request_class
    from .llada_frontier_accept import frontier_class
    from .llada_hfull_single_factor import request_classes
    from .llada_window_sample import variants
    base = request_classes()[1]
    kw = dict(cross=("head", False))
    if q < 32:
        window, paid = variants(base)
        base = window if q < 16 else paid
        control = base
    elif q == 32:
        kw["close_policy"] = "edit_once"
        control = frontier_class(base)
    else:
        kw = dict(cross=("quarter", False), coverage="head", warm_head_sample=True)
        control = request_class(base)
    return dict(live=control, ribbon=ribbon(base)), kw


def first_difference(a, b):
    """Same first differing write must come from identical paid inputs."""
    for i, (x, y) in enumerate(zip(a["trace"], b["trace"])):
        require(all(x[k] == y[k] for k in ("pre_state","rows","sampled","proposals","dirty_before",
                                            "sample_clock","layer_physical_q","edited","revoked")
                    if k in x or k in y), "diverged before an isolated admission")
        if x["admitted"] != y["admitted"]:
            u,v = set(x["admitted"]),set(y["admitted"])
            require((v-u) <= set(y["ribbon_additions"]) and (u|v) <= set(y["sampled"]),
                    "admission difference not paid")
            return dict(tick=i, before_only=sorted(u-v), after_only=sorted(v-u), q=y["q"])
    require(compact(a)==compact(b) and a["output_ids"]==b["output_ids"], "unexplained trajectory")
    return None


def run(args):
    import torch
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    require(sha(args.samples)==SAMPLES and sha(scoring.__file__)==SCORER_SHA and
            torch.cuda.device_count()==1 and os.environ["CUDA_VISIBLE_DEVICES"].startswith("GPU-"),
            "incorrect device or manifest")
    tasks=load_tasks(args.samples)
    args.out.mkdir(exist_ok=False)
    save(args.out/"run.json", dict(protocol="llada-risk-ribbon-v1", indices=ANCHORS, q=args.q,
         risk=RISK, device=os.environ["CUDA_VISIBLE_DEVICES"], samples=SAMPLES, scorer=SCORER_SHA,
         source=sha(__file__), checkpoint=str(args.checkpoint)))
    tok, model = load_gpu(args, lambda: None)
    classes,kw = arms(args.q)
    for j,i in enumerate(ANCHORS):
        ref=json.loads((args.reference/f"{i:02d}-native32.json").read_bytes())
        require(ref["index"]==i and ref["generation_cap"]==CAP,"wrong frozen sample")
        vals={k:evaluate_fixed(model,tok,tasks[i],ref["prompt_ids"],args.q,Law(),
                               request_cls=classes[k],audit_admission=True,**kw)
              for k in (tuple(classes) if j%2==0 else tuple(reversed(classes)))}
        a,b=vals["live"],vals["ribbon"]
        require(all(a[k]==b[k]==tasks[i][k] for k in KEYS) and a["input_sha256"]==b["input_sha256"] and
                all(r["max_decoder_Q"]<=args.q and r["Q"]==sum(e["q"] for e in r["trace"])
                    for r in vals.values()), "unpaired input or bill")
        first=first_difference(a,b)
        row=dict(**{k:a[k] for k in KEYS}, input=a["input_sha256"], first=first,
                 added=sum(len(e["ribbon_additions"]) for e in b["trace"]),**{k:compact(v) for k,v in vals.items()})
        save(args.out/f"{i:02d}.json",row)
        print(i, bool(first), a["F"], b["F"], flush=True)
    save(args.out/"complete.json",dict(indices=ANCHORS, q=args.q))


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for field in ("checkpoint","samples","reference","out"):
        ap.add_argument("--"+field,type=Path,required=True)
    ap.add_argument("--q",type=int,choices=TIERS,required=True)
    run(ap.parse_args())
