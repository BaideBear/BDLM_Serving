"""Compare complete same-board triples; never select different arms by offer."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from .coverage_span_eval import ANCHORS, KEYS, SAMPLES, SCORER_SHA, TIERS
from .risk_ribbon_report import bill


def check(ok, msg):
    if not ok:
        raise ValueError(msg)


def audit(root, manifest):
    m = json.loads(manifest.read_bytes())
    check(m["samples_sha256"] == SAMPLES, "unexpected frozen manifest")
    inputs = {}
    result = {}
    for q in TIERS:
        d = root / f"llada-coverage-span-v1-q{q}"
        reg = json.loads((d/"run.json").read_bytes())
        check(reg["offer"] == q and reg["protocol"] == "coverage-span-v1"
              and reg["samples"] == SAMPLES and reg["scorer"] == SCORER_SHA
              and tuple(reg["indices"]) == ANCHORS, "registration mismatch")
        check(json.loads((d/"complete.json").read_bytes()) == dict(indices=list(ANCHORS), q=q)
              and {p.stem for p in d.glob("[0-9][0-9].json")} == {f"{i:02}" for i in ANCHORS},
              "incomplete or unexpected pilot cohort")
        h = sha256()
        rows = []
        for i in ANCHORS:
            raw = (d / f"{i:02}.json").read_bytes()
            r = json.loads(raw)
            check(all(r[k] == m["samples"][i][k] for k in KEYS)
                  and len(r["input"]) == 64 and inputs.setdefault(i, r["input"]) == r["input"],
                  "case or prompt differs")
            for arm in ("live", "span", "covered"):
                a = r[arm]
                check(0 < a["max_decoder_Q"] <= q and
                      0 < a["F"] <= a["Q"] <= a["F"]*a["max_decoder_Q"] and
                      a["N"] >= 0 and 0 <= a["two_block_F"] <= a["F"],
                      "invalid physical bill")
            check(r["first_covered"] is not None or r["span"] == r["covered"],
                  "covered diverged without an intervention")
            rows.append(r)
            h.update(sha256(raw).digest())
        result[q] = dict(receipt_sha256=h.hexdigest(), registration=reg, cohorts={})
        for label, group in (("original", rows[:3]), ("additional", rows[3:]), ("all", rows)):
            stats = {}
            for arm in ("live", "span", "covered"):
                stats[arm] = dict(**bill(group, arm),
                    new_failures=sum(bool(r[arm]["failure"]) and not r["live"]["failure"] for r in group),
                    rescued=sum(r[arm]["correct"] and not r["live"]["correct"] for r in group),
                    lost=sum(r["live"]["correct"] and not r[arm]["correct"] for r in group),
                    two_block_F=sum(r[arm]["two_block_F"] for r in group))
            result[q]["cohorts"][label] = dict(n=len(group),
                covered_exposed=sum(r["first_covered"] is not None for r in group), **stats)
    check(len({tuple(sorted(x["registration"]["sources"].items())) for x in result.values()}) == 1,
          "mixed executor versions")
    return result


def decide(tiers):
    check(set(tiers) == set(TIERS), "expected all five offers")
    gs = [tiers[q]["cohorts"]["all"] for q in TIERS]
    check(all(g["n"] == len(ANCHORS) for g in gs), "unexpected cohort size")
    out = {}
    for arm in ("span", "covered"):
        passed = all(not g[arm]["new_failures"] and g[arm]["correct"] >= g["live"]["correct"]-1 for g in gs)
        gains = [g[arm]["macro_TPF"] > g["live"]["macro_TPF"] for g in gs]
        ties_or_better = [g[arm]["correct"] >= g["live"]["correct"] and
                          g[arm]["macro_TPF"] >= g["live"]["macro_TPF"] for g in gs]
        high = [j for j,q in enumerate(TIERS) if q in (32,64)]
        out[arm] = dict(passed=passed and sum(gains)>=3 and any(gains[j] and ties_or_better[j] for j in high),
                        tpf_gains=sum(gains), no_worse_both=sum(ties_or_better),
                        worst_high_tpf_ratio=min(gs[j][arm]["macro_TPF"]/gs[j]["live"]["macro_TPF"] for j in high))
    candidates = [a for a in out if out[a]["passed"]]
    best = max(candidates, key=lambda a:(out[a]["no_worse_both"], out[a]["worst_high_tpf_ratio"], a=="span"),
               default=None)
    return dict(arms=out, expand=best)


def extend(args, pilot):
    """Run the preselected whole algorithm against live on the 81 other IDs."""
    import os
    import torch
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.methods.t2t.budget_family import Law
    from . import coverage_span, coverage_span_eval
    from .cross_race_llada import load_gpu, save
    from .llada_fixed_family_eval import CAP, evaluate_fixed
    from .llada_hfull_single_factor import request_classes
    from .llada_risk_ribbon import arms, compact

    q = args.q
    choice = decide(pilot)["expand"]
    check(choice in ("span", "covered"), "screen did not pass; do not expand")
    parent = pilot[q]
    reg = parent["registration"]
    check(all(getattr(args, k) is not None for k in ("samples", "checkpoint", "reference")), "missing inputs")
    def sha(p):
        return sha256(Path(p).read_bytes()).hexdigest()
    check(sha(args.samples) == SAMPLES and sha(scoring.__file__) == SCORER_SHA and
          all(sha(p)==reg["sources"][Path(p).name] for p in (coverage_span.__file__,coverage_span_eval.__file__)),
          "changed registered sources, samples, or scorer")
    check(os.environ["CUDA_VISIBLE_DEVICES"] == reg["device"] and torch.cuda.device_count() == 1,
          "confirmation must use the pilot board")
    tasks = load_tasks(args.samples)
    # Balance datasets and old/new cases throughout the queue, not only at the end.
    ids = [i+j for i in range(30) for j in (0,30,60) if i+j not in ANCHORS]
    out = args.root / f"llada-coverage-span-v1-confirm-{choice}-q{q}"
    out.mkdir(exist_ok=False)
    save(out/"run.json", dict(**reg, extension=True, anchors_excluded=list(ANCHORS),
                               confirm_indices=ids, winner=choice, pilot_receipt_sha256=parent["receipt_sha256"],
                               checkpoint=str(args.checkpoint), code_sha256=sha(__file__)))
    tok, model = load_gpu(args, lambda:None)
    live, kw = arms(q)
    specs = dict(live=(live["live"],kw))
    specs[choice] = (coverage_span.request(request_classes()[1],choice=="covered"),
                     dict(cross=("quarter" if q==64 else "head",False)))
    for j,i in enumerate(ids):
        ref = json.loads((args.reference/f"{i:02d}-native32.json").read_bytes())
        check(ref["index"]==i and ref["generation_cap"]==CAP, "wrong native control")
        vals = {k:evaluate_fixed(model,tok,tasks[i],ref["prompt_ids"],q,Law(),
                                 request_cls=specs[k][0],audit_admission=True,**specs[k][1])
                for k in (tuple(specs) if j%2==0 else tuple(reversed(specs)))}
        check(all(all(r[key]==tasks[i][key] for key in KEYS) and r["input_sha256"]==vals["live"]["input_sha256"]
                  and r["max_decoder_Q"]<=q and r["Q"]==sum(e["q"] for e in r["trace"])
                  for r in vals.values()), "unmatched prompt, physical cap or ledger")
        row = dict(**{k:tasks[i][k] for k in KEYS},input=vals["live"]["input_sha256"])
        for k,r in vals.items():
            row[k] = dict(**compact(r), two_block_F=sum(e["head_q"]>0 and e["tail_q"]>0 for e in r["trace"]))
        save(out/f"{i:02d}.json",row)
        print(i, *(vals[k]["F"] for k in specs),flush=True)
    save(out/"complete.json",dict(q=q,indices=ids,winner=choice))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--confirm", dest="q", type=int, choices=TIERS)
    for key in ("checkpoint", "samples", "reference"):
        ap.add_argument("--"+key, type=Path)
    args = ap.parse_args()
    tiers = audit(args.root, args.manifest)
    if args.q is None:
        print(json.dumps(dict(tiers=tiers, decision=decide(tiers)), indent=2))
    else:
        extend(args,tiers)
