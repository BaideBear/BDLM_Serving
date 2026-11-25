"""Frozen native replay: can an abstaining logit-free ranker avoid displacing MASKs?"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from scipy.special import expit

from tools.edit_opportunity import fit, load
from bdlm_serving.evaluation.frozen import MANIFEST


def compare(valid, masks, old, fresh, ranked, positives, q, cap, threshold):
    seats = min(len(valid), max(q, len(fresh)))
    free = seats - len(fresh)
    available = max(0, free-int(bool(masks)))
    free_old = max(0, free-len(masks))
    # Always spend slack. Only displacement requires sufficient predicted hazard.
    chosen = [i for j,(score,i) in enumerate(ranked[:min(cap,available)])
              if j < free_old or score >= threshold]
    keep_masks = masks[:free-len(chosen)]
    selected = set(fresh) | set(chosen) | set(keep_masks)
    for i in old:
        if len(selected) == seats:
            break
        selected.add(i)
    assert len(selected) == seats and set(fresh) <= selected
    return Counter(steps=1, targets=len(positives), reserved=len(chosen),
        hits=len(positives.intersection(chosen)), incl_backfill=len(positives.intersection(selected)),
        masks_lost=min(len(masks),free)-len(keep_masks), reserved_rounds=int(bool(chosen)),
        physical_selected=len(selected))


def main(raw, out, weights=None):
    assert json.loads((raw/"verified.json").read_text()) == dict(model="llada21",matched_requests=30)
    tasks = json.loads(MANIFEST.read_text())["samples"]
    paths = [load(raw / f"{t['index']:02d}.trace.jsonl.gz",t) for t in tasks]
    data = defaultdict(Counter)
    if weights:
        from bdlm_serving.methods.t2t.hazard import read_weights
        frozen_weights = read_weights(weights)
    for fold in range(5):
        w = frozen_weights if weights else fit([path for i,path in enumerate(paths) if i%5 != fold])
        for i in range(fold,30,5):
            for phase,valid,masks,old,fresh,_,_,x,labels in paths[i]:
                if phase != "selected":
                    continue
                ranked = sorted(zip(expit(x @ w) if old else [], old),key=lambda v:(-v[0],v[1]))
                for q,cap in ((4,1),(8,1),(16,2),(16,4)):
                    for gate in (0.,.025,.05,.10,1.):
                        c = compare(valid,masks,old,fresh,ranked,labels.intersection(old),q,cap,gate)
                        for group in ("all",f"fold{fold}"):
                            data[group,q,cap,gate].update(c)
    rows = []
    for (fold,q,cap,gate),c in sorted(data.items()):
        ref = data[fold,q,cap,0.]
        rows.append(dict(group=fold,q=q,cap=cap,threshold=gate,**c,
            retained_hits=c["hits"]/ref["hits"] if ref["hits"] else "",
            mask_displacement_saved=1-c["masks_lost"]/ref["masks_lost"] if ref["masks_lost"] else ""))
        if fold == "all":
            assert c["targets"] == 191 and c["physical_selected"] == ref["physical_selected"]
    with out.open("w",newline="") as f:
        writer = csv.DictWriter(f,rows[0],lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("raw","out"):
        p.add_argument("--"+k, type=Path, required=True)
    p.add_argument("--weights", type=Path, help="Use a single, independently calibrated coefficient vector")
    a = p.parse_args()
    main(a.raw,a.out,a.weights)
