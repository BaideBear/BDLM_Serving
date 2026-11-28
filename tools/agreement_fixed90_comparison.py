"""Independent paired comparison on compact, complete fixed-budget receipts."""
import argparse
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import random
from statistics import mean

SIX = {7, 13, 25, 33, 55, 71}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def decompose(a, b):
    """Symmetric accounting of N/F, not a causal intervention."""
    n, f = a["N"], a["F"]
    m, g = b["N"], b["F"]
    require(f > 0 and g > 0, "cannot decompose an unbilled request")
    return (m-n)*(1/f+1/g)/2, (n+m)*(1/g-1/f)/2


def cohorts(rows):
    require(set(rows) == set(range(90)), "not 90 unique paired requests")
    result = {}
    for cohort, indices in (("frozen90", range(90)), ("original30", range(30)),
                            ("new60", range(30, 90)), ("screen6", sorted(SIX)),
                            ("outside84", sorted(set(rows)-SIX))):
        cases = [rows[i] for i in indices]
        s = {}
        for key in ("stock", "apply"):
            arms = [v[key] for v in cases]
            sums = {k: sum(v[k] for v in arms) for k in ("N", "F", "Q", "correct")}
            sums.update(tpf=mean(v["N"]/v["F"] for v in arms), te=sums["N"]/sums["Q"])
            s[key] = sums
        deltas = [decompose(row["stock"], row["apply"]) for row in cases]
        den, frc = map(mean, zip(*deltas))
        require(isclose(den+frc, s["apply"]["tpf"]-s["stock"]["tpf"], abs_tol=1e-10),
                "TPF decomposition drift")
        s.update(requests=len(cases), d_tpf_from_N=den, d_tpf_from_F=frc,
                 gained=[i for i in indices if rows[i]["apply"]["correct"] and not rows[i]["stock"]["correct"]],
                 lost=[i for i in indices if not rows[i]["apply"]["correct"] and rows[i]["stock"]["correct"]],
                 F_less=sum(r["apply"]["F"] < r["stock"]["F"] for r in cases),
                 F_equal=sum(r["apply"]["F"] == r["stock"]["F"] for r in cases),
                 F_more=sum(r["apply"]["F"] > r["stock"]["F"] for r in cases))
        result[cohort] = s
    return result


def paired_intervals(rows, dataset, count=4000):
    """Descriptive paired percentile intervals, resampling requests within dataset."""
    categories = sorted({r["dataset"] for r in dataset})
    if len(rows)!=len(dataset) or len(categories)!=3 or count<100:
        raise ValueError("incomplete cohort or too few bootstrap draws")
    groups=[]
    for category in categories:
        group=[]
        for t in dataset:
            if t["dataset"] != category: continue
            a,b=(rows[t["index"]][k] for k in ("stock","apply"))
            group.append((int(b["correct"])-int(a["correct"]),b["N"]/b["F"]-a["N"]/a["F"],
                          b["F"]-a["F"],a["N"],a["Q"],b["N"],b["Q"]))
        if len(group)!=30:raise ValueError("wrong dataset size")
        groups.append(group)
    rng=random.Random(1128)
    draws={k:[] for k in ("accuracy_pp","macro_tpf","mean_F","pooled_te_pp")}
    for _ in range(count):
        v=[x for group in groups for x in rng.choices(group,k=len(group))]
        accuracy,tpf,df,an,aq,bn,bq=map(sum,zip(*v))
        for label,x in zip(draws,(100*accuracy/len(v),tpf/len(v),df/len(v),100*(bn/bq-an/aq)),strict=True):
            draws[label].append(x)
    def interval(values):
        a=sorted(values)
        return [a[round((count-1)*p)] for p in (.025,.975)]
    return dict(seed=1128,draws=count,delta="apply_minus_stock",
                ci95={k:interval(v) for k,v in draws.items()})


def prefill_digests(path):
    """Hash the token IDs physically passed to the historical prefill encoder."""
    found = {}
    with path.open() as stream:
        for line in stream:
            if not line.startswith("RECEIPT "):
                continue
            case = json.loads(line[8:])
            if case.get("kind") != "case":
                continue
            i = case["index"]
            require(i not in found, "duplicate historical prefill")
            tokens = case["arms"]["paid"]["encoder"][0][0]
            require(tokens and all(type(t) is int for t in tokens), "bad prefill tape")
            found[i] = sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
    require(set(found) == set(range(90)), "missing historical prefills")
    return found


def dataset_cost(rows, dataset):
    groups = {}
    worst = []
    for item in dataset:
        x = rows[item["index"]]
        a, b = x["stock"], x["apply"]
        name = item["dataset"]
        g = groups.setdefault(name, dict(requests=0, stock_correct=0, apply_correct=0,
                                         delta_N=0, delta_F=0, delta_Q=0))
        g["requests"] += 1
        g["stock_correct"] += a["correct"]
        g["apply_correct"] += b["correct"]
        for key in ("N", "F", "Q"):
            g[f"delta_{key}"] += b[key] - a[key]
        worst.append(dict(index=item["index"], dataset=name, delta_F=b["F"]-a["F"],
                          stock=a, apply=b))
    return dict(datasets=groups, greatest_extra_F=sorted(
        worst, key=lambda x:(-x["delta_F"], x["index"]))[:3])


def run(root, model, samples, replay_dir=None, *, gemma_dir=None, fixed_q=None,
        with_intervals=False):
    rows, hashes = {}, {}
    q = fixed_q if fixed_q is not None else (128 if model=="gemma" else 16)
    require((model=="llada" and q==16 and not gemma_dir and not replay_dir) or
            (model=="gemma" and ((q==128 and not gemma_dir) or
                                  (q==64 and gemma_dir and not replay_dir))), "unknown receipt profile")
    dataset = json.loads(samples.read_bytes())["samples"]
    require(sha256(samples.read_bytes()).hexdigest() ==
            "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed", "sample drift")
    def add(d):
        i = d["index"]
        require(i not in rows and i in range(90), "duplicate or foreign receipt")
        rows[i] = {}
        for key in ("stock", "apply"):
            a = d["arms"][key]
            bill = a.get("bill", a)
            rows[i][key] = {k: bill[k] for k in ("F", "Q")} | {k: a[k] for k in ("N", "correct")}
            require(type(a["correct"]) is bool and a["N"] >= 0 and
                    bill["F"] > 0 and bill["Q"] > 0, "missing or invalid bill")
    if model == "gemma":
        validated = set()
        if q == 64:
            prompt_source = root/"gemma-ibs-certified-eos-a0-q64-20261128T110532Z"/"raw.log"
            prompts = prefill_digests(prompt_source)
            hashes["prefill_reference"] = sha256(prompt_source.read_bytes()).hexdigest()
        for i in range(90):
            path = (gemma_dir or root/"gemma-paid-agreement-q128-frozen90-01")/f"case-{i:02}.json"
            raw = path.read_bytes()
            d = json.loads(raw)
            if d["error"] is not None and replay_dir:
                proof_path = replay_dir/f"case-{i:02}.json"
                proof_raw = proof_path.read_bytes()
                v = json.loads(proof_raw)
                require(v["original_sha256"] == sha256(raw).hexdigest()
                        and all(v[k] == d[k] for k in ("index","task_id","dataset","seed","order","arms")),
                        "replay changed generation or bill")
                hashes[f"replay/{proof_path.name}"] = sha256(proof_raw).hexdigest()
                validated.add(i)
                d = v
            require(d["index"] == i and d["stock_observer_identical"] and d["error"] is None
                    and all(d[k] == dataset[i][k] for k in ("index","dataset","task_id","seed")),
                    "missing comparable GPU pair; do not exclude a failed case")
            if q==64:
                require(d["order"] == (["stock","observer","apply"] if i%2==0 else
                                       ["apply","observer","stock"]), "unexpected arm order")
                require(d["prompt_sha256"] == prompts[i] and
                        all(d["arms"]["stock"][k] == d["arms"]["observer"][k] for k in
                            ("bill","trigger","payment","correct","N","scoring","rng_after")),
                        "reference prompt or observer changed")
                if i in SIX:
                    dev = json.loads((gemma_dir.parent/"six"/path.name).read_bytes())
                    require(all(d[k]==dev[k] for k in ("index","seed","task_id","dataset","prompt_sha256",
                                                       "first_extra","error","arms")), "six screen not reproduced")
            for key in ("stock", "apply", "observer"):
                b = d["arms"][key]["bill"]
                require(b["peak_decoder_Q"] == q and b["D"]+b["E"] == b["F"] and
                        0<b["Q"]<=q*b["D"] and b["admissions"]==b["paid_rows"], "wrong Q, F, or debt")
            hashes[path.name] = sha256(raw).hexdigest()
            add(d)
        if replay_dir:
            manifest = json.loads((replay_dir/"replay.json").read_bytes())
            require(sorted(validated) == manifest["verified"] and len(validated) == manifest["n"]
                    and manifest["gpu"] == "GPU-94c61e79-d80b-cae2-d85f-6bd2a2c99a1c",
                    "replay identity mismatch")
    else:
        for name, ids in (("llada-ibs-q16-paid-agreement-h0-v1", SIX),
                          ("llada-ibs-q16-paid-agreement-frozen90-h0-v1", set(range(90))-SIX)):
            folder = root/name
            raw = (folder/"receipts.ndjson").read_bytes()
            notes = [json.loads(line) for line in raw.splitlines()]
            require({d["index"] for d in notes} == ids and len(notes) == len(ids), "unsealed cohort")
            require((folder/"returned.identity").is_file(), "H0 handoff missing")
            hashes[name] = sha256(raw).hexdigest()
            for d in notes:
                i = d["index"]
                pair = (folder/"pairs"/f"{i:02}.json").read_bytes()
                require(sha256(pair).hexdigest() == d["raw_sha"], "raw pair changed")
                raw_arms = json.loads(pair)["arms"]
                stock = raw_arms["stock"]
                require(all(stock[k] == dataset[i][k] for k in ("index","dataset","task_id","seed"))
                        and all(d["arms"]["stock"][k] == d["arms"]["tracked"][k]
                                for k in ("N","F","Q","ids_sha","correct","failure")), "identity or observer drift")
                for key, a in raw_arms.items():
                    bills = d["arms"][key]
                    calls = a["calls"][1:]  # first call is prompt prefill
                    failed = a["status"] == "generation_failed"
                    require((a["extra"].get("physical_caps") == [q] or
                             (failed and a["request_failure"]["type"] == "optimus_progress_limit"
                              and not bills["correct"] and bills["N"] == 0))
                            and a["extra"]["fixed_chunk"] == q and a["verified_layers"] == 20
                            and a["prompt_ids"] == stock["prompt_ids"]
                            and len(calls) == a["generation_forwards"] == bills["F"]
                            and sum(c["rows"] for c in calls) == a["generation_query_rows"] == bills["Q"]
                            and all(0 < c["rows"] == c["physical_rows"] <= q for c in calls)
                            and a["output_tokens"] == bills["N"], "physical LLaDA bill mismatch")
                add(d)
    out=dict(model=model, fixed_q=q, hashes=hashes, cohorts=cohorts(rows))
    if with_intervals:
        out["paired_intervals"]=paired_intervals(rows,dataset)
        out["cost_breakdown"]=dataset_cost(rows,dataset)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("model", choices=["llada", "gemma"])
    p.add_argument("samples", type=Path)
    p.add_argument("--replay-dir", type=Path)
    p.add_argument("--gemma-dir", type=Path)
    p.add_argument("--fixed-q", type=int)
    p.add_argument("--with-intervals",action="store_true")
    args = p.parse_args()
    print(json.dumps(run(args.root, args.model, args.samples, args.replay_dir,
                         gemma_dir=args.gemma_dir, fixed_q=args.fixed_q,
                         with_intervals=args.with_intervals), sort_keys=True, indent=2))
