"""Independent paired comparison on compact, complete fixed-budget receipts."""
import argparse
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
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


def run(root, model, samples, replay_dir=None):
    rows, hashes = {}, {}
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
        for i in range(90):
            path = root/"gemma-paid-agreement-q128-frozen90-01"/f"case-{i:02}.json"
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
            for key in ("stock", "apply"):
                b = d["arms"][key]["bill"]
                require(b["peak_decoder_Q"] == 128 and b["D"]+b["E"] == b["F"], "wrong offer or F")
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
                stock = json.loads(pair)["arms"]["stock"]
                require(all(stock[k] == dataset[i][k] for k in ("index","dataset","task_id","seed"))
                        and all(d["arms"]["stock"][k] == d["arms"]["tracked"][k]
                                for k in ("N","F","Q","ids_sha","correct","failure")), "identity or observer drift")
                add(d)
    return dict(model=model, fixed_q=128 if model=="gemma" else 16,
                hashes=hashes, cohorts=cohorts(rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("model", choices=["llada", "gemma"])
    p.add_argument("samples", type=Path)
    p.add_argument("--replay-dir", type=Path)
    args = p.parse_args()
    print(json.dumps(run(args.root, args.model, args.samples, args.replay_dir), sort_keys=True, indent=2))
