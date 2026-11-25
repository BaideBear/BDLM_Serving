"""Read the original run tables and assemble the three frozen30 study views."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
BASE, PREFIX, OBS = "baselines", "flowblock-prefix", "optimus-masked-obs"
L20, CACHE, GN, GW = "llada20-frozen30", "llada20-cached-b32", "diffusiongemma-native", "diffusiongemma-wave"
MODELS = {"llada20": "LLaDA 2.0-mini", "llada21": "LLaDA 2.1-mini", "sdar": "SDAR-8B", "gemma": "DiffusionGemma 26B-A4B"}
CHUNKS = (4, 8, 16, 32)
GEMMA_CHUNKS = (*CHUNKS, 64, 128, 256)
GO = "diffusiongemma-optimus"
ORDER = ["native", *(f"flow_w{w}" for w in range(1, 5)), "b3",
         *(f"{m}_{c}" for m in ("ibs", "obs") for c in GEMMA_CHUNKS), "hf_uncached"]


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def boolean(value):
    return {"True": 1, "False": 0, "1": 1, "0": 0}[str(value)]


def policy(p):
    if p.startswith("optimus_"):
        return p.removeprefix("optimus_").replace("_c", "_")
    if p.startswith("native"):
        return "native"
    return "b3" if p.startswith("b3") else p


def collect():
    manifest = json.loads((ROOT/"benchmarks/frozen30/manifest.json").read_text())["samples"]
    tasks = {int(t["index"]): t for t in manifest}
    raw = defaultdict(dict)
    for run in (BASE, PREFIX, OBS, L20, CACHE, GN, GW, GO):
        for r in read(ROOT/"archive"/run/"requests.csv"):
            model = r.get("model", "llada20" if run in (L20, CACHE) else "gemma")
            orig = (r["arm"]+"_"+r["cap"]) if run == OBS else r.get("policy", "native256")
            p = "hf_uncached" if run == L20 and orig == "native32" else policy(orig)
            i = int(r["index"])
            task = tasks[i]
            assert task["dataset"] == r["dataset"] and task["task_id"] == r.get("task_id", r.get("task"))
            assert "seed" not in r or int(r["seed"]) == task["seed"]
            keys = ("N F Q physical_Q" if run == OBS else
                    "output_tokens generation_forwards generation_query_rows generation_physical_padded_rows")
            n, f, q, qp = (int(r[k]) for k in keys.split())
            assert n >= 0 and qp >= q >= f > 0
            d = dict(index=i, dataset=task["dataset"], task_id=task["task_id"],
                     N=n, F=f, Q=q, physical_Q=qp, correct=boolean(r["correct"]),
                     failed=boolean(r.get("failed", r.get("status") == "generation_failed")),
                     truncated=boolean(r.get("truncated", False)))
            if run == CACHE:
                assert q == 32*f and int(r["commit_q"])+int(r["denoise_q"]) == q
            group = raw[run, model, p]
            assert i not in group
            group[i] = d
    assert all(set(g) == set(tasks) for g in raw.values())
    # Reused OBS/IBS and Gemma Native are aliases, not additional executions.
    for (run, m, p), g in raw.items():
        if run == OBS and not p.startswith("bobs_"):
            assert g == raw[BASE, m, p]
        if run == GW and p == "native":
            assert g == raw[GN, m, p]
    studies = {k: [] for k in (BASE, PREFIX, OBS)}
    def add(study, model, p, source, role):
        rows = raw[source, model, p]
        assert len(rows) == 30, (source, model, p)
        tag = dict(model=model, policy=p, role=role, run=source)
        studies[study].append((tag, [rows[i] for i in sorted(rows)]))
    for m in MODELS:
        for p in ORDER:
            if p == "hf_uncached" and m != "llada20":
                continue
            opt = p.startswith(("obs_", "ibs_"))
            if m != "gemma" and opt and int(p.split("_")[1]) not in CHUNKS:
                continue
            if m == "gemma":
                src = GO if opt else GN if p == "native" else GW
            elif m == "llada20":
                src = CACHE if p == "native" else L20
            else:
                src = PREFIX if p == "flow_w4" else BASE
            add(BASE, m, p, src, "uncached_reference" if p == "hf_uncached" else "baseline")
            if m in ("llada21", "sdar") and p in ("flow_w2", "flow_w3"):
                add(BASE, m, p, PREFIX, "paired_repeat")
        if m != "gemma":
            for w in (2, 3, 4):
                for method in ("flow", "prefix"):
                    add(PREFIX, m, f"{method}_w{w}", L20 if m == "llada20" else PREFIX,
                        "control" if method == "flow" else "variant")
        if m in ("llada21", "sdar"):
            for c in CHUNKS:
                for method, role in (("ibs", "reference"), ("obs", "control"), ("bobs", "variant")):
                    add(OBS, m, f"{method}_{c}", OBS if method == "bobs" else BASE, role)
    measured = {(t["run"], t["model"], t["policy"]) for groups in studies.values() for t, _ in groups}
    aliases = {k for k in raw if (k[0] == OBS and not k[2].startswith("bobs_"))
               or (k[0] == GW and k[2] == "native")}
    assert measured == raw.keys()-aliases and len(measured) == 84
    return studies


def metrics(rows):
    t = {k: sum(r[k] for r in rows) for k in ("N", "F", "Q", "physical_Q", "correct", "failed", "truncated")}
    return dict(requests=len(rows), **t, tpf=mean(r["N"]/r["F"] for r in rows),
                efficiency=t["N"]/t["Q"], q_per_f=t["Q"]/t["F"],
                physical_efficiency=t["N"]/t["physical_Q"])


def tables(groups):
    requests, summary = [], []
    for tag, group in groups:
        requests.extend({**tag, **r} for r in group)
        for dataset in ("all", "gsm8k", "math500", "humaneval"):
            sub = group if dataset == "all" else [r for r in group if r["dataset"] == dataset]
            assert len(sub) == (30 if dataset == "all" else 10)
            summary.append(dict(**tag, dataset=dataset, **metrics(sub)))
    return requests, summary


def pairs(study, groups):
    lookup = {(t["model"], t["policy"]): (t, g) for t, g in groups}
    result = []
    for variant, treatment in groups:
        if variant["role"] != "variant":
            continue
        m, p = variant["model"], variant["policy"]
        refs = [p.replace("prefix_", "flow_")] if study == PREFIX else [
            p.replace("bobs_", prefix) for prefix in ("obs_", "ibs_")]
        for ref in refs:
            base, control = lookup[m, ref]
            assert [r["task_id"] for r in treatment] == [r["task_id"] for r in control]
            a, b = metrics(control), metrics(treatment)
            flips = [y["correct"]-x["correct"] for x, y in zip(control, treatment)]
            result.append(dict(model=m, baseline=ref, variant=p, baseline_run=base["run"],
                variant_run=variant["run"], baseline_role=base["role"], gained=flips.count(1),
                lost=flips.count(-1), **{f"{side}_{k}": v[k] for k in ("tpf","efficiency","correct","q_per_f","N","F","Q","physical_Q")
                                         for side,v in (("baseline",a),("variant",b))},
                delta_tpf=b["tpf"]-a["tpf"], delta_efficiency_pp=100*(b["efficiency"]-a["efficiency"]),
                delta_correct=b["correct"]-a["correct"]))
    return result
