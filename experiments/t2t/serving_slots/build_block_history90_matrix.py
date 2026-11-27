"""Join the published controls and history tables; never infer missing tiers."""
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "controls"
COHORT = {
    "original-0-29": "original30", "original30": "original30",
    "added-30-89": "added60", "added60": "added60", "new60": "added60",
    "all-0-89": "all90", "all90": "all90", "merged90": "all90",
}
SIZE = {"original30": 30, "added60": 60, "all90": 90}


def read(name):
    with (ROOT / name).open(newline="") as src:
        yield from csv.DictReader(src)


def collect():
    rows = {}

    def add(model, method, tier, r, keys, source):
        part = COHORT[r["cohort"]]
        identity = model, method, int(tier), part
        requests, correct, tpf, eff, n, f, q = (r[k] for k in keys)
        if (identity in rows or int(requests) != SIZE[part]
                or not 0 <= int(correct) <= int(requests) or int(f) <= 0
                or not math.isclose(float(eff), int(n) / int(q), rel_tol=1e-8)):
            raise ValueError(f"Invalid or duplicate row: {identity}, {source}")
        rows[identity] = dict(model=model, method=method, setting=tier, cohort=part,
            requests=requests, correct=correct, macro_tpf=tpf, token_efficiency="",
            N=n, F=f, prefix_kv_Q="", physical_Q=q, physical_token_efficiency=eff, source=source)

    common = ("requests", "correct", "macro_tpf", "token_efficiency", "N", "F", "physical_Q")
    for name in ("block-history90-audited.csv", "block-history90-gemma-ibs512-h800.csv"):
        for r in read(name):
            # Retain the A800 sensitivity run in its source, but compare both
            # saturated IBS endpoints on the H800 in the main matrix.
            if (name == "block-history90-audited.csv" and r["model"] == "gemma"
                    and r["method"] == "ibs" and r["setting"] == "512"):
                continue
            add(r["model"], r["method"], r["setting"], r, common, name)
    extended = ("requests", "correct", "mean_tpf", "physical_token_efficiency",
                "sum_output_tokens", "sum_generation_forwards", "sum_generation_physical_padded_rows")
    for name in ("block-history90-llada-b3.csv", "block-history90-llada-w3-w4.csv",
                 "block-history90-gemma-w3-w4-b3.csv"):
        for r in read(name):
            if r["cohort"] not in COHORT:
                continue
            is_flow = r["policy"].startswith("flow_w")
            model = "gemma" if r["model"] == "diffusiongemma" else r["model"]
            tier = r["policy"].removeprefix("flow_w") if is_flow else (256 if model == "gemma" else 32)
            add(model, "flow" if is_flow else "b3", tier, r, extended, name)
    for model, label in (("gemma", "gemma"), ("llada21", "llada")):
        name = f"block-history90-{label}.csv"
        for r in read(name):
            add(model, "block_history", r["C"], r, common, name)
    # Only these series have audited counts of rows selected for prefix KV.
    pending = {k for k in rows if k[1] in ("native", "ibs", "block_history")}
    for r in read("block-history90-prefix-kv.csv"):
        k = r["model"], r["method"], int(r["setting"]), r["cohort"]
        if k not in pending:
            raise ValueError(f"Duplicate or unexpected prefix count: {k}")
        pending.remove(k)
        old = rows[k]
        q = int(r["prefix_kv_Q"])
        if (any(int(old[col]) != int(r[col]) for col in ("requests", "N", "F", "physical_Q"))
                or not 0 < q <= int(old["physical_Q"])
                or not math.isclose(float(r["prefix_kv_token_efficiency"]), int(old["N"]) / q)):
            raise ValueError(f"Prefix counts disagree with published records: {k}")
        old.update(prefix_kv_Q=q, token_efficiency=int(old["N"]) / q)
    if pending:
        raise ValueError(f"Missing prefix counts: {sorted(pending)}")
    for model, b in (("llada21", 32), ("gemma", 256)):
        for method, tiers in {"native": [b], "b3": [b], "flow": [1, 2, 3, 4],
                              "ibs": [b//8, b//4, b//2, b, 2*b],
                              "block_history": [b//8, b//4, b//2, b, 2*b]}.items():
            for c in tiers:
                slices = [rows[model, method, c, part] for part in SIZE]
                counts = ["requests", "correct", "N", "F", "physical_Q"]
                if slices[2]["prefix_kv_Q"] != "":
                    counts.append("prefix_kv_Q")
                for k in counts:
                    if int(slices[0][k]) + int(slices[1][k]) != int(slices[2][k]):
                        raise ValueError(f"Nonadditive cohorts: {model}/{method}/{c}/{k}")
                yield from slices
    assert len(rows) == 32 * 3


if __name__ == "__main__":
    matrix = list(collect())
    with (ROOT / "block-history90-matrix.csv").open("w", newline="") as dest:
        writer = csv.DictWriter(dest, matrix[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(matrix)
