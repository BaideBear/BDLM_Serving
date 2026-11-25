"""Export only measured counts and correctness; never export answers or prompt IDs."""
import csv
import json
from pathlib import Path


def export_cached(run, dest, manifest):
    with (run/"scores.csv").open(newline="") as f:
        scores = sorted(csv.DictReader(f), key=lambda r: int(r["index"]))
    assert len(scores) == len(manifest) == 30
    rows = []
    for m, s in zip(manifest, scores):
        r = json.loads((run/f'{m["index"]:02d}-native32_cached.json').read_text())
        for k in ("index", "seed", "task_id", "dataset"):
            assert r[k] == m[k]
        for k in ("index", "task_id", "dataset", "model", "policy", "status", "output_tokens",
                  "generation_forwards", "generation_query_rows", "generation_physical_padded_rows", "max_q"):
            assert str(r[k]) == s[k]
        assert r["policy"] == "native32_cached" and r["model"] == "llada20"
        x = r["extra"]
        pf, pq = x["phase_forwards"], x["phase_q"]
        f, q = r["generation_forwards"], r["generation_query_rows"]
        assert r["q_histogram"] == {"32": f} and q == r["generation_physical_padded_rows"] == 32*f
        assert f == pf["denoise"] + pf["commit"] and q == pq["denoise"] + pq["commit"]
        assert pq["commit"] == 32*pf["commit"] and pq["denoise"] == 32*pf["denoise"]
        prefill = len(r["prompt_ids"])//32*32
        assert prefill == r["prefill_rows"] == pq.get("prefill", 0)
        assert pf.get("prefill", 0) == int(prefill > 0)
        assert x["max_visible_kv"] == prefill + 32*pf["commit"]
        assert x["terminal_commit"] and x["layers_observed"] == 20
        assert r["generation_cap"] == 2048 and r["t2t_edits"] == 0
        rows.append({k: r[k] for k in ("policy", "index", "dataset", "task_id", "seed",
            "status", "output_tokens", "generation_forwards", "generation_query_rows",
            "generation_physical_padded_rows", "max_q", "truncated", "t2t_edits",
            "remaining_masks", "prefill_rows")} | dict(
            denoise_f=pf["denoise"], denoise_q=pq["denoise"], commit_f=pf["commit"],
            commit_q=pq["commit"], prefill_f=pf.get("prefill", 0), max_visible_kv=x["max_visible_kv"],
            layers_observed=x["layers_observed"], correct=int(s["correct"].lower()=="true")))
    with Path(dest).open("w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
