"""Read private frozen90 request logs; print only pooled prefix-read counters.

Arguments: frozen90 manifest.json, archive root, JSON array of
[model, method, C, relative file glob(s)]. Globs must select exactly 90 requests.
"""
import csv
import json
import sys
from pathlib import Path


def require(ok, why):
    if not ok:
        raise ValueError(why)


def count(r, model, method, c):
    events = r["events"]
    if model == "llada21":
        steps = [e for e in events if e["phase"] != "prefill"]
        require(len(steps) == r["generation_forwards"], "LLaDA forward ledger")
        if method == "block_history":
            h = [e["history_q"] for e in steps]
            require(h == r["history_q_per_forward"] and sum(h) == r["history_rows"]
                    and all(0 <= n <= e["decoder_q"] for n, e in zip(h, steps))
                    and all((n > 0) == (e["history_edges"] > 0) for n, e in zip(h, steps))
                    and all(e.get("nominal_C", c) == c for e in steps),
                    "LLaDA history row/edge mismatch")
        elif method == "native":
            require(r["policy"] == f"native{c}" and all(e["phase"] in ("denoise", "commit")
                    for e in steps), "LLaDA native identity")
            h = [e["q_tokens"] if e["prefix_kv_length"] > 0 else 0 for e in steps]
        else:
            require(r["policy"] == f"optimus_ibs_c{c}" and all(e["phase"] == "optimus"
                    and e["mode"] == "ibs" and e["fixed_chunk"] == c and e["block"] > 0
                    and e["visible_kv"] >= 32 * e["block"]
                    and e["q_tokens"] == len(set(e["absolute_positions"])) for e in steps),
                    "Expected IBS with nonempty history and recorded selected rows")
            h = [e["q_tokens"] for e in steps]
    else:
        require(events[0][0] == "encoder" and events[0][1] > 0
                and len(events) - 1 == r["generation_forwards"]
                and sum(n for _, n in events[1:]) == r["generation_physical_padded_rows"],
                "Gemma generation ledger")
        decoder = [n for phase, n in events[1:] if phase == "decoder"]
        if method == "block_history":
            h = [s["history_q"] for s in r["trace"]]
            require(r["nominal_history_q"] == c and decoder == [s["q"] for s in r["trace"]]
                    and sum(h) == r["history_query_rows"]
                    and all(0 <= n <= q for n, q in zip(h, decoder))
                    and all((n > 0) == (s["history_edges"] > 0) for n, s in zip(h, r["trace"])),
                    "Gemma history row/edge mismatch")
        else:
            require(r["policy"] == (f"ibs_{c}" if method == "ibs" else f"native{c}"),
                    "Gemma baseline identity")
            if method == "ibs":
                require(decoder == [s["q"] for s in r["trace"]], "Gemma IBS selection mismatch")
            h = decoder  # Both dense-history backends attend the nonempty encoded prefix.
    require(0 < sum(h) <= r["generation_physical_padded_rows"], "Invalid prefix rows")
    return r["output_tokens"], r["generation_forwards"], r["generation_physical_padded_rows"], sum(h)


def main(manifest, root, jobs):
    tasks = {s["index"]: s for s in json.loads(manifest.read_text())["samples"]}
    require(set(tasks) == set(range(90)), "Expected frozen90")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("model", "method", "setting", "cohort", "requests", "N", "F",
                     "physical_Q", "prefix_kv_Q", "prefix_kv_token_efficiency"))
    for model, method, c, patterns in jobs:
        require(model in ("llada21", "gemma") and method in ("native", "ibs", "block_history"),
                "Unknown plot series")
        rows = {}
        for pattern in patterns:
            for f in root.glob(pattern):
                r = json.loads(f.read_bytes())
                i = r["index"]
                require(i not in rows and all(r[k] == tasks[i][k] for k in
                        ("index", "seed", "dataset", "task_id")), f"Duplicate or wrong task: {f}")
                rows[i] = count(r, model, method, c)
        require(rows.keys() == tasks.keys(), f"Incomplete {model} {method} {c}")
        for part, indices in (("original30", range(30)), ("added60", range(30, 90)),
                              ("all90", range(90))):
            n, forwards, physical, h = map(sum, zip(*(rows[i] for i in indices)))
            writer.writerow((model, method, c, part, len(indices), n, forwards, physical, h, n/h))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3]))
