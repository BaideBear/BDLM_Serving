"""CPU-only frozen30 audit of observed ranked strict hits, not a veto effect.

Run with python -m experiments.t2t.exchange.write_veto.historical --help.
Archived marks are cross-checked, not rescored. No raw text, tokens or row
positions are exported. A hit is a logged reserved row written strictly on
the actual ranked path; repeated writes to the same position count again.
"""

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from math import isclose
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[4]
IDENTITY = ("index", "dataset", "task_id", "seed")
DATASETS = ("gsm8k", "math500", "humaneval")
BRIDGE = (*IDENTITY, "output_ids", "output_tokens", "generation_forwards",
          "generation_query_rows", "generation_physical_padded_rows", "max_q")
WORK = dict(N="output_tokens", F="generation_forwards", Q="generation_query_rows",
            physical_Q="generation_physical_padded_rows", max_q="max_q")
WIDTH = 32
LAYERS = 20
SOURCES = {"ranked": "exchange-v1-edit4", "zero": "hard-v1-llada21"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def identity(row, task, fields=IDENTITY):
    require(all(row[k] == task[k] for k in fields), f"Wrong identity at {task['index']}")


def unique(rows, key, expected):
    result = {key(r): r for r in rows}
    require(len(result) == len(rows) and result.keys() == set(expected),
            "Duplicate, missing or extra keys in published evidence")
    return result


def check_csv(row, expected, label):
    for k, v in expected.items():
        matching = (isclose(float(row[k]), v, rel_tol=0, abs_tol=1e-12)
                    if isinstance(v, float) else row[k] == str(v))
        require(matching, f"{label}: disagreement in {k}")


def policy(q, ranked):
    return f"t2t_q{q}_f0_m{q-1}_hard_exchange_edit4" if ranked else f"t2t_q{q}_f0_m{q}_hard_noedit"


def positions(values):
    require(isinstance(values, list) and all(type(p) is int for p in values)
            and values == sorted(set(values)), "Invalid physical/reserved position list")
    return set(values)


def scan(row, q, prompt_length, ranked):
    """Validate archived accounting and extract only observed (row, event) counts."""
    require(row["status"] == "success", "Unsuccessful request")
    n, f, total_q, physical_q, max_q = (row[k] for k in WORK.values())
    require(f > 0 and total_q == physical_q >= f and n >= 0, "Invalid request work")
    require(type(row["eos"]) is bool and type(row["truncated"]) is bool
            and row["eos"] != row["truncated"] and len(row["output_ids"]) == n + row["eos"],
            "Output length/EOS mismatch")
    require(isclose(row["tpf"], n/f, rel_tol=0, abs_tol=1e-12)
            and isclose(row["token_efficiency"], n/total_q, rel_tol=0, abs_tol=1e-12),
            "Raw TPF/efficiency mismatch")
    fs, qs, hist, cost = Counter(), Counter(), Counter(), Counter()
    first_swap = first_hit = first_hit_forward = None
    prefill = edits = 0
    prior_block = prior_masks = None
    for clock, e in enumerate(row["events"], 1):
        stage, w, start = e["stage"], e["physical_q"], e["block"] * WIDTH
        label = f"index {row['index']}, clock {clock}"
        picked = positions(e["absolute_positions"])
        reserved = positions(e["reserved_extra_edit_positions"])
        hits = positions(e["reserved_strict_edit_positions"])
        fresh, strict = e["mandatory_fresh"], e["hard_strict_edits"]
        require(e["hard_clock"] == clock and e["q_budget"] == q and e["F"] == 1
                and w > 0 and w == e["actual_q"] == e["q_tokens"] == len(picked)
                and e["layer_physical_q"] == [w] * LAYERS
                and e["overflow"] == max(0, w-q), f"Physical width/clock mismatch: {label}")
        require(hits <= reserved <= picked and len(hits) <= strict <= w
                and len(reserved) <= int(ranked and stage == "selected"),
                f"Invalid reserved/strict positions: {label}")
        if stage == "prefill":
            require(clock == 1 and e["phase"] == "prefill" and start == prompt_length//WIDTH*WIDTH
                    and picked == set(range(start)) and strict == 0, f"Incorrect prefill: {label}")
            prefill += w
            continue
        if stage == "selected":
            editable = range(max(start, prompt_length), start + WIDTH)
            require(e["approximate_same_block_kv"] and picked <= set(editable)
                    and 0 <= fresh <= len(editable) and w == min(len(editable), max(q, fresh)),
                    f"Incorrect selected width: {label}")
            masks = e["selected_masked_before"]
            displaced = min(e["masked_before"], w-fresh) - masks
            neutral = w-fresh-masks-len(reserved)
            require(e["masked_before"] > 0 and 0 <= displaced <= len(reserved)
                    and (not reserved or masks > 0) and neutral >= 0,
                    f"Invalid MASK/mandatory/reserved partition: {label}")
            cost.update(selected_steps=1, selected_Q=w, mandatory_Q=fresh, mask_Q=masks,
                        neutral_backfill_Q=neutral, mandatory_overflow_Q=max(0, fresh-q),
                        reserved_seats=len(reserved), reserved_strict_hits=len(hits),
                        displaced_mask_rows=displaced, steps_displacing_mask=bool(displaced),
                        reserved_hits_on_displacing_steps=len(hits)*bool(displaced))
        else:
            require(stage in ("bootstrap", "verify", "commit") and w == WIDTH
                    and picked == set(range(start, start+WIDTH)), f"Incorrect dense width: {label}")
            cost["nonsparse_nondiagnostic_Q"] += w
        if stage != "commit":
            before, selected, accepted, after = (e[k] for k in
                ("masked_before", "selected_masked_before", "accepted", "masked_after"))
            require(0 <= accepted <= selected <= min(w, before) <= WIDTH
                    and before <= WIDTH and before-accepted <= after <= before-accepted+e["edited"]
                    and strict == e["edited"] - (after-before+accepted)
                    and (prior_block != e["block"] or prior_masks == before),
                    f"MASK-transition/strict mismatch: {label}")
            require(stage == "selected" or selected == before, f"Dense MASK count: {label}")
            edits += e["edited"]
            prior_block, prior_masks = e["block"], after
        else:
            require(strict == 0, f"Commit has strict writes: {label}")
        fs[stage] += 1
        qs[stage] += w
        hist[str(w)] += 1
        if reserved and first_swap is None:
            first_swap = clock
        if hits and first_hit is None:
            first_hit, first_hit_forward = e, sum(fs.values())
    require(prefill == row["prefill_rows"] == prompt_length//WIDTH*WIDTH
            and edits == row["t2t_edits"] and dict(fs) == row["forwards_by_phase"]
            and dict(qs) == row["q_by_phase"] and dict(hist) == row["q_histogram"]
            and sum(fs.values()) == f and sum(qs.values()) == total_q and max(map(int, hist)) == max_q,
            "Event/F/Q/edits/prefill totals disagree")
    return cost, first_swap, first_hit, first_hit_forward, fs, qs


def check_first_swap(ranked, zero, swap):
    """Width is comparable at the shared state, not on later diverged paths."""
    a, b = ranked["events"], zero["events"]
    if swap is None:
        require(a == b and all(ranked[k] == zero[k] for k in (*BRIDGE, "eos", "truncated")),
                "No-swap request diverged")
        return None
    i = swap - 1
    require(i < len(b) and a[:i] == b[:i], "Histories differ before first swap")
    x, y = a[i], b[i]
    require(all(x[k] == y[k] for k in ("stage", "block", "q_budget", "hard_clock", "step_in_block",
                "physical_q", "layer_physical_q", "context_version", "prefix_kv_length",
                "masked_before", "mandatory_fresh")), "First swap does not preserve state/width")
    new, old = set(x["absolute_positions"]), set(y["absolute_positions"])
    require(new-old == set(x["reserved_extra_edit_positions"]) and len(old-new) == len(new-old) == 1,
            "First swap does not replace exactly one physical row")
    displaced = y["selected_masked_before"] - x["selected_masked_before"]
    require(displaced in (0, 1), "Invalid first-swap MASK displacement")
    return x["physical_q"]


def aggregate(rows):
    out = dict(requests=len(rows), rank_correct=sum(r["rank_correct"] for r in rows),
               zero_correct=sum(r["zero_correct"] for r in rows))
    out.update({k: sum(r["outcome"] == k for r in rows) for k in ("both", "win", "loss", "neither")})
    out.update(hit_requests=sum(r["hits"] > 0 for r in rows), hits=sum(r["hits"] for r in rows),
               swap_requests=sum(r["first_swap_clock"] is not None for r in rows),
               rank_tpf=mean(r["rank_tpf"] for r in rows), zero_tpf=mean(r["zero_tpf"] for r in rows),
               mean_delta_tpf=mean(r["delta_tpf"] for r in rows),
               contribution_to_all_delta_tpf=sum(r["delta_tpf"] for r in rows)/30)
    clocks = [r["first_hit_clock"] for r in rows if r["hits"]]
    out.update(first_hit_clock_min=min(clocks, default=None), first_hit_clock_median=median(clocks) if clocks else None,
               first_hit_clock_max=max(clocks, default=None))
    return out


def strata(rows):
    groups = {("all", "all"): rows}
    for r in rows:
        clock = r["first_hit_clock"]
        timing = "no_hit" if clock is None else "01-32" if clock <= 32 else "33-128" if clock <= 128 else "129+"
        for group in (("ever_hit", "yes" if r["hits"] else "no"), ("hit_count", str(r["hits"])),
                      ("first_hit_clock", timing), ("outcome", r["outcome"])):
            groups.setdefault(group, []).append(r)
    return [dict(q=rows[0]["q"], group=k, level=v, **aggregate(rs)) for (k, v), rs in sorted(groups.items())]


def load_arm(folder, book, tasks, q, ranked):
    p = policy(q, ranked)
    config = read(folder/"run.json")
    arm = dict(budget=q, fresh=0, mask=q-int(ranked), selection=p.split("_", 4)[-1])
    require(all(config[k] == v for k, v in dict(model="llada21", batch_size=1, block=WIDTH, cap=2048).items())
            and config.get("seed_offset", 0) == 0 and config["policies"].count(p) == 1
            and config["arms"][config["policies"].index(p)] == arm,
            f"Wrong run protocol: {p}")
    if ranked:
        require(config["exchange_rates_sha256"] == sha(book/"rates.json"), "Exchange rates digest mismatch")
    require(read(folder/"qualified.json") == dict(model="llada21", full_width=WIDTH, matched=30)
            and read(folder/f"complete-{p}.json") == dict(policy=p, requests=30), f"Incomplete arm: {p}")
    marks = read(folder/f"marks-{p}.json")
    indices = set(tasks)
    require(marks.keys() == {str(i) for i in indices} and all(type(v) is bool for v in marks.values())
            and {x.name for x in folder.glob(f"*-{p}.json") if x.name[0].isdigit()}
            == {f"{i:02d}-{p}.json" for i in indices}, f"Incomplete or unexpected raw requests/marks: {p}")
    table = unique([r for r in csv_rows(book/"llada21-online-requests.csv") if r["policy"] == p],
                   lambda r: int(r["index"]), indices)
    raws, expected = {}, {}
    for i, task in tasks.items():
        raw = read(folder/f"{i:02d}-{p}.json")
        identity(raw, task)
        require(raw["model"] == "llada21" and raw["policy"] == p, f"Wrong model/arm: {p}, {i}")
        expected[i] = dict(**{k: raw[v] for k, v in WORK.items()}, correct=int(marks[str(i)]),
                           failed=0, truncated=int(raw["truncated"]))
        check_csv(table[i], dict(**{k: task[k] for k in IDENTITY[:-1]}, **expected[i], model="llada21",
                                nominal_q=q, fresh=0, mask=q-int(ranked), edit=int(ranked)), f"{p}, {i}")
        raws[i] = raw
    summaries = unique([r for r in csv_rows(book/"llada21-online-summary.csv") if r["policy"] == p],
                       lambda r: r["dataset"], ("all", *DATASETS))
    for name, s in summaries.items():
        rows = [r for i, r in expected.items() if name == "all" or tasks[i]["dataset"] == name]
        totals = {k: sum(r[k] for r in rows) for k in ("N", "F", "Q", "physical_Q", "correct", "failed", "truncated")}
        check_csv(s, dict(**totals, requests=len(rows), max_q=max(r["max_q"] for r in rows),
                          tpf=mean(r["N"]/r["F"] for r in rows), q_per_f=totals["Q"]/totals["F"],
                          token_efficiency=totals["N"]/totals["Q"]), p)
    return raws, expected


def analyse(private, project):
    book = project/"experiments/t2t/exchange"
    zero_book = book.parent/"hard"
    manifest_path = project/"benchmarks/frozen30/manifest.json"
    manifest = read(manifest_path)
    samples = project.parents[1]/"archive/fixed-baselines-20261124/samples.json"
    native = project.parents[1]/"results/fixed-baselines-llada-gpu4-v4/requests"
    require(sha(samples) == manifest["samples_sha256"], "Wrong frozen samples digest")
    tasks = unique(manifest["samples"], lambda r: r["index"], range(30))
    fixed = unique(read(samples)["samples"], lambda r: r["index"], tasks)
    require(Counter(t["dataset"] for t in tasks.values()) == dict.fromkeys(DATASETS, 10), "Wrong frozen datasets")
    prompts = {}
    for i, task in tasks.items():
        identity(fixed[i], task)
        with gzip.open(private/"llada21"/f"{i:02d}.trace.jsonl.gz", "rt") as f:
            identity(json.loads(next(f)), task)
            prompts[i] = json.loads(next(f))["prompt_length"]
        require(type(prompts[i]) is int and prompts[i] > 0, "Invalid native prompt length")
        ref = read(native/f"{i:02d}-native32.json")
        identity(ref, task)
        for folder in SOURCES.values():
            bridge = read(private/folder/f"{i:02d}-fullwidth.json")
            require(all(bridge[k] == ref[k] for k in BRIDGE), f"Native full-width bridge mismatch: {folder}, {i}")

    costs = csv_rows(book/"paired-cost.csv")
    paired = csv_rows(book/"placebo/exchange-placebo-paired-requests.csv")
    requests, groups, online = [], [], []
    for q in (4, 8):
        p, z = policy(q, True), policy(q, False)
        a, ar = load_arm(private/SOURCES["ranked"], book, tasks, q, True)
        b, br = load_arm(private/SOURCES["zero"], zero_book, tasks, q, False)
        cp = unique([r for r in paired if r["reference"] == "zero" and r["nominal_q"] == str(q)],
                    lambda r: int(r["index"]), tasks)
        rows, cost_total, phase_f, phase_q = [], Counter(), {}, {}
        for i, task in sorted(tasks.items()):
            for key, raw in ((p, a[i]), (z, b[i])):
                cost, swap, hit, hit_forward, fs, qs = scan(raw, q, prompts[i], ranked=key == p)
                phase_f.setdefault(key, Counter()).update(fs)
                phase_q.setdefault(key, Counter()).update(qs)
                if key == p:
                    rcost, first_swap, first_hit, first_hit_forward = cost, swap, hit, hit_forward
            swap_width = check_first_swap(a[i], b[i], first_swap)
            cost_total.update(rcost)
            at, bt = a[i]["tpf"], b[i]["tpf"]
            ac, bc = ar[i]["correct"], br[i]["correct"]
            require(first_swap is not None or ac == bc, f"No-swap score diverged: {q}, {i}")
            outcome = {(1, 1): "both", (1, 0): "win", (0, 1): "loss", (0, 0): "neither"}[ac, bc]
            check_csv(cp[i], dict(model="llada21", **{k: task[k] for k in IDENTITY[:-1]},
                                  ranked_correct=ac, reference_correct=bc, win=int(outcome == "win"),
                                  loss=int(outcome == "loss"), delta_correct=ac-bc, ranked_tpf=at,
                                  reference_tpf=bt, delta_tpf=at-bt,
                                  **{f"{side}_{k}": v for k in ("N", "F", "Q", "physical_Q")
                                     for side, v in (("ranked", ar[i][k]), ("reference", br[i][k]),
                                                     ("delta", ar[i][k]-br[i][k]))}), f"Paired {q}, {i}")
            rows.append(dict(q=q, **{k: task[k] for k in IDENTITY}, rank_correct=ac, zero_correct=bc,
                             outcome=outcome, rank_tpf=at, zero_tpf=bt, delta_tpf=at-bt,
                             hits=rcost["reserved_strict_hits"], reservations=rcost["reserved_seats"],
                             displaced_mask_rows=rcost["displaced_mask_rows"], first_swap_clock=first_swap,
                             first_swap_width=swap_width, first_hit_forward=first_hit_forward,
                             first_hit_clock=first_hit["hard_clock"] if first_hit else None,
                             first_hit_block=first_hit["block"] if first_hit else None,
                             first_hit_step_in_block=first_hit["step_in_block"] if first_hit else None,
                             hit_lag_from_first_swap=first_hit["hard_clock"]-first_swap if first_hit else None,
                             output_ids_identical=int(a[i]["output_ids"] == b[i]["output_ids"]),
                             **{f"{side}_{k}": result[i][k] for side, result in (("rank", ar), ("zero", br))
                                for k in ("N", "F", "Q", "truncated")}))
        totals = aggregate(rows)
        for key, source in ((p, book), (z, zero_book)):
            phases = unique([r for r in csv_rows(source/"llada21-online-phases.csv") if r["policy"] == key],
                            lambda r: r["phase"], phase_f[key])
            for stage, r in phases.items():
                check_csv(r, dict(F=phase_f[key][stage], Q=phase_q[key][stage]), f"{key}, {stage}")
        published, = [r for r in costs if r["policy"] == p]
        check_csv(published, dict(**cost_total, model="llada21", nominal_q=q, requests=30,
                                  correct=totals["rank_correct"], reference_correct=totals["zero_correct"],
                                  only_policy=totals["win"], only_reference=totals["loss"],
                                  tpf=totals["rank_tpf"], reference_tpf=totals["zero_tpf"],
                                  delta_tpf=totals["mean_delta_tpf"],
                                  **{k: sum(r[k] for r in ar.values()) for k in ("N", "F", "Q")}), p)
        online.append(dict(q=q, ranked_policy=p, zero_policy=z, ranked_cost=dict(cost_total),
                           **{side: {k: sum(r[k] for r in values.values()) for k in ("N", "F", "Q")}
                              for side, values in (("ranked", ar), ("zero", br))},
                           ranked_mask="".join(str(ar[i]["correct"]) for i in sorted(tasks)),
                           zero_mask="".join(str(br[i]["correct"]) for i in sorted(tasks)),
                           no_swap_requests=sum(r["first_swap_clock"] is None for r in rows),
                           same_output_ids=sum(r["output_ids_identical"] for r in rows)))
        groups.extend(strata(rows))
        requests.extend(rows)
    meta = dict(sources=SOURCES, manifest_sha256=sha(manifest_path), samples_sha256=manifest["samples_sha256"],
                exchange_rates_sha256=sha(book/"rates.json"), online=online,
                interpretation="Post-treatment strata on observed ranked trajectories; not estimates of suppressing writes.",
                timing="1-based hard clock includes prefill; generation forward excludes prefill; block is absolute.",
                scoring="Archived correctness masks checked against both published request tables and paired rows; not rescored.",
                width="All per-layer/event widths and phases checked; first physical swap checked at shared state only.",
                retention="Not identifiable from logged hit position, aggregate later edits and final tokens alone.")
    return requests, groups, meta


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--private", type=Path, default=ROOT.parent/"t2t-native-1125",
                   help="Private runtime parent of original archives and the new output directory")
    args = p.parse_args()
    private = args.private.resolve(strict=True)
    out = private/"exchange-veto-historical"
    requests, groups, meta = analyse(private, ROOT)
    out.mkdir(mode=0o700)
    write_csv(out/"requests.csv", requests)
    write_csv(out/"summary.csv", groups)
    with (out/"validation.json").open("x", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"Validated and wrote {len(requests)} requests and {len(groups)} summary rows to {out}")


if __name__ == "__main__":
    main()
