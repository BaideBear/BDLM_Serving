"""Score both complete Gemma wave references and K0 Q256 on the same frozen90."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from bdlm_serving.evaluation.frozen import ROOT, sha
from experiments.t2t.prefix_window.score_run import link_requests, snapshot
from tools.prefix_cohorts import load_cases, load_request_set, select_window_cases, _window_row, score_files


def read(path):
    return json.loads(path.read_text())


def wave_rows(data, original, cases, policy, cache):
    run = data/"gemma-wave-reference90"
    meta = read(run/"run.json")
    if (meta["suite"],meta["block_size"],meta["samples_sha256"],meta["policies"],
            meta["flowblock_spawn_threshold"],meta["seed_offset"]) != (
            "frozen90",256,sha(data/"samples.json"),["flow_w1","flow_w2"],0.8,0):
        raise ValueError("Invalid wave metadata")
    if read(run/f"complete-{policy}.json") != dict(policy=policy,requests=90):
        raise ValueError("Incomplete wave arm")
    old = [original/f"{i:02d}-{policy}.json" for i in range(30)]
    if read(run/f"reproduced30-{policy}.json") != dict(policy=policy, matched=30,
            reference_sha256=[sha(path) for path in old]):
        raise ValueError("Original30 references changed")
    with (ROOT/"archive/diffusiongemma-wave/requests.csv").open() as f:
        public = {int(r["index"]):r for r in csv.DictReader(f) if r["policy"] == policy}
    folder = link_requests(run, policy, cache)
    rows,m,_,_ = load_request_set(folder,cases,"wave",policy)
    if m != "gemma" or set(public) != set(range(30)):
        raise ValueError("Wrong archived model or incomplete original30")
    active = {}
    for r in rows:
        i = r["index"]
        raw = read(folder/f"{i:02d}-{policy}.json")
        t = raw["trace"]
        active[i] = (sum(s["active"] for s in t),len(t))
        if i < 30:
            h = public[i]
            if (raw != read(old[i]) or r["task_id"] != h["task_id"] or
                    (r["correct"],r["N"],r["F"],r["physical_Q"],*active[i]) != tuple(
                    int(h[k]) for k in ("correct","output_tokens","generation_forwards",
                                         "generation_physical_padded_rows","active_slot_sum","decoder_forwards"))):
                raise ValueError(f"Archived row or score mismatch: {policy}/{i}")
    return {r["index"]:r for r in rows}, active


def report(data, original, out):
    cache = out/".inputs"
    cases = load_cases(data/"samples.json")
    labels = ["flow_w1","flow_w2","t2t_q256_f0_m256_hard_prefix_zero"]
    groups = {}
    active = {}
    for p in labels[:2]:
        groups[p],active[p] = wave_rows(data, original,cases,p,cache)
    p = labels[-1]
    dest = snapshot(data/"gemma-grid-k1",p,sha(data/"samples.json"),cache)
    rs,m,_,q = load_request_set(dest,cases,"K0",p)
    if (m,q) != ("gemma",256):
        raise ValueError("Wrong fixed Q256 reference")
    groups[p] = {r["index"]:r for r in rs}
    summary = []
    for window in (30,60,90):
        subset = select_window_cases(cases,window)
        for base,arm in ((labels[0],labels[1]), (labels[0],p),(labels[1],p)):
            row = _window_row(subset,groups[base],groups[arm],window,"all","gemma",base,arm,None)
            for side, label in (("baseline",base),("arm",arm)):
                row[f"{side}_mean_active_blocks_per_decoder"] = ""
                if label in active:
                    counts = [active[label][r["index"]] for r in subset]
                    row[f"{side}_mean_active_blocks_per_decoder"] = sum(a for a,_ in counts)/sum(f for _,f in counts)
            summary.append(row)
            if window == 90:
                print(base,arm,*(row[k] for k in ("baseline_correct","arm_correct","baseline_macro_tpf",
                      "arm_macro_tpf","baseline_pooled_token_efficiency","arm_pooled_token_efficiency")))
    with (out/"gemma-wave-vs-fixed.csv").open("w",newline="") as f:
        w = csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator="\n")
        w.writeheader()
        w.writerows(summary)


def active_report(data, out):
    """Same mask on both arms. Qualify the full-width control before scoring compact W2."""
    from experiments.t2t.serving_slots.mask_pair_gemma import active_settings, require_complete
    samples = data/"samples.json"
    ref, run = data/"gemma-wave-reference90", data/"gemma-fixed512-active-w2"
    control, compact = "flow_w2", "active_w2"
    base = read(ref/"run.json")
    meta = read(run/"run.json")
    if (meta != active_settings(base) or base["suite"] != "frozen90" or
            base["samples_sha256"] != sha(samples) or base["flowblock_spawn_threshold"] != 0.8):
        raise ValueError("Not the registered compact W2 control")
    require_complete(run, [control, compact])
    require_complete(ref, [control])
    for i in range(90):
        name = f"{i:02d}-{control}.json"
        if read(run/name) != read(ref/name):
            raise ValueError(f"Changed full-width W2 control: {i}")
    return score_files(link_requests(run, control, out/".inputs"),
                       link_requests(run, compact, out/".inputs"),
                       samples, out/"gemma-fixed512-active-vs-flow.csv",
                       baseline_policy=control, arm_policy=compact, nominal_q=512)


def tick_cost(event, b):
    """Paid work anchored to a decoder call; later encoder work is separate."""
    debt = sum(map(len, event["dirty_before"]))
    tail = len(event["dirty_before"][1]) if event["active"] == 2 else 0
    overflow = max(0, event["q"] - event["offer"])
    if overflow and not (event["refresh_only"] == 1 and tail and overflow == b):
        raise ValueError("Unexplained whole-canvas overrun")
    return Counter(ticks=1, offered_Q=event["offer"], decoder_Q=event["q"],
                   overrun_Q=overflow, overrun_ticks=int(overflow > 0),
                   mandatory_rows=debt, mandatory_tail_rows=tail,
                   tail_refresh_only_Q=b*event["refresh_only"],
                   tail_rows_needed_on_refresh_only=tail*event["refresh_only"],
                   underfill_Q=max(0, event["offer"]-event["q"]),
                   encoder_F=len(event["encoded"]), encoder_Q=b*len(event["encoded"]),
                   new_writes=sum(map(len, event["admitted"])),
                   t2t_writes=sum(map(len, event["strict_edits"])),
                   promotions=len(event["promoted"]))


def transition_costs(trace, b):
    """Lagged windows overlap; never add lag rows to recover total work."""
    for tick, e in enumerate(trace):
        a, now = trace[tick-1]["offer"] if tick else 0, e["offer"]
        for lag in range(3):
            if tick+lag < len(trace):
                yield (a, now, lag), tick_cost(trace[tick+lag], b)


def external_report(data, out):
    from bdlm_serving.methods.t2t.wave_offer import TRACES
    from experiments.t2t.serving_slots.expanded_handoff import SAMPLES
    from experiments.t2t.serving_slots.mask_pair_gemma import (
        active_settings, require_complete, require_switch_artifacts,
    )
    from experiments.t2t.serving_slots.mask_pair_report import efficiency_interval
    from tools.prefix_cohorts import normalize_request

    samples = data/"samples.json"
    cases = load_cases(samples)
    ref, active, run = (data/n for n in (
        "gemma-wave-reference90", "gemma-fixed512-active-w2", "gemma-external-256-512"))
    base = read(ref/"run.json")
    if (sha(samples) != SAMPLES or base["samples_sha256"] != SAMPLES
            or (base["suite"], base["block_size"], base["cap"], base["flowblock_spawn_threshold"])
            != ("frozen90", 256, 2048, 0.8)):
        raise ValueError("Wrong reference, cohort or runtime settings")
    require_complete(ref, ("flow_w1", "flow_w2"))
    require_switch_artifacts(data, active_settings(base))
    flat = {"wave_bridge_256": (ref, "flow_w1"), "wave_bridge_512": (active, "active_w2")}
    tables = {p: {} for p in TRACES}
    totals, transitions = defaultdict(Counter), defaultdict(Counter)
    case_counts = Counter()
    for c in cases:
        i = c["index"]
        cohorts = ("all90", "original30" if i < 30 else "added60")
        for p in TRACES:
            raw = read(run/f"{i:02d}-{p}.json")
            if raw["policy"] != p or raw["model"] != "diffusiongemma":
                raise ValueError(f"Wrong arm or model in {i}:{p}")
            if p in flat:
                src, label = flat[p]
                core = [{k: e[k] for k in ("width","active","q","head")} for e in raw["trace"]]
                if dict(raw, trace=core, policy=label) != read(src/f"{i:02d}-{label}.json"):
                    raise ValueError(f"Flat bridge differs from reference at {i}:{p}")
            tables[p][i] = row = normalize_request(raw, c, p)
            path = raw["trace"]
            work = Counter()
            for e in path:
                work.update(tick_cost(e, 256))
            if (work["ticks"]+work["encoder_F"] != row["F"]
                    or work["decoder_Q"]+work["encoder_Q"] != row["physical_Q"]):
                raise ValueError("Step and request ledgers differ")
            for group in cohorts:
                totals[group,p].update(work)
                totals[group,p]["requests"] += 1
            seen = set()
            for key, x in transition_costs(path, 256):
                for group in cohorts:
                    tag = (group,p,*key)
                    transitions[tag].update(x)
                    seen.add(tag)
            case_counts.update(seen)
    pairs = []
    for w in (30,60,90):
        subset = select_window_cases(cases,w)
        labels = list(TRACES)
        for j, base_label in enumerate(labels):
            for arm in labels[j+1:]:
                r = _window_row(subset, tables[base_label], tables[arm], w, "all", "gemma",
                                base_label, arm, None)
                r["efficiency_ci_low"], r["efficiency_ci_high"] = efficiency_interval(
                    subset,tables[base_label],tables[arm])
                pairs.append(r)
    out.mkdir(parents=True,exist_ok=True)
    for name, rows in (
        ("gemma-external-256-512-pairs", pairs),
        ("gemma-external-256-512-work", [dict(cohort=k,policy=p,**v)
                                        for (k,p),v in sorted(totals.items())]),
        ("gemma-external-256-512-transitions", [dict(cohort=k,policy=p,from_q=a,to_q=z,lag=lag,
             requests=case_counts[k,p,a,z,lag],**v) for (k,p,a,z,lag),v in sorted(transitions.items())]),
    ):
        with (out/(name+".csv")).open("w",newline="") as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ("data","original","out"):
        parser.add_argument("--"+k, type=Path,required=k!="original")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--active", action="store_true", help="Compare Q512 active-only vs padded W2")
    modes.add_argument("--external", action="store_true", help="Qualify 256/512 bridges and compare external offers")
    args = parser.parse_args()
    if args.external:
        external_report(args.data, args.out)
    elif args.active:
        active_report(args.data, args.out)
    elif args.original:
        report(args.data, args.original, args.out)
    else:
        parser.error("Without --active, provide --original for original30 control qualification")
