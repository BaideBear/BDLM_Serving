"""Audit and score frozen90 single-offer mask interventions (CPU only).

Input: INDEX-POLICY.json work rows, with unified90 `trace` (Gemma) or
llada_all_slots90 `events` (LLaDA). On both arms the row-level witness is
`offer_mask_witness[str(target)] = {first_tick, pretrace_sha256,
geometry_sha256, rounds, edge_pairs}`. `first_tick` uses the generation offer
clock; geometry is null (or its SHA256) if unexposed, and pretrace hashes the
*full* trajectory if unexposed (otherwise the strict prefix, including any
prompt forward).
Witness counts describe eligible edges in each arm's own trajectory, not
actual openings in the causal control. They need not match after first touch.

Output: aggregate-only JSON. Work is for the entire request, not just rounds
at the target offer. Deltas are unmasked minus causal; exposed-only results
are descriptive, not a substitute for the full paired cohort.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from statistics import mean


COHORTS = {"full90": (0, 90), "original30": (0, 30), "new60": (30, 90)}
DATASETS = ("gsm8k", "math500", "humaneval")
IDENTITY = ("index", "task_id", "dataset", "seed")
WORK = ("output_tokens", "generation_forwards", "generation_physical_padded_rows")
DIGEST = re.compile(r"[0-9a-f]{64}")
NULL_DIGEST = hashlib.sha256(b"null").hexdigest()
ARM = re.compile(r"[A-Za-z0-9_-]+")
MODELS = {"gemma": "diffusiongemma", "diffusiongemma": "diffusiongemma",
          "llada": "llada21", "llada21": "llada21"}
REPLAYS = {"diffusiongemma": (512, 512, 128, 32, 64, 256, 512, 32),
           "llada21": (4, 8, 16, 32, 64)}


@dataclass(frozen=True)
class Pair:
    target: int
    control: str
    trial: str


@dataclass(frozen=True)
class Fields:
    exposure: str
    opened: str | None = None
    witness: str = "offer_mask_witness"
    ignore_events: tuple[str, ...] = ()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def is_digest(value):
    return isinstance(value, str) and DIGEST.fullmatch(value) is not None


def event_value(event, path, *, null_parent=False):
    value = event
    for name in path.split("."):
        # E.g., mask_window=None has no eligible edges.
        if null_parent and value is None:
            return None
        require(isinstance(value, dict) and name in value, f"Missing event field {path}")
        value = value[name]
    return value


def ledger(row, model):
    n, f, q = (row[k] for k in WORK)
    require(integer(n) and integer(f, 1) and integer(q, f), "Invalid work totals")
    for field, expected in (("tpf", n/f), ("token_efficiency", n/q)):
        if field in row:
            value = row[field]
            require(type(value) in (int, float) and value == expected, f"Inconsistent {field}")
    if model == "diffusiongemma":
        events = gen = row["trace"]
        offer_key, tick_key, width = "offer", "tick", "q"
        require(is_digest(row["input_sha256"]) and integer(row["prefill_q"]),
                "Invalid Gemma input witness")
        calls = [e["encoded"] for e in gen]
        require(all(isinstance(c, list) for c in calls), "Invalid Gemma encoder ledger")
        reencodes = sum(map(len, calls))
        require(integer(row["decoder_q"]) and integer(row["encoder_q"])
                and row["decoder_q"] == sum(e[width] for e in gen)
                and row["encoder_q"] == 256 * reencodes and f == len(gen) + reencodes
                and q == row["decoder_q"] + row["encoder_q"], "Gemma work does not reconstruct")
    else:
        events = row["events"]
        offer_key, tick_key, width = "q_budget", "offer_tick", "physical_q"
        prompt = row["prompt_ids"]
        require(isinstance(prompt, list) and all(integer(t) for t in prompt),
                "Invalid LLaDA prompt IDs")
        prefill = sum(e[width] for e in events if e["phase"] == "prefill")
        offset = int(prefill > 0)
        gen = events[offset:]
        require(row["prefill_rows"] == prefill == len(prompt)//32*32
                and all(e["phase"] == "prefill" for e in events[:offset])
                and all(e["phase"] != "prefill" for e in gen), "Invalid LLaDA prefill")
        require(all(integer(e[width], 1) and integer(e["F"], 1) and e["F"] == 1
                    and e["actual_q"] == e[width] for e in events)
                and f == len(gen) and q == sum(e[width] for e in gen),
                "LLaDA work does not reconstruct")
        if "generation_query_rows" in row:
            require(row["generation_query_rows"] == q, "Inconsistent LLaDA query rows")
    require(isinstance(events, list) and gen, "Missing generation trace")
    require(all(integer(e[tick_key]) and e[tick_key] == tick
                and integer(e[offer_key], 1) and integer(e[width], 1)
                for tick, e in enumerate(gen)), "Invalid offer clock or physical width")
    return events, gen, offer_key


def witness(row, gen, offer_key, target, fields, *, trial):
    record = row[fields.witness][str(target)]
    first, pretrace, geometry, rounds, edges = (
        record[k] for k in ("first_tick", "pretrace_sha256", "geometry_sha256",
                            "rounds", "edge_pairs"))
    if first is None and pretrace is None and row.get("model") == "llada21":
        pretrace = record.get("trajectory_sha256")
    require(is_digest(pretrace) and integer(rounds) and integer(edges, rounds),
            "Malformed mask witness")
    require((geometry in (None, NULL_DIGEST) and rounds == edges == 0) if first is None else
            (integer(first) and first < len(gen) and is_digest(geometry)
             and geometry != NULL_DIGEST and rounds > 0),
            "Inconsistent exposure witness")
    eligible, edge_counts = [], []
    for tick, event in enumerate(gen):
        raw = event_value(event, fields.exposure, null_parent=True)
        require(raw is None or type(raw) is bool or integer(raw), "Invalid exposure value")
        on_target = event[offer_key] == target
        if on_target and raw:
            eligible.append(tick)
            edge_counts.append(raw)
        if fields.opened:
            opened = event_value(event, fields.opened)
            should_open = trial and on_target and bool(raw)
            require(integer(opened) and bool(opened) == should_open
                    and (not should_open or type(raw) is not int or opened == raw),
                    "Actual mask opening disagrees with arm or target offer")
    require(first == (eligible[0] if eligible else None) and rounds == len(eligible),
            "Witness first tick or rounds disagree with target exposure trace")
    if all(type(v) is int for v in edge_counts):
        require(edges == sum(edge_counts), "Witness edge_pairs disagree with eligible edges")
    if fields.opened and trial:
        require(edges == sum(event_value(e, fields.opened) for e in gen),
                "Witness edge_pairs disagree with actual openings")
    return first, pretrace, geometry, rounds, edges


def audit_pair(a, b, task, model, pair, fields, offers=None):
    """Validate a control/trial pair before trusting either prediction or work."""
    for arm, row in ((pair.control, a), (pair.trial, b)):
        require(row["policy"] == arm and row["model"] == model and row["status"] == "success",
                "Wrong policy, model or unsuccessful request")
        require(all(type(row[k]) is type(task[k]) and row[k] == task[k] for k in IDENTITY),
                "Wrong frozen index, task ID, dataset or seed")
        require(isinstance(row["output_text"], str) and isinstance(row["output_ids"], list)
                and all(integer(token) for token in row["output_ids"]),
                "Missing or invalid output")
        require(integer(row["output_tokens"]) and row["output_tokens"] <= len(row["output_ids"]),
                "Invalid counted output tokens")
    input_fields = (("input_sha256", "prefill_q") if model == "diffusiongemma"
                    else ("prompt_ids", "prefill_rows", "special_ids", "eos_id"))
    require(all(a[k] == b[k] for k in input_fields), "Different paired input")
    seqs, witnesses, ticks = [], [], []
    for trial, row in enumerate((a, b)):
        events, gen, offer_key = ledger(row, model)
        seq = [e[offer_key] for e in gen]
        require(offers is None or all(v == offers[i % len(offers)] for i, v in enumerate(seq)),
                "Trace differs from the registered offer schedule")
        seqs.append(seq)
        w = witness(row, gen, offer_key, pair.target, fields, trial=bool(trial))
        witnesses.append(w)
        ignored = set(fields.ignore_events)
        if fields.opened and "." not in fields.opened:
            ignored.add(fields.opened)
        protected = {offer_key, "tick", "offer_tick", "phase", "q", "encoded", "F",
                     "physical_q", "actual_q", fields.exposure.split(".")[0]}
        require(not (ignored & protected), "Cannot ignore clock, work or exposure fields")
        ticks.append([{k: v for k, v in e.items() if k not in ignored} for e in events])
    shared = min(map(len, seqs))
    require(seqs[0][:shared] == seqs[1][:shared], "Paired offers differ")
    wa, wb = witnesses
    require(wa[:3] == wb[:3], "Different first tick, pretrace or first geometry digest")
    first = wa[0]
    if first is None:
        require(ticks[0] == ticks[1], "Unexposed trace changed")
        for field in (*WORK, "output_ids", "output_text", "truncated"):
            require(a.get(field) == b.get(field), f"Unexposed {field} changed")
    else:
        # first_tick excludes prompt-only work; compare it as part of the prefix.
        stop = first + len(ticks[0]) - len(seqs[0])
        require(ticks[0][:stop] == ticks[1][:stop], "Trace changed before first intervention")
        keys = (("head", "q", "phase", "mandatory_q", "prefix", "parked")
                if model == "diffusiongemma" else
                ("block", "stage", "phase", "top_action", "absolute_positions",
                 "sampled_positions", "refresh_only_positions", "visible_kv", "physical_q"))
        require(all(ticks[0][stop].get(k) == ticks[1][stop].get(k) for k in keys),
                "First intervention geometry changed")
    return wa, wb


def arm_metrics(values):
    n, f, q = (sum(v[i] for v in values) for i in (1, 2, 3))
    return dict(correct=sum(v[0] for v in values), quality=mean(v[0] for v in values),
                macro_TPF=mean(v[1]/v[2] for v in values), micro_efficiency=n/q, N=n, F=f, Q=q,
                trigger_requests=sum(v[4] > 0 for v in values),
                trigger_rounds=sum(v[4] for v in values), edge_pairs=sum(v[5] for v in values))


def paired_ci(cases, draws, seed):
    """Stratified percentile bootstrap, using shared request draws for both arms."""
    by_dataset = defaultdict(list)
    for dataset, a, b in cases:
        by_dataset[dataset].append((b[0]-a[0], b[1]/b[2]-a[1]/a[2], a[1], a[3],
                                    b[1], b[3], b[2]-a[2], b[3]-a[3]))
    strata = list(by_dataset.values())
    rng = random.Random(seed)
    keys = ("quality", "macro_TPF", "micro_efficiency", "F", "Q")
    samples = [[] for _ in keys]
    size = len(cases)
    for _ in range(draws):
        selected = [v for items in strata for v in rng.choices(items, k=len(items))]
        quality, tpf, na, qa, nb, qb, f, q = map(sum, zip(*selected))
        for sample, value in zip(samples, (quality/size, tpf/size, nb/qb-na/qa, f, q)):
            sample.append(value)

    def quantile(values, p):
        index = (len(values)-1)*p
        lo = int(index)
        return values[lo] + (index-lo)*(values[min(lo+1, len(values)-1)]-values[lo])

    for sample in samples:
        sample.sort()
    return {k: [quantile(s, .025), quantile(s, .975)] for k, s in zip(keys, samples)}


def summarize(verified, *, draws=4000, seed=17000):
    require(integer(draws, 2), "Need at least two bootstrap draws")
    result = []
    for cohort, (lo, hi) in COHORTS.items():
        for scope in ("all", "exposed", "unexposed"):
            cases = [(dataset, a, b) for i, dataset, a, b in verified if lo <= i < hi
                     and (scope == "all" or (b[4] > 0) == (scope == "exposed"))]
            row = dict(cohort=cohort, scope=scope, requests=len(cases))
            if not cases:
                result.append(dict(row, control=None, trial=None, delta=None, paired_ci95=None,
                                   rescued=0, lost=0))
                continue
            control = arm_metrics([a for _, a, _ in cases])
            trial = arm_metrics([b for _, _, b in cases])
            from tools.prefix_cohorts import quality_ci
            paired_counts = defaultdict(lambda: [0,0,0,0])
            for dataset,a,b in cases:
                paired_counts[dataset][(0 if a[0] else 2)+(0 if b[0] else 1)] += 1
            _,_,lo95,hi95,fallback = quality_ci(paired_counts)
            result.append(dict(
                row, control=control, trial=trial, delta={k: trial[k]-control[k] for k in control},
                paired_quality_score_ci95=[lo95,hi95], quality_fallback_strata=fallback,
                rescued=sum(b[0] and not a[0] for _, a, b in cases),
                lost=sum(a[0] and not b[0] for _, a, b in cases),
                paired_ci95=paired_ci(cases, draws, seed)))
    return result


def default_fields(model):
    if model == "diffusiongemma":
        return Fields("opportunity")
    return Fields("mask_window.opened_edges", "opened_edges")


def offer_schedule(model, control):
    if control == "replay":
        return REPLAYS[model]
    fixed = re.fullmatch(r"(?:q|fixed)([1-9][0-9]*)", control)
    alt = re.fullmatch(r"alternate(32|64|128|256)",control)
    if model=="diffusiongemma" and alt:
        return (512,int(alt[1]))
    return (int(fixed[1]),) if fixed else None


def read_registration(group):
    path = group / "run.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def discover_pairs(group, metadata=None):
    found = set()
    if metadata and metadata.get("model") == "llada21" and metadata.get("kernel") == "SpawnGatedKernel":
        return [Pair(q, "causal", f"open{q}") for q in metadata["targets"]]
    registered = (metadata or {}).get("unmask_only_on_offer")
    policies = registered if isinstance(registered, dict) else (
        path.stem.partition("-")[2] for path in group.glob("*-*_unmask*.json"))
    for policy in policies:
        m = re.fullmatch(r"([A-Za-z0-9_-]+)_unmask([1-9][0-9]*)", policy)
        if m:
            control, target = m.groups()
            found.add(Pair(int(target), control, policy))
    return sorted(found, key=lambda p: (p.control, p.target))


def check_registration(metadata, pairs, samples_digest):
    """Use the worker receipt when present; coverage is checked from rows."""
    if "samples_sha256" in metadata:
        require(metadata["samples_sha256"] == samples_digest, "Run registered different samples")
    if "indices" in metadata:
        require(metadata["indices"] == list(range(90)), "Run is not registered for full90")
    if "unmask_only_on_offer" in metadata:
        targets = metadata["unmask_only_on_offer"]
        require(isinstance(targets, dict)
                and all(p.control in targets and targets[p.control] is None
                        and type(targets.get(p.trial)) is int and targets[p.trial] == p.target
                        for p in pairs), "Pair disagrees with registered mask targets")
    if metadata.get("model") == "llada21" and metadata.get("kernel") == "SpawnGatedKernel":
        arms, trace = metadata["arms"], metadata["trace"]
        require(all(p.target in metadata["targets"] and p.target in trace
                    and arms[p.control]["target"] is None and arms[p.trial]["target"] == p.target
                    and arms[p.control]["trace"] == arms[p.trial]["trace"] == trace for p in pairs),
                "Pair disagrees with registered LLaDA mask targets")


def score_group(group, model, pairs, tasks, judge, *, fields=None, offers=None,
                draws=4000, seed=17000):
    """Audit every requested pair before scoring; reuse judgments in shared arms."""
    group = Path(group)
    model = MODELS[model]
    fields = fields or default_fields(model)
    tasks = sorted(tasks, key=lambda r: r["index"])
    require([r["index"] for r in tasks] == list(range(90))
            and len({(r["dataset"], r["task_id"]) for r in tasks}) == 90
            and all(Counter(r["dataset"] for r in tasks[lo:hi])
                    == dict.fromkeys(DATASETS, (hi-lo)//3) for lo, hi in COHORTS.values()),
            "Expected balanced frozen90, original30 and new60")
    require(pairs and len(set(pairs)) == len(pairs), "Need distinct target/arm pairs")
    require(integer(draws, 2), "Need at least two bootstrap draws")
    arms = set()
    for p in pairs:
        require(integer(p.target, 1) and p.control != p.trial
                and all(ARM.fullmatch(arm) for arm in (p.control, p.trial)), "Invalid pair")
        arms.update((p.control, p.trial))
    paths = {arm: {} for arm in arms}
    for path in group.glob("*.json"):
        match = re.fullmatch(r"([0-9]+)-(.+)\.json", path.name)
        if match and match[2] in paths:
            entries = paths[match[2]]
            index = int(match[1])
            require(index not in entries, f"Duplicate work row for {match[2]} index {index}")
            entries[index] = path
    for arm, entries in paths.items():
        require(set(entries) == set(range(90)), f"Expected exactly 90 frozen work rows for {arm}")

    plans = []
    for p in pairs:
        schedule = offers or offer_schedule(model, p.control)
        require(schedule and all(integer(v, 1) for v in schedule) and p.target in schedule,
                f"No registered schedule containing target {p.target} for {p.control}; use --offers")
        plans.append(schedule)
    # Retain predictions and reconciled scalars, not potentially large raw ledgers.
    pending = [[] for _ in pairs]
    texts = {}
    for i, task in enumerate(tasks):
        rows = {}
        for arm in sorted(arms):
            try:
                rows[arm] = json.loads(paths[arm][i].read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON for {arm} index {i} (line {e.lineno})") from None
        for p, schedule, bucket in zip(pairs, plans, pending):
            a, b = rows[p.control], rows[p.trial]
            try:
                w = audit_pair(a, b, task, model, p, fields, schedule)
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                raise ValueError(f"target {p.target} {p.control}/{p.trial} index {i}: {e}") from None
            bucket.append(tuple((*[r[k] for k in WORK], *v[3:]) for r, v in zip((a, b), w)))
        for arm, row in rows.items():
            texts[arm, i] = row["output_text"]
    judgments = {}
    for (arm, i), text in texts.items():
        try:
            judgments[arm, i] = bool(judge(text, tasks[i])["correct"])
        except Exception:
            # Do not emit judge exceptions that may contain private answer text.
            raise RuntimeError(f"Quality judge failed for {arm} index {i}") from None
    return [
        dict(model=model, target=p.target, control_policy=p.control, trial_policy=p.trial,
             offers=list(schedule), statistics=summarize([
                 (i, task["dataset"], (judgments[p.control, i], *a), (judgments[p.trial, i], *b))
                 for i, (task, (a, b)) in enumerate(zip(tasks, bucket))], draws=draws, seed=seed))
        for p, schedule, bucket in zip(pairs, plans, pending)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--group", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path, help="registered frozen90 samples.json")
    parser.add_argument("--model", choices=MODELS, default="gemma")
    parser.add_argument("--pair", nargs=3, action="append", metavar=("TARGET", "CONTROL", "TRIAL"),
                        help="repeatable; defaults to discovering CONTROL_unmaskTARGET arms")
    parser.add_argument("--offers", nargs="+", type=int,
                        help="override the repeating offer schedule for every selected pair")
    parser.add_argument("--exposure-field", help="event bool or eligible-edge count (dotted path)")
    parser.add_argument("--opened-field", help="top-level actual opened edge count; 'none' disables")
    parser.add_argument("--witness-field", default="offer_mask_witness")
    parser.add_argument("--ignore-event-field", action="append", default=[],
                        help="repeatable top-level arm metadata to omit from comparisons")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=17000)
    parser.add_argument("--out", type=Path, help="new JSON file; default stdout")
    args = parser.parse_args(argv)
    try:
        require(integer(args.draws, 2), "--draws must be at least two")
        require(args.opened_field is None or "." not in args.opened_field,
                "--opened-field must be a top-level event key")
        require(args.group.is_dir(), "--group must be a directory")
        from bdlm_serving.evaluation.frozen90 import load_tasks
        from bdlm_serving.evaluation.scoring import score
        defaults = default_fields(MODELS[args.model])
        fields = Fields(args.exposure_field or defaults.exposure,
                        None if args.opened_field == "none" else args.opened_field or defaults.opened,
                        args.witness_field, tuple(args.ignore_event_field))
        metadata = read_registration(args.group)
        pairs = ([Pair(int(t), c, b) for t, c, b in args.pair] if args.pair
                 else discover_pairs(args.group, metadata))
        samples_digest = hashlib.sha256(args.samples.read_bytes()).hexdigest()
        check_registration(metadata, pairs, samples_digest)
        registered_trace = metadata.get("trace") if metadata.get("kernel") == "SpawnGatedKernel" else None
        require(not registered_trace or not args.offers or registered_trace == args.offers,
                "Offer override differs from registered trace")
        results = score_group(args.group, args.model, pairs, load_tasks(args.samples), score,
                              fields=fields, offers=args.offers or registered_trace, draws=args.draws, seed=args.seed)
        payload = json.dumps(dict(
            schema_version=1, direction="trial minus control",
            uncertainty=dict(method="dataset-stratified paired percentile bootstrap",
                             confidence=.95, draws=args.draws, seed=args.seed,
                             simultaneous=False),
            samples_sha256=samples_digest,
            exposure_field=fields.exposure, opened_field=fields.opened,
            witness_field=fields.witness, ignored_event_fields=fields.ignore_events,
            pairs=results), indent=2, allow_nan=False)
        if args.out:
            with args.out.open("x", encoding="utf-8") as out:
                out.write(payload + "\n")
        else:
            print(payload)
    except (OSError, ValueError, RuntimeError) as e:
        parser.exit(2, f"offer_mask_score: {e}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
