"""Read-only, fixed-arm paired report for cross_race and layout worker receipts.

Pass receipt files or directories of receipt JSON files. No model or scorer is
loaded. Reports go to stdout; partial coverage is not a full frozen90 result.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from statistics import mean


MANIFEST = Path(__file__).resolve().parents[3] / "benchmarks/frozen90/manifest.json"
COHORTS = {"original30": range(30), "new60": range(30, 90), "frozen90": range(90)}
KINDS = ("cross", "layout_sweep", "layout_llada")
CATEGORIES = ("rebuild", "verify", "mandatory", "prefix", "unresolved", "clean-other")
TAIL = ("tail_open", "tail_discard", "tail_touched_positions", "tail_last_candidate",
        "tail_final_matches", "discarded_tail_Q", "tail_Q")
SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, order=True)
class Arm:
    kind: str
    budget: int
    name: str


@dataclass(frozen=True)
class Trial:
    arm: Arm
    identity: tuple
    input_sha: str | None
    scorer_sha: str | None
    correct: bool | None
    truncated: bool
    work: dict

    @property
    def index(self):
        return self.identity[1]


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _nat(value, name, minimum=0):
    _check(type(value) is int and value >= minimum, f"invalid {name}")
    return value


def _sha(value, name):
    _check(isinstance(value, str) and SHA256.fullmatch(value), f"invalid {name}")
    return value


def _identity(row, snapshot):
    index = _nat(row["index"], "index")
    _check(index < 90, "index outside frozen90")
    dataset, task = row["dataset"], row["task_id"]
    _check(dataset in ("gsm8k", "math500", "humaneval") and isinstance(task, str) and task,
           "invalid dataset/task_id")
    return snapshot, index, dataset, task, _nat(row["seed"], "seed")


def _cross(budget, gate, slack=False, den=4, prefix=(1, 8)):
    return Arm("cross", budget,
               f"C{budget}-gate{gate}-slack{int(slack)}-{den}-pref{prefix[0]}of{prefix[1]}")


def _sweep(budget, layout, cold):
    return Arm("layout_sweep", budget, f"{layout}-cold{cold}-C{budget}")


def _llada(budget, cold, den):
    return Arm("layout_llada", budget, f"cold{cold}-pref1of{den}-C{budget}")


def _comparisons():
    for gate in ("eager", "half", "quarter"):
        yield _cross(512, "off"), _cross(512, gate)
    for c in (64, 128):
        base = _cross(c, "off")
        for prefix in ((0, 8), (1, 4), (1, 2)):
            yield base, _cross(c, "off", prefix=prefix)
        for den in (4, 2, 8):
            yield base, _cross(c, "off", True, den)
    for c in (32, 64, 128):
        for cold in (50, 100):
            for layout in ("rect", "halo"):
                yield _sweep(c, "split", cold), _sweep(c, layout, cold)
        for layout in ("split", "rect", "halo"):
            for cold in ((100,) if layout == "split" else (0, 100)):
                yield _sweep(c, layout, 50), _sweep(c, layout, cold)
    for c in (4, 8, 16):
        for cold, den in ((0, 8), (50, 4), (100, 8), (0, 4), (50, 16)):
            yield _llada(c, 50, 8), _llada(c, cold, den)


COMPARISONS = tuple(_comparisons())
ARMS = {arm.name: arm for pair in COMPARISONS for arm in pair}


def _cross_work(row, c, history):
    d_f, e_f, d_q, e_q, peak, f, joint = (
        _nat(row[k], k) for k in ("decoder_F", "encoder_F", "decoder_Q", "encoder_Q",
                                   "max_decoder_Q", "F", "joint_F"))
    _check(d_f > 0 and 0 < peak <= c and peak + d_f - 1 <= d_q <= peak * d_f
           and f == d_f + e_f and e_q == 256 * e_f and joint <= d_f and history == d_q,
           "cross decoder cap or accounting mismatch")
    if "Q" in row:
        _check(_nat(row["Q"], "Q") == d_q + e_q, "cross physical Q mismatch")
    categories = row["categories"]
    _check(isinstance(categories, dict) and set(categories) == set(CATEGORIES),
           "invalid decoder categories")
    categories = {k: _nat(categories[k], k) for k in CATEGORIES}
    _check(sum(categories.values()) == d_q, "decoder categories do not reconcile")
    for key, total in (("route_F", d_f), ("route_Q", d_q)):
        if key in row:
            _check(isinstance(row[key], dict) and sum(_nat(x, key) for x in row[key].values()) == total,
                   f"{key} does not reconcile")
    tail = {k: _nat(row[k], k) for k in TAIL}
    _check(tail["tail_discard"] <= tail["tail_open"]
           and tail["tail_final_matches"] <= tail["tail_last_candidate"]
           <= tail["tail_touched_positions"] <= tail["tail_Q"] <= d_q
           and tail["discarded_tail_Q"] <= tail["tail_Q"],
           "tail counters do not reconcile")
    return dict(F=f, physical_Q=d_q + e_q, decoder_Q=d_q, encoder_Q=e_q,
                decoder_F=d_f, encoder_F=e_f, max_decoder_Q=peak, joint_F=joint,
                categories=categories, **tail)


def _layout_work(row, arm, history):
    f = _nat(row["generation_forwards"], "generation_forwards", 1)
    q = _nat(row["generation_physical_padded_rows"], "generation_physical_padded_rows", 1)
    local = _nat(row["local_attention_Q"], "local_attention_Q")
    if arm.kind == "layout_llada":
        phase_work = {k: _nat(row[k], k) for k in ("denoise_history_rows", "commit_history_rows")}
        ticks = [_nat(v, "history_q_per_forward") for v in row["history_q_per_forward"]]
        _check(history == sum(phase_work.values()) == sum(ticks) and len(ticks) == f and max(ticks) <= 32
               and q == 32 * f and local == q, "LLaDA layout accounting mismatch")
        d_f, e_f, width = f, 0, 32
    else:
        trace, events = row["trace"], row["events"]
        _check(len(events) == f + 1 and events[0][0] == "encoder" and bool(trace),
               "missing Gemma layout prefill or generation trace")
        d_f, e_f, width = len(trace), f - len(trace), 256
        _check(e_f >= 0 and q == width * f
               and sum(tuple(e) == ("decoder", width) for e in events[1:]) == d_f
               and sum(tuple(e) == ("encoder", width) for e in events[1:]) == e_f,
               "Gemma layout physical ledger mismatch")
        ticks = [_nat(s["history_q"], "trace.history_q") for s in trace]
        excess, local_sum = 0, 0
        for s, h in zip(trace, ticks):
            _check(type(s["close"]) is bool and s["q"] == width and h <= width
                   and (h <= arm.budget or s["close"]), "invalid Gemma layout trace")
            excess += max(0, h - arm.budget)
            expected = {"split": width, "rect": h, "halo": min(width, 2*h)}[row["layout"]]
            _check(s.get("local_attention_q", width) == expected, "invalid local attention Q")
            local_sum += expected
        _check(sum(ticks) == history and local_sum == local
               and excess == _nat(row["close_history_excess"], "close_history_excess"),
               "Gemma layout history ledger mismatch")
        phase_work = {"close_history_excess": excess}
    return dict(F=f, physical_Q=q, decoder_Q=d_f*width, encoder_Q=e_f*width,
                decoder_F=d_f, encoder_F=e_f, max_decoder_Q=width,
                max_history_Q=max(ticks), local_attention_Q=local,
                over_history_budget_F=sum(tick > arm.budget for tick in ticks), **phase_work)


def normalize(row):
    """Project one worker receipt onto identity, arm, score and measured work."""
    _check(isinstance(row, dict), "receipt must be an object")
    identity = _identity(row, _sha(row["samples_sha256"], "samples_sha256"))
    c = _nat(row["budget"], "budget", 1)
    history = _nat(row["history_query_rows"], "history_query_rows")
    schema = row.get("schema")
    input_sha = None
    if schema == "single-factor-cross-v1":
        _check(type(row["slack"]) is bool, "invalid slack flag")
        den = _nat(row["late_den"], "late_den", 1)
        dose = tuple(_nat(i, "prefix") for i in row["prefix"])
        _check(len(dose) == 2, "invalid prefix dose")
        arm = _cross(c, row["gate"], row["slack"], den, dose)
        n = _nat(row["N"], "N")
        input_sha = _sha(row["input_sha256"], "input_sha256")
    elif schema == "rect-cold-v1":
        arm = _sweep(c, row["layout"], _nat(row["cold_share"], "cold_share"))
        n = _nat(row["output_tokens"], "output_tokens")
    elif schema is None and row.get("layout") == "split" and "prefix_den" in row:
        arm = _llada(c, _nat(row["cold_share"], "cold_share"), _nat(row["prefix_den"], "prefix_den", 1))
        n = _nat(row["output_tokens"], "output_tokens")
    else:
        raise ValueError(f"unsupported receipt schema {schema!r}")
    _check(ARMS.get(arm.name) == arm, f"unregistered arm {arm.name}")
    work = _cross_work(row, c, history) if arm.kind == "cross" else _layout_work(row, arm, history)
    _check(type(row["truncated"]) is bool, "invalid truncated flag")
    correct = row.get("correct")
    if "score_pending" in row:
        _check("correct" not in row and isinstance(row["score_pending"], str),
               "ambiguous score_pending")
    else:
        _check(type(correct) is bool, "expected Boolean correct or score_pending")
    scorer = _sha(row["scorer_sha256"], "scorer_sha256") if "scorer_sha256" in row else None
    return Trial(arm, identity, input_sha, scorer, correct, row["truncated"],
                 dict(N=n, history_Q=history, **work))


def _register(manifest):
    snapshot = _sha(manifest["samples_sha256"], "manifest samples_sha256")
    ids = [_identity(row, snapshot) for row in manifest["samples"]]
    _check([key[1] for key in ids] == list(COHORTS["frozen90"])
           and len({key[2:4] for key in ids}) == 90, "noncanonical frozen90 manifest")
    return snapshot, {key[1]: key for key in ids}


def _ratio(n, d):
    return n / d if d else None


def _work_summary(rows):
    keys = rows[0].work.keys() - {"max_decoder_Q", "max_history_Q", "categories"}
    totals = {k: sum(t.work[k] for t in rows) for k in sorted(keys)}
    totals["peak_decoder_Q"] = max(t.work["max_decoder_Q"] for t in rows)
    if "max_history_Q" in rows[0].work:
        totals["peak_history_Q"] = max(t.work["max_history_Q"] for t in rows)
    totals["truncated"] = sum(t.truncated for t in rows)
    totals["macro_TPF"] = mean(t.work["N"] / t.work["F"] for t in rows)
    totals["N_per_history_Q"] = _ratio(totals["N"], totals["history_Q"])
    totals["N_per_physical_Q"] = _ratio(totals["N"], totals["physical_Q"])
    if "tail_Q" in totals:
        totals["tail_match_rate"] = _ratio(totals["tail_final_matches"], totals["tail_last_candidate"])
        totals["discarded_tail_Q_share"] = _ratio(totals["discarded_tail_Q"], totals["tail_Q"])
        totals["categories"] = {k: sum(t.work["categories"][k] for t in rows) for k in CATEGORIES}
    return totals


def _interval(draws, count):
    if len(draws) != count:
        return None  # A pooled rate had a zero denominator in a resample.
    draws.sort()
    def quantile(p):
        j = (len(draws)-1)*p
        lo = int(j)
        return draws[lo] + (draws[min(lo+1, len(draws)-1)]-draws[lo])*(j-lo)
    return [quantile(.025), quantile(.975)]


def _bootstrap(pairs, scored, count, seed):
    """Resample matched requests, not tokens/forwards; quality uses scored pairs."""
    result = {}
    if count == 0:
        return result
    if len(scored) >= 2:
        rng = random.Random(seed)
        differences = [int(b.correct) - int(a.correct) for a, b in scored]
        draws = [sum(rng.choices(differences, k=len(scored))) / len(scored) for _ in range(count)]
        result["accuracy_delta"] = _interval(draws, count)
    if len(pairs) >= 2:
        rng = random.Random(seed)
        vectors = []
        has_tail = "tail_Q" in pairs[0][0].work
        rates = ("N_per_history_Q_delta",) + (("tail_match_rate_delta",) if has_tail else ())
        averages = ("macro_TPF_delta", "physical_Q_per_request_delta") + (
            ("tail_Q_per_request_delta",) if has_tail else ())
        for a, b in pairs:
            l, r = a.work, b.work
            v = (r["N"]/r["F"] - l["N"]/l["F"], r["physical_Q"] - l["physical_Q"])
            v += (r["tail_Q"] - l["tail_Q"],) if has_tail else ()
            v += (l["N"], l["history_Q"], r["N"], r["history_Q"])
            if has_tail:
                v += tuple(w[k] for w in (l, r) for k in ("tail_final_matches", "tail_last_candidate"))
            vectors.append(v)
        draws = {k: [] for k in (*averages, *rates)}
        for _ in range(count):
            sums = tuple(map(sum, zip(*rng.choices(vectors, k=len(pairs)))))
            for i, key in enumerate(averages):
                draws[key].append(sums[i] / len(pairs))
            for i, key in enumerate(rates):
                x, xd, y, yd = sums[len(averages)+4*i:len(averages)+4*(i+1)]
                if xd and yd:
                    draws[key].append(y/yd - x/xd)
        result.update((k, _interval(v, count)) for k, v in draws.items())
    return result


def _coverage(rows, indices):
    found = rows.keys() & set(indices)
    pending = sorted(i for i in found if rows[i].correct is None)
    return dict(expected=len(indices), received=len(found), scored=len(found)-len(pending),
                missing_indices=sorted(set(indices)-found), score_pending_indices=pending,
                status="complete" if len(found) == len(indices) and not pending else "incomplete")


def _paired(left, right, indices, draws, seed):
    shared = left.keys() & right.keys() & set(indices)
    mismatch = sorted(i for i in shared if left[i].identity != right[i].identity)
    pairs = [(left[i], right[i]) for i in sorted(shared-set(mismatch))]
    scored = [(a, b) for a, b in pairs if a.correct is not None and b.correct is not None]
    result = dict(status=("no_common" if not pairs else "complete" if len(scored) == len(indices)
                          else "incomplete"), expected=len(indices), paired_work=len(pairs),
                  paired_scored=len(scored), identity_mismatch_indices=mismatch,
                  reference_only=len(set(indices) & (left.keys()-right.keys())),
                  trial_only=len(set(indices) & (right.keys()-left.keys())),
                  paired_score_pending_indices=[a.index for a, b in pairs
                                                 if a.correct is None or b.correct is None])
    if pairs:
        l, r = (_work_summary(rows) for rows in zip(*pairs))
        result["work"] = dict(reference=l, trial=r, delta={
            k: (r[k]-l[k] if r[k] is not None and l[k] is not None else None)
            for k in l if k != "categories"})
        result["confidence95"] = _bootstrap(pairs, scored, draws, seed)
    if scored:
        nc_ref = sum(a.correct for a, _ in scored)
        nc_trial = sum(b.correct for _, b in scored)
        result["quality"] = dict(reference_correct=nc_ref, trial_correct=nc_trial,
                                 rescue=sum(not a.correct and b.correct for a, b in scored),
                                 loss=sum(a.correct and not b.correct for a, b in scored),
                                 accuracy_delta=(nc_trial-nc_ref)/len(scored))
    return result


def summarize(trials, *, manifest=None, kinds=None, comparisons=None, omit_empty=False,
              draws=1000, seed=0):
    """Summarize normalized trials; CLI always supplies a frozen90 manifest."""
    _nat(draws, "bootstrap draws")
    pinned, expected = _register(manifest) if manifest is not None else (None, None)
    rows, inputs, scorers, snapshots, seen = defaultdict(dict), {}, {}, set(), set()
    for trial in trials:
        arm, i = trial.arm, trial.index
        snapshots.add(trial.identity[0])
        if expected is not None:
            _check(trial.identity == expected[i], f"frozen identity mismatch: {arm.name} index {i}")
        _check(i not in rows[arm], f"duplicate receipt: {arm.name} index {i}")
        rows[arm][i] = trial
        seen.add(arm.kind)
        if trial.input_sha is not None:
            key = (arm.kind, trial.identity)
            _check(inputs.setdefault(key, trial.input_sha) == trial.input_sha,
                   f"input digest mismatch: {arm.name} index {i}")
        if trial.scorer_sha is not None:
            _check(scorers.setdefault(arm.kind, trial.scorer_sha) == trial.scorer_sha,
                   f"scorer digest mismatch: {arm.kind}")
    _check(len(snapshots) <= 1, "multiple samples snapshots; report separately")
    selected = tuple(kinds) if kinds else tuple(k for k in KINDS if k in seen) or ("cross",)
    _check(set(selected) <= set(KINDS), "invalid worker kind")
    pairs = [pair for pair in COMPARISONS if pair[0].kind in selected] if comparisons is None else [
        (ARMS[a], ARMS[b]) for a, b in comparisons]
    for a, b in pairs:
        _check(a != b and a.kind == b.kind and a.budget == b.budget
               and (not kinds or a.kind in selected),
               "compare distinct arms of the same worker and budget")
    coverage = {}
    for arm in sorted({arm for pair in pairs for arm in pair}):
        coverage[arm.name] = dict(worker=arm.kind, budget=arm.budget,
                                  budget_axis="decoder_Q" if arm.kind == "cross" else "history_Q",
                                  cohorts={k: _coverage(rows[arm], ix) for k, ix in COHORTS.items()})
    comparisons_out = []
    complete = bool(pairs)
    for a, b in pairs:
        cohorts = {k: _paired(rows[a], rows[b], ix, draws, seed) for k, ix in COHORTS.items()}
        complete &= cohorts["frozen90"]["status"] == "complete"
        empty = [k for k, v in cohorts.items() if v["status"] == "no_common"]
        comparisons_out.append(dict(reference=a.name, trial=b.name, worker=a.kind, budget=a.budget,
                                    status=cohorts["frozen90"]["status"],
                                    cohorts={k: v for k, v in cohorts.items() if not omit_empty or k not in empty},
                                    omitted_no_common=empty if omit_empty else []))
    return dict(schema="cross-race-report-v1", status="complete" if complete else "incomplete",
                samples_sha256=pinned or next(iter(snapshots), None), manifest_verified=manifest is not None,
                bootstrap=dict(method="paired request percentile; conditional on matched coverage",
                               draws=draws, seed=seed, level=.95),
                metric_notes=dict(history_Q="cross: selected decoder Q; layout: selected history Q",
                                  physical_Q="generation rows including encoder; prefill excluded",
                                  tail_match_rate="final token matches / last speculative candidates, not causal utility",
                                  work="common generated pairs, including score_pending",
                                  quality="common pairs with two scores"),
                arms=coverage, comparisons=comparisons_out)


def _read_trials(paths):
    for source in paths:
        if source.is_dir():
            files = sorted(source.glob("*.json"))
        else:
            files = [source]
        for path in files:
            try:
                yield normalize(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
                raise ValueError(f"{path}: {exc}") from exc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("receipts", type=Path, nargs="+", help="receipt JSON files or worker output directories")
    ap.add_argument("--manifest", type=Path, default=MANIFEST, help="frozen90 identity manifest")
    ap.add_argument("--schema", choices=KINDS, action="append",
                    help="report this worker, even when its directory is empty; repeatable")
    ap.add_argument("--compare", nargs=2, action="append", metavar=("REFERENCE", "TRIAL"),
                    help="compare registered worker file stems without the -<index> suffix; repeatable")
    ap.add_argument("--omit-empty", action="store_true", help="omit no-common cohort detail, keep status")
    ap.add_argument("--require-complete", action="store_true", help="emit report, exit 1 if incomplete")
    ap.add_argument("--bootstrap", type=int, default=1000, metavar="DRAWS", help="0 disables intervals")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed, not generation seed")
    args = ap.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        report = summarize(_read_trials(args.receipts), manifest=manifest, kinds=args.schema,
                           comparisons=args.compare, omit_empty=args.omit_empty,
                           draws=args.bootstrap, seed=args.seed)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        ap.error(str(exc))
    print(json.dumps(report, ensure_ascii=True, allow_nan=False, indent=2))
    return int(args.require_complete and report["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
