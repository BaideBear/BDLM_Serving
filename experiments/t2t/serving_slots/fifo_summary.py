"""Independent 30/60/90 summary of the completed Gemma FIFO run.

CPU only.  The script never touches a GPU and never re-scores a request.  It
reads the six pairwise score files written by ``fifo_report`` plus the raw
request and clock records of the completed run, re-derives every published
number, and fails closed when timing, request identity, direction or work
accounting disagrees.  Output: ``fifo/Gemma.csv`` and ``fifo/README.md`` next
to this module.

Gemma/frozen90 only: the run metadata is checked and anything else is rejected,
so the generated conclusion cannot be applied to another model or dataset.

Reproduce from the repository root (replace the placeholders):

    PYTHONPATH=. <python> -m experiments.t2t.serving_slots.fifo_summary \
      --run <fifo-run> --scores <fifo-scores> --cases <samples.json> \
      --out experiments/t2t/serving_slots/fifo
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_right
from itertools import combinations
from pathlib import Path
from statistics import mean, median

from bdlm_serving.evaluation.frozen import sha
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.methods.t2t.protocol import verify_work

from .fifo import MODES, SLOTS, Pressure, arrival_trace

MODEL = "gemma"
WINDOWS = (("original30", 0, 30), ("added60", 30, 90), ("all90", 0, 90))
PAIR_WINDOW = {"original30": 30, "added60": 60, "all90": 90}
POLICY = {"fixed_high": (64, 64), "busy_high": (64, 32),
          "busy_low": (32, 64), "fixed_low": (32, 32)}
TIMING = (("inject_lag", "planned", "enqueued"), ("wait", "enqueued", "start"),
          ("service", "start", "finish"), ("response", "enqueued", "finish"),
          ("scheduled_response", "planned", "finish"))

ARM_FIELDS = [
    "record_type", "model", "mode", "scope", "n", "correct", "quality",
    "macro_tpf", "pooled_token_efficiency", "N", "F", "physical_Q", "q_per_f",
    "failed", "truncated", "low_offers", "high_offers", "requests_switched",
    "busy_offers", "idle_offers", "inherit_offers", "inject_lag_mean_ms",
    "inject_lag_p50_ms", "inject_lag_p95_ms", "max_inject_lag_ms",
    "wait_mean_ms", "wait_p50_ms", "wait_p95_ms", "service_mean_ms",
    "service_p50_ms", "service_p95_ms", "response_mean_ms", "response_p50_ms",
    "response_p95_ms", "scheduled_response_mean_ms", "scheduled_response_p50_ms",
    "scheduled_response_p95_ms", "unit_ms", "clock_pass", "gap_error_p95_ms",
    "arrival_span_relerr", "achieved_requests_s", "samples_sha256",
]
PAIR_FIELDS = [
    "record_type", "model", "baseline", "arm", "scope", "n",
    "baseline_correct", "arm_correct", "quality_delta", "quality_ci_low",
    "quality_ci_high", "quality_direction", "quality_ci_excludes_zero",
    "rescued", "lost", "baseline_macro_tpf", "arm_macro_tpf", "tpf_delta",
    "tpf_ci_low", "tpf_ci_high", "tpf_direction", "tpf_ci_excludes_zero",
    "baseline_physical_Q", "arm_physical_Q", "q_delta", "q_delta_pct",
    "q_direction", "baseline_F", "arm_F", "f_delta_pct", "f_direction",
    "baseline_response_mean_ms", "arm_response_mean_ms", "response_delta_ms",
    "response_delta_pct", "response_direction", "samples_sha256",
]
FIELDS = ARM_FIELDS + [f for f in PAIR_FIELDS if f not in ARM_FIELDS]


def read_json(path):
    return json.loads(Path(path).read_text())


def p95(values):
    return sorted(values)[max(0, math.ceil(.95 * len(values)) - 1)]


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def clean(value):
    return "" if isinstance(value, float) and math.isnan(value) else value


def clock_quality(times, unit):
    span = times[-1]["planned"] - times[0]["planned"]
    lags = [r["enqueued"] - r["planned"] for r in times]
    gaps = [abs((v["enqueued"] - u["enqueued"]) - (v["planned"] - u["planned"]))
            for j, (u, v) in enumerate(zip(times, times[1:])) if j % 6 != 5]
    span_error = abs(lags[-1] - lags[0]) / span
    return dict(
        clock_pass=p95(lags) <= unit / 80 and p95(gaps) <= unit / 80 and span_error <= .01,
        inject_lag_p95_ms=p95(lags) / 1e6,
        inject_lag_max_ms=max(lags) / 1e6,
        gap_error_p95_ms=p95(gaps) / 1e6,
        arrival_span_relerr=span_error,
        achieved_requests_s=len(times) * 1e9 / (times[-1]["finish"] - times[0]["planned"]),
    )


def validate_clock(times, trace, mode, slots):
    """Re-implement fifo_report.validate so the check is independent."""
    seq = [r["enqueued"] for r in times]
    if len(times) != len(trace) or sorted(seq) != seq:
        raise ValueError(f"{mode}: incomplete or reordered FIFO arrivals")
    previous, state = -1, Pressure(mode, slots)
    for rank, (r, (i, target)) in enumerate(zip(times, trace, strict=True)):
        a, s, f = r["enqueued"], r["start"], r["finish"]
        if r["index"] != i or r["planned"] != target or not target <= a <= s < f or s < previous:
            raise ValueError(f"{mode}: FIFO schedule/interval/order mismatch at rank {rank}")
        previous, prior = f, s
        for j, (tick, t, depth, q) in enumerate(r["offers"]):
            if (tick != j or not prior <= t <= f
                    or depth != bisect_right(seq, t) - rank - 1
                    or state.offer(depth) != q):
                raise ValueError(f"{mode}: wrong arrival snapshot or offered budget at rank {rank}/{j}")
            prior = t
    if [r["index"] for r in times] != [i for i, _ in trace] or len({r["index"] for r in times}) != 90:
        raise ValueError(f"{mode}: request indices are not the 90 unique trace indices")


def load_arm(run, mode, tasks, trace, unit):
    if read_json(run / f"complete-{mode}.json") != dict(policy=mode, requests=90):
        raise ValueError(f"Incomplete FIFO arm: {mode}")
    clock = read_json(run / f"clock-{mode}.json")
    if clock.get("mode") != mode:
        raise ValueError(f"Wrong clock arm: {clock.get('mode')} != {mode}")
    times = clock["requests"]
    validate_clock(times, trace, mode, SLOTS[MODEL])
    by_index = {r["index"]: r for r in times}
    expected = [f"{i:02d}-{mode}.json" for i in range(90)]
    found = sorted(p.name for p in run.glob(f"[0-9][0-9]-{mode}.json"))
    if found != expected:
        raise ValueError(f"{mode}: missing or extra request files ({len(found)} of 90)")
    rows = {}
    for i in range(90):
        row = read_json(run / f"{i:02d}-{mode}.json")
        if (row["index"], row["policy"], row["model"]) != (i, mode, MODEL):
            raise ValueError(f"Request identity mismatch: {mode}/{i}")
        for key in ("task_id", "dataset", "seed"):
            if row[key] != tasks[i][key]:
                raise ValueError(f"Case identity mismatch: {mode}/{i}/{key}")
        verify_work(row)
        if row["status"] != "success" or row["truncated"]:
            raise ValueError(f"Non-success or truncated request: {mode}/{i}")
        observed = [e["slot_offer"] for e in row["events"] if "slot_offer" in e]
        if observed != [d[-1] for d in by_index[i]["offers"]]:
            raise ValueError(f"Backend and clock offers disagree: {mode}/{i}")
        rows[i] = row
    return dict(mode=mode, times=times, by_index=by_index, rows=rows,
                clock=clock_quality(times, unit))


def arm_summary(arm, lo, hi):
    rs = [arm["by_index"][i] for i in range(lo, hi)]
    out = dict(n=len(rs))
    for name, a, b in TIMING:
        vals = [(r[b] - r[a]) / 1e6 for r in rs]
        out[f"{name}_mean_ms"], out[f"{name}_p50_ms"] = mean(vals), median(vals)
        out[f"{name}_p95_ms"] = p95(vals)
    out["max_inject_lag_ms"] = max((r["enqueued"] - r["planned"]) / 1e6 for r in rs)
    low, high = SLOTS[MODEL]
    offers = [d for r in rs for d in r["offers"]]
    out["low_offers"] = sum(d[-1] == low for d in offers)
    out["high_offers"] = sum(d[-1] == high for d in offers)
    out["requests_switched"] = sum(len({d[-1] for d in r["offers"]}) > 1 for r in rs)
    out["busy_offers"] = sum(d[2] >= 2 for d in offers)
    out["idle_offers"] = sum(d[2] == 0 for d in offers)
    out["inherit_offers"] = sum(d[2] == 1 for d in offers)
    out["N"] = sum(arm["rows"][i]["output_tokens"] for i in range(lo, hi))
    out["F"] = sum(arm["rows"][i]["generation_forwards"] for i in range(lo, hi))
    out["Q"] = sum(arm["rows"][i]["generation_query_rows"] for i in range(lo, hi))
    out["physical_Q"] = sum(arm["rows"][i]["generation_physical_padded_rows"] for i in range(lo, hi))
    out["failed"] = sum(arm["rows"][i]["status"] != "success" for i in range(lo, hi))
    out["truncated"] = sum(bool(arm["rows"][i]["truncated"]) for i in range(lo, hi))
    out["q_per_f"] = out["physical_Q"] / out["F"]
    out["macro_tpf"] = mean(arm["rows"][i]["tpf"] for i in range(lo, hi))
    out["pooled_token_efficiency"] = out["N"] / out["physical_Q"]
    return out


def pair_window(scope):
    return next((lo, hi) for name, lo, hi in WINDOWS if name == scope)


def load_pair(scores, baseline, arm, arms):
    name = f"{baseline}-vs-{arm}"
    rows = [r for r in csv.DictReader((scores / f"{name}.csv").open()) if r["scope"] == "all"]
    by_window = {int(r["window"]): r for r in rows}
    if set(by_window) != {30, 60, 90}:
        raise ValueError(f"{name}: pooled windows are {sorted(by_window)}, not 30/60/90")
    paired = list(csv.DictReader((scores / f"{name}-paired.csv").open()))
    out = {}
    for scope, window in PAIR_WINDOW.items():
        lo, hi = pair_window(scope)
        ps = [p for p in paired if int(p["window"]) == window]
        if len(ps) != hi - lo or sorted(int(p["index"]) for p in ps) != list(range(lo, hi)):
            raise ValueError(f"{name}/{scope}: paired cohort is not the unique {hi - lo} cases")
        rec = dict(
            baseline_correct=sum(int(p["baseline_correct"]) for p in ps),
            arm_correct=sum(int(p["arm_correct"]) for p in ps),
            rescued=sum(int(p["baseline_correct"]) == 0 and int(p["arm_correct"]) == 1 for p in ps),
            lost=sum(int(p["baseline_correct"]) == 1 and int(p["arm_correct"]) == 0 for p in ps),
            baseline_macro_tpf=mean(float(p["baseline_tpf"]) for p in ps),
            arm_macro_tpf=mean(float(p["arm_tpf"]) for p in ps),
            baseline_pooled_token_efficiency=(sum(int(p["baseline_N"]) for p in ps)
                                              / sum(int(p["baseline_Q"]) for p in ps)),
            arm_pooled_token_efficiency=(sum(int(p["arm_N"]) for p in ps)
                                         / sum(int(p["arm_Q"]) for p in ps)),
        )
        r = by_window[window]
        for key, val in rec.items():
            if not close(r[key], val):
                raise ValueError(f"{name}/{scope}/{key}: paired recompute {val} != CSV {r[key]}")
        rec["n"] = hi - lo
        for side, mode in (("baseline", baseline), ("arm", arm)):
            raw = arms[mode]["rows"]
            work = (sum(raw[i]["output_tokens"] for i in range(lo, hi)),
                    sum(raw[i]["generation_forwards"] for i in range(lo, hi)),
                    sum(raw[i]["generation_physical_padded_rows"] for i in range(lo, hi)))
            if (int(r[f"{side}_N"]), int(r[f"{side}_F"]), int(r[f"{side}_physical_Q"])) != work:
                raise ValueError(f"{name}/{scope}/{side}: raw work {work} != pair CSV")
            if int(r[f"{side}_Q"]) != int(r[f"{side}_physical_Q"]):
                raise ValueError(f"{name}/{scope}/{side}: nominal Q != physical Q")
            if not close(r[f"{side}_q_per_f"], int(r[f"{side}_physical_Q"]) / int(r[f"{side}_F"])):
                raise ValueError(f"{name}/{scope}/{side}: q_per_f disagrees")
        rec.update(
            quality_delta=float(r["quality_delta"]),
            quality_ci_low=float(r["quality_delta_ci_low"]),
            quality_ci_high=float(r["quality_delta_ci_high"]),
            tpf_delta=float(r["macro_tpf_delta"]),
            tpf_ci_low=float(r["macro_tpf_delta_ci_low"]),
            tpf_ci_high=float(r["macro_tpf_delta_ci_high"]),
            baseline_physical_Q=int(r["baseline_physical_Q"]),
            arm_physical_Q=int(r["arm_physical_Q"]),
            baseline_F=int(r["baseline_F"]),
            arm_F=int(r["arm_F"]),
        )
        rec["quality_direction"] = ("arm_better" if rec["quality_ci_low"] > 0
                                    else "arm_worse" if rec["quality_ci_high"] < 0 else "tie")
        rec["tpf_direction"] = ("arm_better" if rec["tpf_ci_low"] > 0
                                else "arm_worse" if rec["tpf_ci_high"] < 0 else "tie")
        rec["q_delta"] = rec["arm_physical_Q"] - rec["baseline_physical_Q"]
        rec["q_delta_pct"] = 100 * rec["q_delta"] / rec["baseline_physical_Q"]
        rec["q_direction"] = ("arm_less_q" if rec["q_delta"] < 0
                              else "arm_more_q" if rec["q_delta"] > 0 else "tie")
        rec["f_delta_pct"] = 100 * (rec["arm_F"] - rec["baseline_F"]) / rec["baseline_F"]
        rec["f_direction"] = ("arm_more_f" if rec["arm_F"] > rec["baseline_F"]
                              else "arm_less_f" if rec["arm_F"] < rec["baseline_F"] else "tie")
        rec["baseline_response_mean_ms"] = (arms[baseline]["all90"]["response_mean_ms"]
                                            if scope == "all90" else float("nan"))
        rec["arm_response_mean_ms"] = (arms[arm]["all90"]["response_mean_ms"]
                                       if scope == "all90" else float("nan"))
        rec["response_delta_ms"] = (rec["arm_response_mean_ms"] - rec["baseline_response_mean_ms"]
                                    if scope == "all90" else float("nan"))
        rec["response_delta_pct"] = (100 * rec["response_delta_ms"] / rec["baseline_response_mean_ms"]
                                     if scope == "all90" else float("nan"))
        rec["response_direction"] = ("" if scope != "all90" else
                                    "arm_faster" if rec["response_delta_ms"] < 0
                                    else "arm_slower" if rec["response_delta_ms"] > 0 else "tie")
        out[scope] = rec
    return out


def mode_view(scores):
    """Every mode's per-window metrics must be identical in all six pair files."""
    view = {m: {} for m in MODES}
    for b, a in combinations(MODES, 2):
        for r in csv.DictReader((scores / f"{b}-vs-{a}.csv").open()):
            if r["scope"] != "all":
                continue
            w = int(r["window"])
            for mode, side in ((b, "baseline"), (a, "arm")):
                key = (int(r[f"{side}_correct"]), float(r[f"{side}_macro_tpf"]),
                       int(r[f"{side}_N"]), int(r[f"{side}_F"]), int(r[f"{side}_physical_Q"]))
                if w in view[mode] and view[mode][w] != key:
                    raise ValueError(f"{mode}/{w}: metrics differ across pair files: {view[mode][w]} vs {key}")
                view[mode][w] = key
    return view


def check_clock_csv(scores, arms, unit_ns):
    rows = list(csv.DictReader((scores / "fifo-clock.csv").open()))
    by = {(r["mode"], r["scope"]): r for r in rows}
    if set(by) != {(m, s) for m in MODES for s, _, _ in WINDOWS}:
        raise ValueError("fifo-clock.csv does not cover 4 modes x 3 scopes")
    for mode in MODES:
        for scope, lo, hi in WINDOWS:
            r, s = by[(mode, scope)], arm_summary(arms[mode], lo, hi)
            for key in ("n", "N", "F", "physical_Q", "low_offers", "high_offers", "requests_switched"):
                if int(r[key]) != int(s[key]):
                    raise ValueError(f"fifo-clock.csv disagrees: {mode}/{scope}/{key}")
            for key in ("inject_lag_mean_ms", "inject_lag_p50_ms", "inject_lag_p95_ms",
                        "max_inject_lag_ms", "wait_mean_ms", "wait_p50_ms", "wait_p95_ms",
                        "service_mean_ms", "service_p50_ms", "service_p95_ms",
                        "response_mean_ms", "response_p50_ms", "response_p95_ms",
                        "scheduled_response_mean_ms", "scheduled_response_p50_ms",
                        "scheduled_response_p95_ms", "unit_ms"):
                if not close(r[key], s.get(key, unit_ns / 1e6), 1e-6):
                    raise ValueError(f"fifo-clock.csv disagrees: {mode}/{scope}/{key}")
            if (scope == "all90" and r["clock_pass"] != "True") or (scope != "all90" and r["clock_pass"]):
                raise ValueError(f"fifo-clock.csv clock_pass is wrong: {mode}/{scope}")
            if scope == "all90":
                c = arms[mode]["clock"]
                for key, val in (("in_burst_gap_abs_error_p95_ms", c["gap_error_p95_ms"]),
                                 ("arrival_span_relerr", c["arrival_span_relerr"]),
                                 ("achieved_requests_s", c["achieved_requests_s"])):
                    if not close(r[key], val, 1e-6):
                        raise ValueError(f"fifo-clock.csv disagrees: {mode}/{key}")


def pareto_set(metrics):
    kept = []
    for m, x in metrics.items():
        dominated = any(
            y["correct"] >= x["correct"] and y["tpf"] >= x["tpf"]
            and y["q"] <= x["q"] and y["response"] <= x["response"]
            and (y["correct"] > x["correct"] or y["tpf"] > x["tpf"]
                 or y["q"] < x["q"] or y["response"] < x["response"])
            for n, y in metrics.items() if n != m)
        if not dominated:
            kept.append(m)
    return kept


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def pct(x, nd=2):
    return f"{100 * x:.{nd}f}%"


def signed(x, nd=3):
    return f"{x:+.{nd}f}"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def reading(rec):
    q = ("显著变差" if rec["quality_direction"] == "arm_worse"
         else "显著变好" if rec["quality_direction"] == "arm_better" else "正确总数持平（不证明等价）")
    t = ("显著变差" if rec["tpf_direction"] == "arm_worse"
         else "显著变好" if rec["tpf_direction"] == "arm_better" else "持平")
    return (f"{q}，TPF {t}，Q {signed(rec['q_delta_pct'], 2)}%，"
            f"响应 {signed(rec['response_delta_pct'], 1)}%")


def dominates(rec):
    """A switching arm would have to be no worse on quality/TPF/latency and better on Q."""
    return (rec["quality_direction"] != "arm_worse" and rec["tpf_direction"] != "arm_worse"
            and rec["q_delta"] < 0 and rec["response_delta_ms"] <= 0)


def build_readme(meta, arms, pairs, pareto, samples_sha):
    fh, bh, bl, fl = (arms[m]["all90"] for m in ("fixed_high", "busy_high", "busy_low", "fixed_low"))
    r_bh = pairs[("fixed_high", "busy_high")]["all90"]
    r_bl = pairs[("fixed_high", "busy_low")]["all90"]
    r_fl = pairs[("fixed_high", "fixed_low")]["all90"]
    unit_ms = meta["unit_ns"] / 1e6
    threshold = unit_ms / 80
    any_dominates = dominates(r_bh) or dominates(r_bl)
    all_q = [arms[m]["all90"]["correct"] for m in MODES]
    quality_sentence = (f"四种策略全 90 都答对 **{all_q[0]}/90**" if len(set(all_q)) == 1
                        else "四种策略全 90 分别答对 " + "、".join(f"{m} {q}/90" for m, q in zip(MODES, all_q)))
    quality_half = [(r["all90"]["quality_ci_high"] - r["all90"]["quality_ci_low"]) / 2 for r in pairs.values()]
    tpf_sig = sum(1 for r in pairs.values() if r["all90"]["tpf_direction"] != "tie")
    lag = [arms[m]["clock"]["inject_lag_p95_ms"] for m in MODES]
    gap = [arms[m]["clock"]["gap_error_p95_ms"] for m in MODES]
    span = max(arms[m]["clock"]["arrival_span_relerr"] for m in MODES)
    busy_frac = [arms[m]["all90"]["busy_offers"] / sum(arms[m]["all90"][k] for k in ("busy_offers", "idle_offers", "inherit_offers"))
                 for m in MODES]
    offers = "、".join(f"{arms[m]['all90']['low_offers']}/{arms[m]['all90']['high_offers']}" for m in MODES)
    switched = {m: arms[m]["all90"]["requests_switched"] for m in MODES}

    mode_rows = []
    for m in MODES:
        s = arms[m]["all90"]
        busy, idle = POLICY[m]
        mode_rows.append([m, busy, idle, s["correct"], fmt(s["macro_tpf"]),
                          pct(s["pooled_token_efficiency"]), s["N"], s["F"],
                          f"{s['physical_Q']:,}", fmt(s["q_per_f"], 2),
                          f"{s['response_mean_ms'] / 1000:.3f} s",
                          f"{s['response_p95_ms'] / 1000:.3f} s"])

    scope_rows = []
    for m in MODES:
        cells = [m]
        for scope in ("original30", "added60", "all90"):
            cells.append(arms[m][scope]["correct"])
        for scope in ("original30", "added60", "all90"):
            cells.append(fmt(arms[m][scope]["macro_tpf"]))
        for scope in ("original30", "added60", "all90"):
            cells.append(f"{arms[m][scope]['physical_Q']:,}")
        scope_rows.append(cells)

    pair_rows = []
    for (b, a), by_scope in pairs.items():
        r = by_scope["all90"]
        pair_rows.append([
            f"{b} → {a}",
            f"{signed(r['quality_delta'], 3)} [{signed(r['quality_ci_low'], 3)}, {signed(r['quality_ci_high'], 3)}]",
            f"{signed(r['tpf_delta'], 3)} [{signed(r['tpf_ci_low'], 3)}, {signed(r['tpf_ci_high'], 3)}]",
            f"{r['q_delta']:+,} ({signed(r['q_delta_pct'], 2)}%)",
            f"{r['f_delta_pct']:+.1f}%",
            f"{r['response_delta_ms'] / 1000:+.3f} s ({signed(r['response_delta_pct'], 1)}%)",
            reading(r),
        ])

    verdict_head = ("**不建议把 FIFO 切档设为默认。**" if not any_dominates
                    else "**切档在本数据上有可取之处，但仍需按目标选择。**")
    recommend = ("在本数据上默认仍应是 `fixed_high`。只有在名义 Q 预算受限（名义 Q 可能溢出，物理 Q 以实测为准）、"
                 "且能接受约 +{:.0f}% 响应时间时，才考虑 `busy_low`；`busy_high` 在本数据上的 Q 节省很小，平均响应未改善，"
                 "不建议作为默认或省 Q 方案。".format(r_bl["response_delta_pct"]) if not any_dominates
                 else "切档在某个维度上有优势，但是否作为默认取决于目标；本数据不足以支持通用默认。")

    text = f"""# Gemma FIFO 切档：独立汇总与默认判断

由 `python -m experiments.t2t.serving_slots.fifo_summary` 生成。输入是已完成的 Gemma FIFO 90 题轨迹（4 种策略）和六份配对评分；脚本不重跑 GPU、不重新评分，只从原始请求记录、clock 和配对 CSV 重算并交叉核对。只接受 Gemma frozen90 运行，元数据不符直接报错，不套用其它模型或数据。

## 结论

{verdict_head} 在本次单卡、单次、合成到达计划的 90 题实验中：

- {quality_sentence}（正确总数相同，不证明等价）；六个配对的质量差 95% 区间全部跨 0（半宽约 ±{min(quality_half) * 100:.1f} 到 ±{max(quality_half) * 100:.1f} 个百分点），没有质量证据支持切档。
- `fixed_high`（恒 64 槽）TPF 最高（**{fmt(fh['macro_tpf'])}**）、响应均值最低（**{fh['response_mean_ms'] / 1000:.3f} s**），物理 Q {fh['physical_Q']:,}。
- `busy_high`（忙时 64、闲时 32）与 `fixed_high` 的正确总数、Q、响应接近，但 TPF 低 **{abs(r_bh['tpf_delta']):.3f}**（95% CI [{r_bh['tpf_ci_low']:.3f}, {r_bh['tpf_ci_high']:.3f}]），forward 多 **{r_bh['f_delta_pct']:.1f}%**；Q 只少 **{abs(r_bh['q_delta_pct']):.2f}%**，没有明显优势。
- `busy_low`（忙时 32、闲时 64）物理 Q 少 **{abs(r_bl['q_delta_pct']):.2f}%**，但 TPF 低 **{abs(r_bl['tpf_delta']):.3f}**、响应均值慢 **{r_bl['response_delta_pct']:.1f}%**（{r_bl['response_delta_ms'] / 1000:+.3f} s）；`fixed_low` 的 Q 少 **{abs(r_fl['q_delta_pct']):.2f}%**、响应慢 **{r_fl['response_delta_pct']:.1f}%**、TPF 低 **{abs(r_fl['tpf_delta']):.3f}**。低档省 Q 的代价是更多 forward 和更长时延。
- {recommend}

四个策略在 (正确总数, TPF, 物理 Q, 响应) 上都处于 Pareto 前沿（{'、'.join(pareto)}）：正确总数相同（不证明等价），Q 与时延/推进速度互相牺牲。默认应按目标选；按通用目标（TPF 与时延优先、Q 作次要约束）选 `fixed_high`。

## 全 90 数值

{md_table(["策略", "忙时槽", "闲时槽", "正确", "TPF", "效率", "N", "F", "物理 Q", "Q/F", "响应均值", "响应 p95"], mode_rows)}

- 正确 = 90 题答对数；TPF = 逐题 `N/F` 的均值；效率 = `ΣN/Σ物理 Q`；Q 为物理 padded rows（含批内 padding，纯 prompt prefill 除外）。
- `fixed_high`/`fixed_low` 恒定 64/32；`busy_high` 忙时 64、闲时 32；`busy_low` 忙时 32、闲时 64。忙 = 等待队列深度 ≥ 2，闲 = 0，等于 1 继承前态。
- 到达计划使队列在 {min(busy_frac) * 100:.0f}%–{max(busy_frac) * 100:.0f}% 的选位轮处于忙态；`busy_low` 与 `busy_high` 分别只有 {switched['busy_low']}/90、{switched['busy_high']}/90 个请求同时看到两种档位。两种切档策略大部分时间表现接近其主导档位，本实验不能单独识别“切换”本身的价值。

## 原 30 / 新 60 / 全 90

{md_table(["策略", "原30 正确", "新60 正确", "全90 正确", "原30 TPF", "新60 TPF", "全90 TPF", "原30 Q", "新60 Q", "全90 Q"], scope_rows)}

原 30 = case 0–29（GSM8K/MATH-500/HumanEval 各 10）；新 60 = case 30–89（各 20）；全 90 为两者的并集。原 30 与新 60 互不重叠。

## 六个配对与方向（全 90）

{md_table(["基线 → 臂", "质量差 [95% CI]", "TPF 差 [95% CI]", "Q 差", "F 差", "响应差", "读法"], pair_rows)}

- 质量差 = 臂正确率 − 基线正确率；TPF 差 = 臂 macro TPF − 基线 macro TPF；Q/F 为全 90 实测总量。区间为配对 95% 区间（质量用 Newcombe，TPF 用分层正态），未做多重比较校正。
- 六个 TPF 区间都排除 0；六个质量区间都跨 0。方向：TPF `fixed_high` > `busy_high` > `busy_low` > `fixed_low`；物理 Q 相反（越小越省）；响应 `fixed_high` ≈ `busy_high` < `busy_low` < `fixed_low`。

## 检查

- **计时有效**：单位 {unit_ms:.3f} ms；p95 入队延迟 {min(lag):.3f}–{max(lag):.3f} ms、组内间距误差 p95 {min(gap):.3f}–{max(gap):.3f} ms（判据均为 {threshold:.3f} ms）、到达跨度相对误差 ≤ {span:.2e}（判据 1%）。四臂全部通过；逐请求核对 `planned ≤ enqueued ≤ start < finish`、单服务台不重叠、offer tick/深度/预算映射；clock 为 `monotonic_ns`，计时排除写盘与评分。
- **请求不重复**：四臂各恰好 90 份请求文件，index 0–89 无重复；clock 顺序等于登记 trace；六个配对各有 90 个唯一配对 index；每个策略在六份配对文件中重复出现时逐窗口指标一致。无失败、无截断。
- **方向**：`busy_low` 忙→32/闲→64、`busy_high` 忙→64/闲→32、两个 fixed 恒定；实测 offer（低/高）{offers}。配对方向见上表：正确总数都相同（不证明等价），TPF 的 {tpf_sig}/6 个区间排除 0。
- **work**：360 份请求全部通过 `verify_work`（`F>0`、`Q≥F`、`Q==物理 Q`、`N≥0`、`tpf==N/F`、`token_efficiency==N/Q`）；逐窗口 N/F/物理 Q 与六份配对 CSV 及 `fifo-clock.csv` 逐项相等。Q 是行数，不是 GPU 时间；TPF 是每 forward 的 token 数，不是墙钟速度。

## 复现

汇总（CPU，无需 GPU；路径按本机替换）：

```bash
cd <repo>
PYTHONPATH=. <python> -m experiments.t2t.serving_slots.fifo_summary \\
  --run <fifo-run> --scores <fifo-scores> --cases <samples.json> \\
  --out experiments/t2t/serving_slots/fifo
```

上游评分（CPU）：`PYTHONPATH=. <python> -m experiments.t2t.serving_slots.fifo_report --run <fifo-run> --cases <samples.json> --out <fifo-scores>`。

FIFO 轨迹由 `launch_prefix90.py` 在 GPU0 的 holder 协议下生成；重跑需要 holder 与权重，不属于本汇总。`samples.json` SHA-256 `{samples_sha}`。

## 限制

1. 单次运行、单卡（H800 PCIe，GPU0）、单一合成到达计划；没有生产流量、重复实验或稳定 p95。
2. 质量检验功效不足：90 题、六个质量区间都跨 0，不能排除若干个百分点的真实差异；正确总数相同不证明等价。
3. 队列在 {min(busy_frac) * 100:.0f}%–{max(busy_frac) * 100:.0f}% 的轮次忙，`busy_low`/`busy_high` 大部分时间接近主导档位，只有 {switched['busy_low']}/{switched['busy_high']} 个请求看到两种档位；本实验不能单独识别“切换”本身的价值。
4. 队列深度内生，各策略不强行等深；TPF 不是墙钟速度，Q 不是 GPU 时间；名义 Q 可能溢出，物理 Q 以实测为准。
5. 六个配对 × 三个窗口未做多重比较校正；只分析已完成的 `gemma-fifo90`（早先的 in-memory 尝试因 prefill 审计失败被排除）。
6. 结论只适用于 DiffusionGemma 26B-A4B（B=256）与本工作负载；单请求执行器不是多请求调度器。
"""
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "fifo")
    args = ap.parse_args()

    meta = read_json(args.run / "run.json")
    tasks = load_tasks(args.cases)
    samples_sha = sha(args.cases)
    if (meta["model"] != MODEL or meta["suite"] != "frozen90"
            or meta["samples_sha256"] != samples_sha or meta["batch_size"] != 1
            or meta["clock"] != "monotonic_ns" or list(meta["slots"]) != list(SLOTS[MODEL])
            or list(meta["policies"]) != list(MODES)):
        raise ValueError("Run metadata does not describe the completed Gemma frozen90 FIFO run")
    trace = arrival_trace(tasks, meta["unit_ns"])
    if list(map(tuple, meta["trace"])) != trace or len(trace) != 90 or len({i for i, _ in trace}) != 90:
        raise ValueError("Registered arrival trace is not the 90 unique FIFO cases")

    view = mode_view(args.scores)
    arms = {}
    for mode in MODES:
        arm = load_arm(args.run, mode, tasks, trace, meta["unit_ns"])
        for scope, lo, hi in WINDOWS:
            arm[scope] = arm_summary(arm, lo, hi)
            arm[scope]["correct"] = view[mode][PAIR_WINDOW[scope]][0]
        arms[mode] = arm

    pairs = {}
    for b, a in combinations(MODES, 2):
        pairs[(b, a)] = load_pair(args.scores, b, a, arms)
    check_clock_csv(args.scores, arms, meta["unit_ns"])

    # Every arm must also agree with the pair files on quality and work.
    for mode in MODES:
        for scope, window in (("original30", 30), ("added60", 60), ("all90", 90)):
            s, key = arms[mode][scope], view[mode][window]
            if (s["correct"], round(s["macro_tpf"], 9), s["N"], s["F"], s["physical_Q"]) != (
                    key[0], round(key[1], 9), key[2], key[3], key[4]):
                raise ValueError(f"{mode}/{scope}: arm summary disagrees with pair files")

    pareto = pareto_set({m: dict(correct=arms[m]["all90"]["correct"],
                                 tpf=arms[m]["all90"]["macro_tpf"],
                                 q=arms[m]["all90"]["physical_Q"],
                                 response=arms[m]["all90"]["response_mean_ms"]) for m in MODES})

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "Gemma.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore", restval="", lineterminator="\n")
        w.writeheader()
        for mode in MODES:
            for scope, _, _ in WINDOWS:
                s = arms[mode][scope]
                row = dict(record_type="arm", model=MODEL, mode=mode, scope=scope,
                           correct=s["correct"], quality=s["correct"] / s["n"],
                           unit_ms=meta["unit_ns"] / 1e6,
                           clock_pass=arms[mode]["clock"]["clock_pass"] if scope == "all90" else "",
                           gap_error_p95_ms=arms[mode]["clock"]["gap_error_p95_ms"] if scope == "all90" else "",
                           arrival_span_relerr=arms[mode]["clock"]["arrival_span_relerr"] if scope == "all90" else "",
                           achieved_requests_s=arms[mode]["clock"]["achieved_requests_s"] if scope == "all90" else "",
                           samples_sha256=samples_sha)
                row.update({k: v for k, v in s.items() if k in ARM_FIELDS})
                w.writerow({k: clean(v) for k, v in row.items()})
        for (b, a), by_scope in pairs.items():
            for scope, _, _ in WINDOWS:
                r = by_scope[scope]
                w.writerow(dict(
                    record_type="pair", model=MODEL, baseline=b, arm=a, scope=scope,
                    quality_ci_excludes_zero=r["quality_ci_low"] > 0 or r["quality_ci_high"] < 0,
                    tpf_ci_excludes_zero=r["tpf_ci_low"] > 0 or r["tpf_ci_high"] < 0,
                    samples_sha256=samples_sha, **{k: clean(v) for k, v in r.items() if k in PAIR_FIELDS}))

    readme = build_readme(meta, arms, pairs, pareto, samples_sha)
    (args.out / "README.md").write_text(readme)
    print(f"wrote {args.out.name}/Gemma.csv and {args.out.name}/README.md")
    print(f"arms={len(MODES)} scopes={len(WINDOWS)} pairs={len(pairs)} "
          f"pareto={'+'.join(pareto)} samples={samples_sha[:12]}")


if __name__ == "__main__":
    main()
