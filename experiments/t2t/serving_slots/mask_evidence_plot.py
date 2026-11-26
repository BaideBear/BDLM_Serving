"""Build one standalone SVG from four required mask controls and two optional fixed-offer CPU pilots.

    python mask_evidence_plot.py CONTROLS mask-evidence.svg
    python mask_evidence_plot.py CONTROLS --check

With no output path, write SVG to stdout. No traces, checkpoints or network needed.
"""
import argparse
import csv
from dataclasses import dataclass
import json
from math import ceil, isclose, isfinite
from pathlib import Path
import re
import sys
from typing import Optional
from xml.etree import ElementTree as ET


OBS = "llada21-obs30-mask-paired.csv"
ACTIVE = "gemma-mask-cpu-pilot3.json"
REPLAY = "gemma-unified-mask-pilot3.csv"
LLADA = "llada21-all-slots-mask-pilot3.json"
Q512 = "gemma-unified-q512-mask-pilot3.csv"
ADMISSION = "llada21-q64-admission-mask-pilot3.json"
DATASETS = {0: "gsm8k", 10: "math500", 20: "humaneval"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Arm:
    cases: int
    correct: int
    n: int
    f: int
    q: int                 # physical, never raw request rows
    tpf: Optional[float]   # mean of individual N/F; undefined for an empty subset

    @property
    def eff(self):
        return self.n / self.q if self.q else None


def arm(cases, correct, n, f, q, tpf, eff):
    counts = tuple(map(int, (cases, correct, n, f, q)))
    if counts[0] == 0:
        require(not any(counts) and tpf == "" and eff == "", "Empty scope needs zero counts and blank rates")
        return Arm(*counts, None)
    a = Arm(*counts, float(tpf))
    require(0 <= a.correct <= a.cases and a.cases > 0 and a.n >= 0
            and 0 < a.f <= a.q and isfinite(a.tpf) and a.tpf >= 0, "Invalid work or quality")
    require(isclose(a.eff, float(eff)), "Efficiency differs from N / physical_Q")
    require(a.cases != 1 or isclose(a.tpf, a.n / a.f), "Single-case TPF differs from N/F")
    return a


def pool(arms):
    """Only used within one arm of one CPU executor."""
    arms = tuple(arms)
    totals = [sum(getattr(a, k) for a in arms) for k in ("cases", "correct", "n", "f", "q")]
    return Arm(*totals, sum(a.tpf * a.cases for a in arms if a.cases) / totals[0] if totals[0] else None)


@dataclass(frozen=True)
class Pair:
    label: str
    detail: str
    off: Arm
    on: Arm
    ci: tuple = ()         # published paired accuracy difference, percentage points


def historical(rows):
    windows = {}
    for r in rows:
        if r["scope"] != "all":
            continue       # dataset rows are subsets, not extra observations
        w = int(r["optimus_window"])
        require((r["model"], r["cohort"], r["requests"], r["baseline_policy"], r["arm_policy"],
                 r["efficiency_denominator"], r["exposure_recorded"]) ==
                ("llada21", "original-0-29", "30", f"obs_{w}", f"bobs_{w}", "physical_Q", "no")
                and w not in windows, f"Unexpected OBS cohort, arms or duplicate at W{w}")
        off, on = [arm(r["requests"], *(r[f"{side}_{k}"] for k in
                       ("correct", "N", "F", "physical_Q", "macro_tpf", "pooled_token_efficiency")))
                   for side in ("baseline", "arm")]
        win, lose, both, neither = (int(r[k]) for k in ("rescued", "lost", "both_correct", "both_wrong"))
        require(min(win, lose, both, neither) >= 0 and both + win == on.correct
                and both + lose == off.correct and win + lose + both + neither == 30, "Invalid paired counts")
        lo, hi = (100*float(r["quality_delta_ci_" + k]) for k in ("low", "high"))
        delta = (on.correct-off.correct)/30
        require(isclose(delta, float(r["quality_delta"]), abs_tol=1e-12)
                and -100 <= lo <= 100*delta <= hi <= 100, "Invalid paired direction or interval")
        windows[w] = Pair(f"Window {w}", f"{win} rescued / {lose} lost", off, on, (lo, hi))
    require(set(windows) == {4, 8, 16, 32}, "Expected four complete original OBS windows")
    return [windows[w] for w in sorted(windows)]


def active(data):
    require(data["complete"] is True and data["formal_gpu_score"] is False, "Expected complete CPU pilot")
    cases, exposed = {}, 0
    for c in data["cases"]:
        i = c["index"]
        require(DATASETS.get(i) == c["dataset"]
                and i not in cases, "Unexpected or duplicate pilot case")
        arms = []
        for suffix in ("off", "on"):
            r = c["active_w2_mask_" + suffix]
            require(type(r["correct"]) is bool, "Invalid active_w2 correctness")
            arms.append(arm(1, r["correct"], *(r[k] for k in ("output_tokens", "generation_forwards",
                            "generation_physical_padded_rows", "tpf", "token_efficiency"))))
        tick = c["first_intervention_step"]
        if tick is None:
            require(c["no_second_block_exact"] is True and c["identical_output_ids"] is True
                    and arms[0] == arms[1] and c["dual_block_steps_on"] == c["dual_block_steps_off"] == 0,
                    "Untriggered active_w2 pair is not exact")
            detail = "No intervention"
        else:
            require(type(tick) is int and 0 <= tick < min(c["decoder_steps_on"], c["decoder_steps_off"])
                    and min(c["dual_block_steps_on"], c["dual_block_steps_off"]) > 0, "Invalid trigger")
            exposed += 1
            detail = f"First intervention: decoder {tick+1}"
        cases[i] = Pair(c["task_id"], detail, *arms)
    require(set(cases) == {0, 10, 20}, "Expected all three prespecified active_w2 cases")
    rows = [cases[i] for i in sorted(cases)]
    total = Pair("All three", f"{exposed}/3 requests triggered", pool(p.off for p in rows),
                 pool(p.on for p in rows))
    return [total] + rows


def llada(data):
    cases, exposed = {}, set()
    for c in data["cases"]:
        i, tick = c["index"], c["first_tick"]
        require(type(i) is int and i in DATASETS and DATASETS[i] == c["dataset"]
                and i not in cases, "Unexpected or duplicate LLaDA case")
        sides = []
        for name in ("replay_open", "replay"):
            r = c[name]
            n, f, q = (r[k] for k in ("N", "F", "Q"))
            require(all(type(v) is int for v in (n, f, q)) and n >= 0 and 0 < f <= q
                    and type(r["correct"]) is bool, "Invalid LLaDA case ledger")
            sides.append(Arm(1, int(r["correct"]), n, f, q, n/f))
        detail = "Not exposed"
        if tick is not None:  # A zero tick is also an exposure.
            require(type(tick) is int and tick >= 0, "Invalid LLaDA first_tick")
            detail = f"Exposed at offer tick {tick}"
            exposed.add(i)
        else:
            require(c["same_ids"] is True and sides[0] == sides[1], "Unexposed LLaDA pair differs")
        cases[i] = Pair(f"{i}   {c['dataset']}", detail, *sides)
    require(cases.keys() == DATASETS.keys() and data["pairs"] == 3 and data["exposed"] == len(exposed),
            "LLaDA pilot or exposure count incomplete")

    def group(label, indices, detail):
        return Pair(label, detail, *(pool(getattr(cases[i], s) for i in sorted(indices)) for s in ("off", "on")))

    full = group("All three", cases, f"{len(exposed)}/3 requests exposed")
    require(data["correct"] == {"replay_open": full.off.correct, "replay": full.on.correct},
            "LLaDA published arm scores disagree with cases")
    return [full, *(cases[i] for i in sorted(cases)),
            group("Exposed subset", exposed, "first_tick recorded"),
            group("Unexposed subset", cases.keys()-exposed, "first_tick absent")]


def replay(rows, on_arm="replay"):
    require(on_arm in ("replay", "q512"), "Unknown unified CPU protocol")
    parsed = {}
    for r in rows:
        key = (r["arm"], r["scope"])
        require(r["cohort"] == "pilot3" and key not in parsed, "Mixed or duplicate replay strata")
        parsed[key] = arm(*(r[k] for k in ("cases", "correct", "N", "F", "physical_Q", "macro_TPF",
                                           "token_efficiency")))
    scopes = ("all", "exposed", "unexposed")
    arms = (on_arm + "_open", on_arm)  # causal=False is mask OFF
    require(set(parsed) == {(a, s) for a in arms for s in scopes}, "Incomplete or extra replay arm/scope")
    for a in arms:
        full, exp, unexp = [parsed[a, s] for s in scopes]
        total = pool([exp, unexp])
        require(full.cases == 3 and all(getattr(total, k) == getattr(full, k)
                for k in ("cases", "correct", "n", "f", "q")) and isclose(total.tpf, full.tpf),
                "Replay subsets do not reconstruct their arm")
    result = []
    for scope, label in zip(scopes, ("All three", "Exposed subset", "Unexposed subset")):
        off, on = (parsed[a, scope] for a in arms)
        require(off.cases == on.cases, "Different arm sizes in replay scope")
        result.append(Pair(label, f"{off.cases} of 3 requests", off, on))
    return result


def admission(data):
    indices = data["indices"]
    require(data["suite"] == "cpu-pilot3" and indices == [0, 10, 20]
            and all(type(i) is int for i in indices), "Unexpected Q64 admission cohort")
    require(re.fullmatch(r"[0-9a-fA-F]{64}", data["manifest_sha256"]), "Invalid Q64 manifest digest")
    policies, masks = ("eager", "progress80"), ("open", "causal")
    results = data["results"]
    require(set(results) == {f"{p}-{m}" for p in policies for m in masks}, "Expected four Q64 admission arms")
    rows = []
    for policy in policies:
        sides = [results[f"{policy}-{mask}"] for mask in masks]
        arms = []
        for r in sides:
            require(all(type(r[k]) is int and r[k] >= 0 for k in
                        ("requests", "correct", "N", "F", "Q", "opened", "deferred", "exposed"))
                    and r["requests"] == len(indices), f"Invalid {policy} pilot counts")
            require(all(type(r[k]) in (int, float) for k in ("macro_tpf", "token_efficiency")),
                    f"Invalid {policy} pilot rates")
            arms.append(arm(*(r[k] for k in ("requests", "correct", "N", "F", "Q", "macro_tpf", "token_efficiency"))))
        rows.append(Pair(policy, f"Exposed: {sides[0]['exposed']} \u2192 {sides[1]['exposed']}", *arms))
    return rows


def load(controls):
    def csv_rows(name):
        with (controls / name).open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    def read_json(name):
        return json.loads((controls / name).read_text(encoding="utf-8"))
    panels = {OBS: historical(csv_rows(OBS)), ACTIVE: active(read_json(ACTIVE)),
              REPLAY: replay(csv_rows(REPLAY)), LLADA: llada(read_json(LLADA))}
    for name, read, parse in ((Q512, csv_rows, lambda r: replay(r, "q512")), (ADMISSION, read_json, admission)):
        try:
            data = read(name)
        except FileNotFoundError:
            continue
        panels[name] = parse(data)
    return panels


def figure(panels):
    """Pair values read left to right. The effect plot has its own scale per panel."""
    root = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", width="1200", height="1000",
                      viewBox="0 0 1200 1000", role="img", attrib={"aria-labelledby": "title desc"})
    ET.SubElement(root, "title", id="title").text = "Cross-block mask: paired task evidence"
    ET.SubElement(root, "desc", id="desc").text = (
        "Mask off to mask on. Historical LLaDA OBS original30 and separate LLaDA and DiffusionGemma CPU pilots.")
    ink, mute, teal, orange = "#192d36", "#49616b", "#007568", "#a54524"
    def node(tag, **attrs):
        return ET.SubElement(root, tag, {k.replace("_", "-"): str(v) for k, v in attrs.items()})
    def text(x, y, s, size=14, color=ink, weight=400, anchor="start"):
        node("text", x=x, y=y, font_size=size, fill=color, font_weight=weight, text_anchor=anchor,
             font_family="Arial, Helvetica, sans-serif").text = s
    def line(x, x2, y, y2, color, width=1):
        node("line", x1=x, x2=x2, y1=y, y2=y2, stroke=color, stroke_width=width)

    node("rect", width=1200, height=1000, fill="white")
    text(36, 43, "Cross-block mask: paired task evidence", 29, weight=700)
    text(36, 72, "Every pair reads mask OFF \u2192 ON. ON blocks earlier queries from reading later blocks; compare only within each panel.", 16)
    headings = {
        OBS: ("A   LLaDA 2.1-mini  |  Original Optimus OBS", "Historical GPU runs  /  original30: 10 per dataset  /  fixed windows, not hard caps; no trigger split"),
        ACTIVE: ("B   DiffusionGemma  |  active_w2", "CPU pilot3  /  first original30 case per dataset  /  full requests; first intervention is numbered from 1"),
        REPLAY: ("C   DiffusionGemma  |  Unified external replay", "CPU pilot3  /  separate executor  /  replay_open (OFF) \u2192 replay (ON); subsets partition All three"),
        LLADA: ("D   LLaDA 2.1-mini  |  Five-tier external replay", "CPU pilot3  /  offers 4, 8, 16, 32, 64  /  replay_open (OFF) \u2192 replay (ON); exposure subsets partition the same three cases"),
        Q512: ("E   DiffusionGemma  |  Unified fixed Q512", "CPU pilot3  /  fixed offer, not equal work  /  q512_open (OFF) \u2192 q512 (ON); subsets partition All three"),
        ADMISSION: ("F   LLaDA 2.1-mini  |  Fixed Q64 admission \u00d7 mask", "CPU pilot3  /  cases 0, 10, 20  /  open (OFF) \u2192 causal (ON) within each policy; fixed offer, not equal work"),
    }
    require({OBS, ACTIVE, REPLAY, LLADA} <= panels.keys() <= headings.keys(), "Missing or unknown panel")
    y = 108
    for name, (title, sub) in headings.items():
        if name not in panels:
            continue
        rows, obs = panels[name], name == OBS
        node("rect", x=28, y=y-16, width=1144, height=54, fill="#eff4f5")
        text(40, y+4, title, 20, weight=700)
        text(40, y+26, sub, 14, mute)
        row_heading = "Window / paired outcomes" if obs else "Admission policy" if name == ADMISSION else "Request / scope"
        columns = [(40, row_heading), (325, "Correct"),
                   (465, "Macro TPF"), (656, "Physical efficiency"),
                   (877, "Accuracy change (pp)  |  paired 95% CI" if obs else "TPF change  |  ON minus OFF")]
        for x, label in columns:
            text(x, y+62, label, 14, weight=700)
        delta = [None if not p.on.cases else
                 (100*(p.on.correct-p.off.correct)/p.on.cases if obs else p.on.tpf-p.off.tpf) for p in rows]
        extent = max(abs(v) for p, d in zip(rows, delta) if d is not None for v in (d,) + p.ci)
        step = 10 if obs else 1
        scale = max(step, step*ceil(extent/step))
        center, radius = 952, 70
        px = lambda v: center + radius*v/scale
        for v in (-scale, 0, scale):
            text(px(v), y+80, f"{v:+d}" if v else "0", 11, mute, anchor="middle")
        line(center, center, y+85, y+86+42*len(rows), "#b8c8cd")
        for j, (p, d) in enumerate(zip(rows, delta)):
            z = y+105+42*j
            line(38, 1164, z+24, z+24, "#e4eaec")
            text(40, z, p.label, 15, weight=600)
            text(40, z+16, p.detail, 12, mute)
            a, b = p.off, p.on
            if d is None:
                text(325, z+4, "No cases; rates undefined", 14, mute)
                continue   # No delta point or fictitious 0% efficiency.
            for x, l, r in [(325, f"{a.correct}/{a.cases}", f"{b.correct}/{b.cases}"),
                            (465, f"{a.tpf:.3f}", f"{b.tpf:.3f}"),
                            (656, f"{100*a.eff:.2f}%", f"{100*b.eff:.2f}%")]:
                text(x, z+4, f"{l} \u2192 {r}", 15)
            if not obs:
                for x, k, label in ((325, "n", "N"), (465, "f", "F"), (656, "q", "physical Q")):
                    text(x, z+18, f"{label}: {getattr(a, k):,} \u2192 {getattr(b, k):,}", 12, mute)
            lo, hi = p.ci if obs else (0, d)
            color = teal if d > 0 else orange if d < 0 else mute
            line(px(lo), px(hi), z, z, color, 2)
            if obs:
                for v in (lo, hi):
                    line(px(v), px(v), z-4, z+4, color)
            node("circle", cx=px(d), cy=z, r=4, fill=color)
            text(1039, z+4, f"{d:+.2f}", 15, color, 600)
            if obs:
                text(1039, z+19, f"[{lo:+.2f}, {hi:+.2f}]", 12, mute)
        y += 108+42*len(rows)
    text(40, y+10, "TPF = mean request N/F. Efficiency = total output tokens / total physical Q. No cross-panel average or official frozen90 GPU score.", 14)
    text(40, y+33, f"A: {OBS}     B: {ACTIVE}     C: {REPLAY}", 12, mute)
    text(40, y+51, "     ".join(f"{letter}: {name}" for letter, name in
         (("D", LLADA), ("E", Q512), ("F", ADMISSION)) if name in panels), 12, mute)
    height = y+68
    root.set("height", str(height))
    root.set("viewBox", f"0 0 1200 {height}")
    root.find("rect").set("height", str(height))
    return ET.tostring(root, encoding="unicode")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("controls", type=Path)
    p.add_argument("output", nargs="?", type=Path, help="SVG output (default: stdout)")
    p.add_argument("--check", action="store_true", help="validate inputs and render in memory only")
    a = p.parse_args()
    if a.output and (a.check or a.output.suffix.lower() != ".svg"):
        p.error("use --check alone or give an .svg output")
    try:
        panels = load(a.controls)
        svg = figure(panels)
        if a.check:
            ET.fromstring(svg)
            print(f"OK: {len(panels)} panels, including LLaDA five-tier CPU pilot3. "
                  f"Gemma fixed-Q512: {'present' if Q512 in panels else 'absent'}; "
                  f"LLaDA Q64 admission: {'present' if ADMISSION in panels else 'absent'}.", file=sys.stderr)
        elif a.output:
            a.output.write_text(svg + "\n", encoding="utf-8")
        else:
            print(svg)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        p.exit(1, f"mask evidence: {exc}\n")


if __name__ == "__main__":
    main()
