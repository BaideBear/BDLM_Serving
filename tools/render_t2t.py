"""Render sanitized frozen30 T2T aggregate summaries.

Inputs are the cluster-published ``*-online-summary.csv`` files, optionally
with ``*-online-requests.csv`` / ``*-online-phases.csv`` and a sanitized
``run.json`` manifest.  The renderer never loads weights, traces, prompts,
answers or a scorer, and it never reads raw cluster request JSON.

Only complete 30/30 arms contribute comparisons.  Pending arms declared by the
manifest are shown with their measured denominator; missing values are not
zero.  The native row is a full-width reference, never an equal-budget mate.

--input is a publication directory or a single published CSV (repeatable).
--manifest is an optional run.json with model, block, policies and arms.
--fig-dir optionally writes one PNG and one SVG per model on the cluster side.
"""
import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Optional


DATASETS = ("gsm8k", "math500", "humaneval")
MODES = ("targeted", "random", "near_miss", "gate_random", "noedit", "fresh_random")
CONFIG = "model policy nominal_q fresh mask edit".split()
COUNTS = "N F Q physical_Q correct failed truncated".split()
RATIOS = "tpf token_efficiency q_per_f".split()
SCHEMA = {
    "requests": CONFIG + ["index", "dataset", "task_id", "max_q"] + COUNTS,
    "summary": CONFIG + ["dataset", "requests", "max_q"] + COUNTS + RATIOS,
    "phases": "policy phase F Q".split(),
}
POLICY = re.compile(r"t2t_q(\d+)_f(\d+)_m(\d+)_(\w+)")
EMPTY = "—"
GROUPS = ("all",) + DATASETS
FULL = 30


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path, columns):
    with path.open(encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src, strict=True)
        headers = reader.fieldnames or []
        missing = set(columns) - set(headers)
        require(not missing, f"{path.name} 缺列：{', '.join(sorted(missing))}")
        require(len(headers) == len(set(headers)), f"{path.name} 表头重复")
        for row in reader:
            require(None not in row and None not in row.values() and row["policy"],
                    f"{path.name}:{reader.line_num} 行不完整")
            yield row


def _int(row, key):
    text = str(row[key]).strip()
    require(text != "", f"{key} 为空")
    return int(text)


def _float(row, key):
    text = str(row[key]).strip()
    require(text != "", f"{key} 为空")
    value = float(text)
    require(math.isfinite(value), f"{key} 非有限数")
    return value


@dataclass(frozen=True)
class Config:
    q: int
    fresh: int
    mask: int
    edit: int
    mode: str

    @property
    def split(self):
        if self.mode == "native":
            return EMPTY
        return f"{self.fresh}/{self.mask}/{self.edit}"

    @property
    def role(self):
        return "全宽参考" if self.mode == "native" else "等预算"

    @property
    def order(self):
        return (self.mode != "native", self.q, self.fresh, self.mask, self.edit,
                MODES.index(self.mode) if self.mode in MODES else -1)

    @property
    def group(self):
        return (self.q, self.fresh, self.mask, self.edit)


def config(row, policy):
    q = int(row["nominal_q"])
    require(q > 0, "名义 Q 非正")
    if policy == "native":
        require(not any(str(row[k]).strip() for k in ("fresh", "mask", "edit")),
                "native 席位非空")
        return Config(q, 0, 0, 0, "native")
    m = POLICY.fullmatch(policy)
    require(m is not None and m[4] in MODES, f"不支持的策略：{policy}")
    f, mask, e = (_int(row, k) for k in ("fresh", "mask", "edit"))
    require(tuple(map(int, m.groups()[:3])) == (q, f, mask) and
            f + mask + e == q and e >= 0 and (e == 0) == (m[4] in ("noedit", "fresh_random")),
            "策略名与席位分配不符")
    return Config(q, f, mask, e, m[4])


def config_from_policy(policy):
    m = POLICY.fullmatch(policy)
    require(m is not None and m[4] in MODES, f"不支持的策略：{policy}")
    q, f, mask = map(int, m.groups()[:3])
    e = q - f - mask
    require(q > 0 and f >= 0 and mask >= 0 and e >= 0 and
            (e == 0) == (m[4] in ("noedit", "fresh_random")), "策略名与席位分配不符")
    return Config(q, f, mask, e, m[4])


def request_totals(rows):
    out = {k: sum(int(r[k]) for r in rows) for k in COUNTS}
    return dict(out, requests=len(rows), max_q=max(int(r["max_q"]) for r in rows),
                tpf=mean(int(r["N"]) / int(r["F"]) for r in rows),
                token_efficiency=out["N"] / out["Q"], q_per_f=out["Q"] / out["F"])


@dataclass(frozen=True)
class Metrics:
    correct: int
    tpf: float
    efficiency: float
    q_per_f: float


@dataclass
class Arm:
    cfg: Config
    metrics: Optional[Metrics]
    measured: int
    complete: bool
    dataset_counts: dict
    keys: Optional[frozenset] = None

    @property
    def status(self):
        if self.metrics is None:
            return "待发布（无聚合摘要）"
        if self.complete:
            return "完整 30/30"
        counts = "/".join(str(self.dataset_counts.get(d, 0)) for d in DATASETS)
        return f"未完成 {self.measured}/30（{counts}）"

    @property
    def short_status(self):
        if self.metrics is None:
            return "待发布（无聚合摘要）"
        if self.complete:
            return "完整 30/30"
        return f"未完成 {self.measured}/30"

    @property
    def role(self):
        if self.cfg.mode == "native":
            return "全宽参考"
        if self.metrics is None:
            return "待发布"
        if not self.complete:
            return "等预算（未完成）"
        return "等预算"


def _compare(expected, actual, label):
    if isinstance(expected, int):
        require(int(actual) == expected, f"{label} 与逐题记录不符")
    else:
        require(math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-12),
                f"{label} 与逐题记录不符")


def check_summary(model, policy, summary, requests=None, phases=None):
    require(bool(summary), "缺聚合摘要记录")
    cfg = config(summary[0], policy)
    for row in summary:
        require(row["model"] == model and config(row, policy) == cfg,
                "模型或席位元数据不一致")

    groups = defaultdict(list)
    for row in summary:
        ds = row["dataset"]
        require(ds in GROUPS, f"未知数据集：{ds}")
        groups[ds].append(row)
    require(Counter(groups.keys()) == Counter(GROUPS), "缺汇总分组或分组重复")
    require(all(len(rows) == 1 for rows in groups.values()), "汇总分组重复")

    parsed = {}
    for ds in GROUPS:
        row = groups[ds][0]
        measured = _int(row, "requests")
        cap = FULL if ds == "all" else 10
        require(0 < measured <= cap, f"summary[{ds}].requests 超出 {cap}")
        values = {k: _int(row, k) for k in COUNTS}
        require(values["F"] > 0 and values["Q"] > 0 and values["N"] >= 0,
                f"summary[{ds}] 的 N/F/Q 无效")
        require(values["Q"] >= values["F"], f"summary[{ds}].Q 小于 F")
        require(values["physical_Q"] >= values["Q"],
                f"summary[{ds}].physical_Q 小于 Q")
        require(values["correct"] <= measured and values["failed"] <= measured and
                values["truncated"] <= measured and
                values["correct"] + values["failed"] <= measured,
                f"summary[{ds}] 的题数计数无效")
        max_q = _int(row, "max_q")
        require(max_q > 0, f"summary[{ds}].max_q 非正")
        tpf = _float(row, "tpf")
        efficiency = _float(row, "token_efficiency")
        q_per_f = _float(row, "q_per_f")
        require(tpf >= 0 and efficiency >= 0 and q_per_f >= 0,
                f"summary[{ds}] 的比值为负")
        require(math.isclose(efficiency, values["N"] / values["Q"],
                             rel_tol=1e-9, abs_tol=1e-12),
                f"summary[{ds}].token_efficiency 与 N/Q 不符")
        require(math.isclose(q_per_f, values["Q"] / values["F"],
                             rel_tol=1e-9, abs_tol=1e-12),
                f"summary[{ds}].q_per_f 与 Q/F 不符")
        parsed[ds] = dict(values, requests=measured, max_q=max_q, tpf=tpf,
                          efficiency=efficiency, q_per_f=q_per_f)

    all_rows = parsed["all"]
    for key in ["requests", *COUNTS]:
        require(all_rows[key] == sum(parsed[ds][key] for ds in DATASETS),
                f"summary[all].{key} 与数据集汇总不符")
    require(all_rows["max_q"] == max(parsed[ds]["max_q"] for ds in DATASETS),
            "summary[all].max_q 与数据集最大值不符")

    complete = (all_rows["requests"] == FULL and
                all(parsed[ds]["requests"] == 10 for ds in DATASETS))
    keys = None

    if requests is not None:
        require(bool(requests), "逐题记录为空")
        require(len(requests) == all_rows["requests"],
                "逐题记录数与聚合摘要不符")
        seen_keys, seen_indices, by_dataset = set(), set(), defaultdict(list)
        for row in requests:
            require(row["model"] == model and config(row, policy) == cfg,
                    "逐题模型或席位元数据不一致")
            idx, ds, tid = _int(row, "index"), row["dataset"], row["task_id"].strip()
            require(idx not in seen_indices and ds in DATASETS and tid,
                    "题号重复或题目标识无效")
            seen_indices.add(idx)
            seen_keys.add((idx, ds, tid))
            values = {k: _int(row, k) for k in COUNTS + ["max_q"]}
            require(min(values.values()) >= 0 and values["F"] > 0 and
                    values["Q"] > 0 and values["Q"] >= values["F"] and
                    values["physical_Q"] >= values["Q"] and
                    all(values[k] in (0, 1) for k in
                        ("correct", "failed", "truncated")),
                    "逐题计数或分母无效")
            by_dataset[ds].append(row)
        require(set(by_dataset) == set(DATASETS), "逐题文件缺数据集")
        for ds in DATASETS:
            require(len(by_dataset[ds]) == parsed[ds]["requests"],
                    f"逐题 {ds} 条数与聚合摘要不符")
            actual = request_totals(by_dataset[ds])
            for key in ["requests", *COUNTS, "max_q", *RATIOS]:
                expected = parsed[ds][{"token_efficiency": "efficiency"}.get(
                    key, key)]
                _compare(expected, actual[key], f"summary[{ds}].{key}")
        actual_all = request_totals(requests)
        for key in ["requests", *COUNTS, "max_q", *RATIOS]:
            expected = all_rows[{"token_efficiency": "efficiency"}.get(key, key)]
            _compare(expected, actual_all[key], f"summary[all].{key}")
        keys = frozenset(seen_keys)

    if phases is not None:
        require(bool(phases), "阶段记录为空")
        names = [str(p["phase"]).strip() for p in phases]
        require(all(names) and len(set(names)) == len(names), "阶段名为空或重复")
        for key in ("F", "Q"):
            counts = [int(p[key]) for p in phases]
            require(min(counts) >= 0 and sum(counts) == all_rows[key],
                    f"阶段合计 {key} 与聚合摘要不符")

    metrics = Metrics(all_rows["correct"], all_rows["tpf"],
                      all_rows["efficiency"], all_rows["q_per_f"])
    return Arm(cfg, metrics, all_rows["requests"], complete,
               {ds: parsed[ds]["requests"] for ds in DATASETS}, keys)


@dataclass
class Publication:
    model: str
    arms: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    manifest: bool = False
    block: Optional[int] = None

    @property
    def has_requests(self):
        return any(arm.keys is not None for arm in self.arms.values())


def read_manifest(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} 无法读取：{exc}") from exc
    require(isinstance(data, dict), f"{path.name} 不是 JSON 对象")
    model = data.get("model")
    require(isinstance(model, str) and model.strip(), f"{path.name} 缺 model")
    block = data.get("block", data.get("full_width"))
    if block is not None:
        block = int(block)
        require(block > 0, f"{path.name} 的 block 非正")
    raw_arms, policies = data.get("arms"), data.get("policies")
    expected = {}
    if raw_arms is not None:
        require(isinstance(raw_arms, list), f"{path.name} 的 arms 不是列表")
        for item in raw_arms:
            require(isinstance(item, dict), f"{path.name} 的 arm 不是对象")
            try:
                q = int(item["budget"])
                f, mask = int(item["fresh"]), int(item["mask"])
                mode = str(item.get("selection", item.get("mode", "")))
                require(mode in MODES, f"arm 模式无效：{mode}")
                arm_policy = f"t2t_q{q}_f{f}_m{mask}_{mode}"
                cfg = config_from_policy(arm_policy)
            except (KeyError, TypeError) as exc:
                raise ValueError(f"{path.name} 的 arm 无效：{exc}") from exc
            require(arm_policy not in expected,
                    f"{path.name} 的策略重复：{arm_policy}")
            expected[arm_policy] = cfg
        if policies is not None:
            require(list(policies) == list(expected),
                    f"{path.name} 的 policies 与 arms 顺序不一致")
    elif policies is not None:
        require(isinstance(policies, list), f"{path.name} 的 policies 不是列表")
        for item in policies:
            arm_policy = str(item)
            require(arm_policy not in expected,
                    f"{path.name} 的策略重复：{arm_policy}")
            expected[arm_policy] = config_from_policy(arm_policy)
    else:
        raise ValueError(f"{path.name} 缺 arms 或 policies")
    return model, block, expected


def load_model(model, files, manifest=None):
    pub = Publication(model)
    if "summary" not in files:
        pub.issues.append("缺聚合摘要文件")
        return pub
    groups = {}
    try:
        for kind, columns in SCHEMA.items():
            if kind not in files:
                continue
            groups[kind] = defaultdict(list)
            for row in read_csv(files[kind], columns):
                groups[kind][row["policy"]].append(row)
    except (csv.Error, ValueError, OSError) as exc:
        pub.issues.append(str(exc))
        return pub

    summary_policies = set(groups["summary"])
    if not summary_policies:
        pub.issues.append("聚合摘要为空")
        return pub
    for kind in ("requests", "phases"):
        extra = set(groups.get(kind, {})) - summary_policies
        if extra:
            pub.issues.append(f"{kind} 含摘要外策略：{', '.join(sorted(extra))}")

    expected = {}
    if manifest is not None:
        pub.manifest = True
        try:
            m_model, block, expected = read_manifest(manifest)
        except ValueError as exc:
            pub.issues.append(str(exc))
        else:
            pub.block = block
            if m_model != model:
                pub.issues.append(f"清单模型 {m_model} 与摘要 {model} 不一致")
                expected = {}

    for policy in sorted(summary_policies):
        try:
            pub.arms[policy] = check_summary(
                model, policy, groups["summary"][policy],
                groups.get("requests", {}).get(policy),
                groups.get("phases", {}).get(policy),
            )
        except ValueError as exc:
            pub.issues.append(f"{policy}：{exc}")

    for policy, cfg in expected.items():
        if policy in pub.arms or policy in summary_policies:
            continue
        pub.arms[policy] = Arm(cfg, None, 0, False,
                               {ds: 0 for ds in DATASETS})
    if expected:
        for policy in sorted(summary_policies - set(expected) - {"native"}):
            pub.issues.append(f"{policy}：聚合摘要臂不在运行清单")

    native = pub.arms.get("native")
    if native is None:
        pub.issues.append("缺全宽 native 参考；不计算任何对 native 的差值")
    else:
        if native.metrics is None:
            pub.issues.append("native 没有聚合数值")
        if pub.block is not None and native.cfg.q != pub.block:
            pub.issues.append(
                f"native 名义 Q={native.cfg.q} 与清单 block={pub.block} 不符")
        if native.keys is not None:
            for policy, arm in pub.arms.items():
                if (policy != "native" and arm.keys is not None and
                        arm.keys != native.keys):
                    pub.issues.append(f"{policy}：逐题标识与 native 不匹配")
    if not any(a.complete and a.cfg.mode != "native" for a in pub.arms.values()):
        pub.issues.append("无完整 30/30 稀疏臂")
    return pub


def discover(inputs, models=None, manifests=None):
    bundles = defaultdict(lambda: {"files": {}, "parents": set()})
    for raw in inputs:
        path = Path(raw)
        require(path.exists(), f"输入不存在：{path}")
        candidates = [path] if path.is_file() else sorted(path.rglob("*-online-*.csv"))
        for candidate in candidates:
            match = re.fullmatch(r"(.+)-online-(summary|requests|phases)\.csv",
                                 candidate.name)
            if not match:
                continue
            model, kind = match[1], match[2]
            if models and model not in models:
                continue
            bundle = bundles[model]
            require(not bundle["files"] or
                    next(iter(bundle["parents"])) == candidate.resolve().parent,
                    f"{model} 出现在多个目录；请指定同一批次，不跨批次拼表")
            bundle["files"][kind] = candidate.resolve()
            bundle["parents"].add(candidate.resolve().parent)

    attached = {}
    for raw in inputs:
        path = Path(raw)
        if not path.is_dir() or not (path / "run.json").is_file():
            continue
        try:
            model, _, _ = read_manifest(path / "run.json")
        except ValueError:
            continue
        if (models and model not in models) or model not in bundles:
            continue
        require(model not in attached, f"{model} 有多个 run.json；请显式指定清单")
        attached[model] = path / "run.json"

    for raw in manifests or ():
        path = Path(raw)
        require(path.is_file(), f"清单不存在：{path}")
        model, _, _ = read_manifest(path)
        if models and model not in models:
            continue
        require(model not in attached or
                Path(attached[model]).resolve() == path.resolve(),
                f"{model} 有多个清单；请只指定一个")
        attached[model] = path
        bundles.setdefault(model, {"files": {}, "parents": set()})

    names = sorted(set(models or bundles))
    require(names, "未找到 *-online-summary.csv 或清单模型")
    return [load_model(model, bundles.get(model, {}).get("files", {}),
                       attached.get(model)) for model in names]


def paired(current, reference):
    if (current is None or reference is None or
            current.metrics is None or reference.metrics is None):
        return None
    if not (current.complete and reference.complete):
        return None
    if (current.keys is not None and reference.keys is not None and
            current.keys != reference.keys):
        return None
    return (
        current.metrics.correct - reference.metrics.correct,
        current.metrics.tpf - reference.metrics.tpf,
        current.metrics.efficiency - reference.metrics.efficiency,
        current.metrics.q_per_f - reference.metrics.q_per_f,
    )


def basis(current, reference):
    if current.keys is not None and reference.keys is not None:
        return "逐题标识"
    return "摘要计数"


def fmt(values, denominator=None, delta=False):
    if values is None:
        return [EMPTY] * 4
    if delta:
        return [f"{values[0]:+d}"] + [f"{v:+.4f}" for v in values[1:]]
    require(denominator is not None, "非差值格式缺分母")
    return [f"{values[0]}/{denominator}"] + [f"{v:.4f}" for v in values[1:]]


def metric_values(arm):
    if arm.metrics is None:
        return None
    return (arm.metrics.correct, arm.metrics.tpf,
            arm.metrics.efficiency, arm.metrics.q_per_f)


def table(lines, header, rows):
    def line(cells):
        text = " | ".join(str(c).replace("|", "\\|").replace("\n", " ")
                          for c in cells)
        return f"| {text} |"
    lines.extend([line(header), line(["---"] * len(header))])
    for row in rows:
        require(len(row) == len(header), "表格指标列数与表头不一致")
        lines.append(line(row))
    lines.append("")


def controls(pub):
    by_cfg = {arm.cfg: arm for arm in pub.arms.values()}
    yielded = set()
    for arm in sorted(pub.arms.values(), key=lambda item: item.cfg.order):
        cfg = arm.cfg
        if cfg.mode == "noedit":
            reference = by_cfg.get(Config(cfg.q, cfg.fresh, cfg.mask, 0, "fresh_random"))
            delta = paired(arm, reference)
            if delta is not None:
                yield [cfg.q, cfg.split, "noedit", f"fresh_random ({cfg.split})",
                       basis(arm, reference), *fmt(delta, delta=True)]
            continue
        if cfg.mode not in ("targeted", "near_miss") or cfg.edit <= 0:
            continue
        noedit = Config(cfg.q, cfg.fresh, cfg.q - cfg.fresh, 0, "noedit")
        specs = ([("random", Config(cfg.q, cfg.fresh, cfg.mask, cfg.edit, "random")),
                  ("noedit", noedit)] if cfg.mode == "targeted" else
                 [("gate_random", Config(cfg.q, cfg.fresh, cfg.mask, cfg.edit,
                                         "gate_random")),
                  ("targeted", Config(cfg.q, cfg.fresh, cfg.mask, cfg.edit,
                                      "targeted")),
                  ("noedit", noedit)])
        for label, ref_cfg in specs:
            reference = by_cfg.get(ref_cfg)
            key = (cfg, ref_cfg)
            if reference is None or key in yielded:
                continue
            yielded.add(key)
            delta = paired(arm, reference)
            if delta is None:
                continue
            yield [cfg.q, cfg.split, cfg.mode,
                   f"{label} ({ref_cfg.split})", basis(arm, reference),
                   *fmt(delta, delta=True)]


def markdown(pubs):
    lines = [
        "# 冻结 30 题：T2T 在线聚合摘要", "",
        "数据来自集群发布的聚合摘要；不含原始轨迹、题面、答案、权重或集群路径。"
        "质量为答对题数/实际测量题数：完整臂固定为 /30，未完成臂保留其已测分母。"
        "TPF 为逐题 N/F 的算术均值；效率为 ΣN/ΣQ；观测 Q/F 为 ΣQ/ΣF，不是名义预算。"
        "失败和截断不剔除。", "",
        "全宽 native 是整块参考，不是等预算对照。只有双方均完整 30/30 时才计算 Δ；"
        "Δ = 当前 − 对照，质量差以题计，其余为原单位差。未完成或待发布臂不推断数值，"
        "也不与完整臂比较。", "",
        "若输入含逐题请求文件，按同模型的 (index, dataset, task_id) 对齐；只有聚合摘要时，"
        "按各数据集 10×3 的计数核对完整 30 题，不能验证题目标识。", "",
    ]
    metrics = ["质量（正确/测量）", "TPF", "效率（N/Q）", "观测 Q/F"]
    diffs = ["Δ质量（题）", "ΔTPF", "Δ效率", "Δ观测 Q/F"]
    for pub in pubs:
        lines.extend([f"## {pub.model}", ""])
        if pub.issues:
            lines.append("**注意**")
            lines.extend(f"- {issue}" for issue in pub.issues)
            lines.append("")

        ordered = sorted(pub.arms.values(), key=lambda item: item.cfg.order)
        native = next((arm for arm in ordered if arm.cfg.mode == "native"), None)
        sparse = [arm for arm in ordered if arm.cfg.mode != "native"]

        if native is not None and native.metrics is not None:
            lines.extend(["**全宽原生参考（非等预算）**", ""])
            table(lines, ["角色", "名义 Q", *metrics, "状态"],
                  [[native.role, native.cfg.q,
                    *fmt(metric_values(native), native.measured), native.status]])
        elif native is not None:
            lines.extend(["native 聚合数值缺失；不计算任何对 native 的差值。", ""])
        else:
            lines.extend(["缺全宽 native 参考；不计算任何对 native 的差值。", ""])

        if sparse:
            lines.extend([
                "**等预算臂、未完成臂与待发布臂**", "",
                "同预算指选位轮的名义 Q 相同；随机排名对照保留原配额，noedit 把 edit 席位归还 MASK。实际 Q/F 另列。",
                "未完成臂的数值只覆盖已测请求，不能用它补成 /30。", "",
            ])
            rows = []
            for arm in sparse:
                delta = paired(arm, native)
                rows.append([
                    arm.role, arm.cfg.q, arm.cfg.split, arm.cfg.mode,
                    *fmt(metric_values(arm), arm.measured),
                    arm.status, *fmt(delta, delta=True),
                ])
            if native is None:
                delta_header = "Δ（缺 native 参考）"
            elif native.metrics is None:
                delta_header = "Δ（native 数值缺失）"
            else:
                delta_header = "Δ vs 全宽 native（参考，非等预算）"
            lines.extend([delta_header, ""])
            table(lines, ["角色", "名义 Q", "席位 fresh/mask/edit", "模式", *metrics,
                          "状态", *diffs], rows)

        comparisons = list(controls(pub))
        lines.extend(["**同预算配对（仅双方完整 30/30）**", ""])
        if comparisons:
            table(lines, ["名义 Q", "席位 fresh/mask/edit", "当前模式",
                          "对照模式（席位）", "对齐依据", *diffs], comparisons)
        else:
            lines.extend(["当前没有双方均完整 30/30 的同预算配对；"
                          "未完成或待发布臂见上表状态。", ""])

        pending = [arm for arm in sparse if not arm.complete]
        if pending:
            lines.extend([
                "**未完成/待发布清单**：" + "；".join(
                    f"Q={arm.cfg.q} {arm.cfg.split} {arm.cfg.mode} "
                    f"{arm.short_status}" for arm in pending), "",
            ])
    return "\n".join(lines).rstrip() + "\n"


def figure_rows(pub):
    rows = []
    for arm in sorted(pub.arms.values(), key=lambda item: item.cfg.order):
        if arm.cfg.mode == "native":
            label = f"Native (full-width Q={arm.cfg.q})"
        else:
            label = f"Q={arm.cfg.q}  {arm.cfg.split}  {arm.cfg.mode}"
        label += f"  ·  {arm.short_status}"
        values = metric_values(arm) if arm.complete else None
        if values is None:
            annotations = [arm.short_status, "", "", ""]
        else:
            annotations = [f"{values[0]}/{arm.measured}"] + [
                f"{value:.4f}" for value in values[1:]]
        rows.append(dict(label=label, role=arm.role, values=values,
                         annotations=annotations, group=arm.cfg.group))
    return rows


def figures(pubs, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    headings = ["Correct (measured denominator)", "Mean tokens / forward",
                "Tokens / query row", "Observed rows / forward"]
    colors = {"全宽参考": "#333333", "等预算": "#236d96",
              "等预算（未完成）": "#9aa6ad", "待发布": "#d9d9d9"}
    for pub in pubs:
        rows = figure_rows(pub)
        for row in rows:
            # Cluster matplotlib installations need no optional CJK font.
            def en(s):
                return s.replace("待发布（无聚合摘要）", "pending").replace("未完成", "partial").replace("完整", "complete")
            row["label"] = en(row["label"])
            row["annotations"] = [en(text) for text in row["annotations"]]
        if not rows:
            continue
        fig, axes = plt.subplots(1, 4, sharey=True,
                                 figsize=(14, 2.4 + 0.42 * len(rows)),
                                 constrained_layout=True)
        try:
            for j, ax in enumerate(axes):
                values = [row["values"][j] if row["values"] else 0 for row in rows]
                ax.barh(range(len(rows)), values,
                        color=[colors[row["role"]] for row in rows], height=.64)
                peak = 30 if j == 0 else (max(values) or 1)
                ax.set_xlim(0, peak * 1.35)
                ax.set_title(headings[j], fontsize=11)
                for i, row in enumerate(rows):
                    value = values[i]
                    ax.annotate(row["annotations"][j], (value, i),
                                xytext=(4, 0), textcoords="offset points",
                                va="center", fontsize=9)
                    if i and row["group"] != rows[i - 1]["group"]:
                        ax.axhline(i - .5, color=".88", linewidth=.7)
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
            axes[0].set_yticks(range(len(rows)))
            axes[0].set_yticklabels([row["label"] for row in rows], fontsize=9)
            axes[0].invert_yaxis()
            fig.suptitle(
                f"{pub.model} | Frozen30 online aggregate summaries\n"
                "Only complete 30/30 arms are drawn; native is a full-width "
                "reference, not an equal-budget mate", fontsize=12)
            for ext in ("png", "svg"):
                dest = directory / f"{pub.model}-online.{ext}"
                fig.savefig(dest, dpi=160, bbox_inches="tight",
                            metadata={"Date": None} if ext == "svg" else {})
                if ext == "svg":
                    dest.write_text("\n".join(s.rstrip() for s in dest.read_text().splitlines()) + "\n")
        finally:
            plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="published CSV or directory; repeat for disjoint sources")
    parser.add_argument("--manifest", type=Path, action="append",
                        help="sanitized run.json; repeat for multiple models")
    parser.add_argument("--out", type=Path, required=True,
                        help="output Markdown file, or - for stdout")
    parser.add_argument("--model", action="append",
                        help="expected model; repeat to report missing models too")
    parser.add_argument("--fig-dir", type=Path,
                        help="optional cluster-side PNG/SVG directory (matplotlib)")
    args = parser.parse_args(argv)
    if str(args.out) != "-" and args.out.suffix.lower() != ".md":
        parser.error("--out must be a .md file or -")
    try:
        pubs = discover(args.input, args.model, args.manifest)
        text = markdown(pubs)
        if str(args.out) == "-":
            sys.stdout.write(text)
        else:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
        if args.fig_dir:
            figures(pubs, args.fig_dir)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
