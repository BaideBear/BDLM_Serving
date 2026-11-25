import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from statistics import mean


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import render_t2t as r  # noqa: E402


DATASETS = r.DATASETS
COUNTS = ("N", "F", "Q", "physical_Q", "correct", "failed", "truncated")


def policy(cfg):
    return ("native" if cfg.mode == "native" else
            f"t2t_q{cfg.q}_f{cfg.fresh}_m{cfg.mask}_{cfg.mode}")


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def request_rows(model, cfg, measured=30, correct=None, offset=0):
    correct = correct or {ds: 10 for ds in DATASETS}
    if measured == 30:
        assignment = [ds for ds in DATASETS for _ in range(10)]
    elif measured == 28:
        assignment = ([DATASETS[0]] * 10 + [DATASETS[1]] * 9 +
                      [DATASETS[2]] * 9)
    else:
        assignment = ([ds for ds in DATASETS for _ in range(10)])[:measured]
    rows, used = [], {ds: 0 for ds in DATASETS}
    for index, ds in enumerate(assignment):
        used[ds] += 1
        n = 100 + index
        f = 5 + index % 3
        q = 4 * f
        is_correct = used[ds] <= correct.get(ds, 0)
        rows.append(dict(
            model=model, policy=policy(cfg), nominal_q=cfg.q,
            fresh="" if cfg.mode == "native" else cfg.fresh,
            mask="" if cfg.mode == "native" else cfg.mask,
            edit="" if cfg.mode == "native" else cfg.edit,
            index=index, dataset=ds, task_id=f"{ds}/{offset + index}",
            max_q=cfg.q, N=n, F=f, Q=q, physical_Q=q,
            correct=int(is_correct), failed=0, truncated=0,
        ))
    return rows


def summary_rows(rows):
    first = rows[0]
    out = []
    for group in ("all",) + DATASETS:
        subset = rows if group == "all" else [x for x in rows if x["dataset"] == group]
        n = sum(x["N"] for x in subset)
        f = sum(x["F"] for x in subset)
        q = sum(x["Q"] for x in subset)
        out.append(dict(
            model=first["model"], policy=first["policy"],
            nominal_q=first["nominal_q"], fresh=first["fresh"],
            mask=first["mask"], edit=first["edit"], dataset=group,
            requests=len(subset), correct=sum(x["correct"] for x in subset),
            failed=sum(x["failed"] for x in subset),
            truncated=sum(x["truncated"] for x in subset),
            N=n, F=f, Q=q, physical_Q=sum(x["physical_Q"] for x in subset),
            tpf=mean(x["N"] / x["F"] for x in subset),
            token_efficiency=n / q, q_per_f=q / f,
            max_q=max(x["max_q"] for x in subset),
        ))
    return out


def write_bundle(directory, model, specs, manifest=None, with_requests=False,
                 with_phases=False):
    summary, requests, phases = [], [], []
    for cfg, measured, correct, offset in specs:
        rows = request_rows(model, cfg, measured, correct, offset)
        summary.extend(summary_rows(rows))
        requests.extend(rows)
        if with_phases:
            phases.extend([
                dict(policy=policy(cfg), phase="selected",
                     F=sum(x["F"] for x in rows) - 1,
                     Q=sum(x["Q"] for x in rows) - 4),
                dict(policy=policy(cfg), phase="commit", F=1, Q=4),
            ])
    write_csv(directory / f"{model}-online-summary.csv",
              r.SCHEMA["summary"], summary)
    if with_requests:
        write_csv(directory / f"{model}-online-requests.csv",
                  r.SCHEMA["requests"], requests)
    if with_phases:
        write_csv(directory / f"{model}-online-phases.csv",
                  r.SCHEMA["phases"], phases)
    if manifest is not None:
        (directory / "run.json").write_text(json.dumps(manifest), encoding="utf-8")


def manifest(model, block, arms):
    return dict(
        model=model, block=block,
        policies=[policy(cfg) for cfg in arms],
        arms=[dict(budget=cfg.q, fresh=cfg.fresh, mask=cfg.mask,
                   selection=cfg.mode) for cfg in arms],
    )


def complete_specs():
    native = r.Config(32, 0, 0, 0, "native")
    targeted = r.Config(16, 4, 8, 4, "targeted")
    random = r.Config(16, 4, 8, 4, "random")
    noedit = r.Config(16, 4, 12, 0, "noedit")
    return native, targeted, random, noedit


class RenderTest(unittest.TestCase):
    def test_aggregate_summary_lists_roles_and_pending_without_comparison(self):
        native, targeted, random, noedit = complete_specs()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_bundle(
                directory, "llada21",
                [(native, 30, {ds: 10 for ds in DATASETS}, 0),
                 (targeted, 30, {"gsm8k": 10, "math500": 9, "humaneval": 7}, 0),
                 (random, 28, {"gsm8k": 9, "math500": 8, "humaneval": 7}, 0)],
                manifest=manifest("llada21", 32,
                                  [targeted, random, noedit,
                                   r.Config(8, 2, 4, 2, "targeted")]),
            )
            pubs = r.discover([directory], ["llada21"], [directory / "run.json"])

        pub = pubs[0]
        self.assertEqual(pub.issues, [])
        self.assertFalse(any("native：聚合摘要臂不在运行清单" in issue
                             for issue in pub.issues))
        self.assertEqual(pub.arms[policy(random)].short_status, "未完成 28/30")
        self.assertEqual(pub.arms[policy(noedit)].short_status,
                         "待发布（无聚合摘要）")
        self.assertEqual(pub.arms[policy(r.Config(8, 2, 4, 2, "targeted"))].status,
                         "待发布（无聚合摘要）")
        self.assertEqual(pub.arms[policy(targeted)].role, "等预算")
        self.assertEqual(pub.arms["native"].role, "全宽参考")
        self.assertEqual(list(r.controls(pub)), [])

        text = r.markdown(pubs)
        self.assertIn("全宽原生参考（非等预算）", text)
        self.assertIn("Δ vs 全宽 native（参考，非等预算）", text)
        self.assertIn("26/30", text)  # targeted: 10+9+7
        self.assertIn("24/28", text)  # random: measured subset denominator
        self.assertNotIn("24/30", text)
        self.assertIn("未完成 28/30", text)
        self.assertIn("待发布（无聚合摘要）", text)
        self.assertIn("未完成/待发布清单", text)
        self.assertIn("没有双方均完整 30/30 的同预算配对", text)
        self.assertNotIn("%", text)

        rows = {row["label"]: row for row in r.figure_rows(pub)}
        pending = [row for label, row in rows.items()
                   if "Q=16" in label and "random" in label][0]
        self.assertIsNone(pending["values"])
        self.assertIn("未完成 28/30", pending["label"])
        native_row = [row for label, row in rows.items() if "full-width" in label][0]
        self.assertEqual(native_row["annotations"][0], "30/30")
        self.assertTrue(all(native_row["annotations"]))

    def test_complete_arms_only_in_equal_budget_comparison(self):
        native, targeted, random, noedit = complete_specs()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_bundle(
                directory, "llada21",
                [(native, 30, {ds: 10 for ds in DATASETS}, 0),
                 (targeted, 30, {"gsm8k": 10, "math500": 9, "humaneval": 7}, 0),
                 (random, 30, {"gsm8k": 9, "math500": 8, "humaneval": 8}, 0),
                 (noedit, 30, {"gsm8k": 10, "math500": 9, "humaneval": 7}, 0)],
                manifest=manifest("llada21", 32, [targeted, random, noedit]),
            )
            pubs = r.discover([directory], ["llada21"], [directory / "run.json"])

        pub = pubs[0]
        pairs = list(r.controls(pub))
        self.assertEqual(len(pairs), 2)
        self.assertEqual({row[3].split(" (")[0] for row in pairs},
                         {"random", "noedit"})
        for row in pairs:
            self.assertEqual(row[4], "摘要计数")
            self.assertTrue(all(value != r.EMPTY for value in row[5:]))

        text = r.markdown(pubs)
        self.assertIn("**同预算配对（仅双方完整 30/30）**", text)
        self.assertIn("Δ vs 全宽 native（参考，非等预算）", text)
        self.assertIn("+0", text)
        self.assertNotIn("没有双方均完整 30/30 的同预算配对", text)

        rows = {row["label"]: row for row in r.figure_rows(pub)}
        targeted_row = [row for label, row in rows.items() if "targeted" in label][0]
        self.assertEqual(targeted_row["annotations"][0], "26/30")
        for annotation in targeted_row["annotations"][1:]:
            self.assertRegex(annotation, r"^-?\d+\.\d{4}$")

    def test_request_identity_mismatch_blocks_comparison(self):
        native, targeted, random, _ = complete_specs()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_bundle(
                directory, "llada21",
                [(native, 30, {ds: 10 for ds in DATASETS}, 0),
                 (targeted, 30, {"gsm8k": 10, "math500": 9, "humaneval": 7}, 0),
                 (random, 30, {"gsm8k": 9, "math500": 8, "humaneval": 8}, 500)],
                manifest=manifest("llada21", 32, [targeted, random]),
                with_requests=True,
            )
            pubs = r.discover([directory], ["llada21"], [directory / "run.json"])

        pub = pubs[0]
        self.assertTrue(any("逐题标识与 native 不匹配" in issue
                            for issue in pub.issues))
        self.assertEqual(list(r.controls(pub)), [])
        self.assertIn("逐题标识与 native 不匹配", r.markdown(pubs))

    def test_aggregate_ratio_mismatch_is_rejected(self):
        cfg = r.Config(16, 4, 8, 4, "targeted")
        rows = summary_rows(request_rows("llada21", cfg))
        rows[0]["token_efficiency"] = rows[0]["token_efficiency"] + 0.01
        with self.assertRaisesRegex(ValueError, "token_efficiency"):
            r.check_summary("llada21", policy(cfg), rows)

    def test_missing_native_is_explicit(self):
        _, targeted, _, _ = complete_specs()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_bundle(
                directory, "llada21",
                [(targeted, 30, {"gsm8k": 10, "math500": 9, "humaneval": 7}, 0)],
                manifest=manifest("llada21", 32, [targeted]),
            )
            pubs = r.discover([directory], ["llada21"], [directory / "run.json"])

        text = r.markdown(pubs)
        self.assertIn("缺全宽 native 参考", text)
        self.assertNotIn("Δ vs 全宽 native（参考，非等预算）", text)
        self.assertTrue(all(value == r.EMPTY
                            for value in r.fmt(None, delta=True)))

    def test_cli_stdout_writes_no_report_file(self):
        native, targeted, _, noedit = complete_specs()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_bundle(
                directory, "llada21",
                [(native, 30, {ds: 10 for ds in DATASETS}, 0),
                 (targeted, 30, {"gsm8k": 10, "math500": 9, "humaneval": 7}, 0)],
                manifest=manifest("llada21", 32, [targeted, noedit]),
            )
            output = io.StringIO()
            with redirect_stdout(output):
                r.main(["--input", str(directory),
                        "--manifest", str(directory / "run.json"),
                        "--out", "-"])
            self.assertIn("# 冻结 30 题：T2T 在线聚合摘要", output.getvalue())
            self.assertEqual(list(directory.glob("*.md")), [])


if __name__ == "__main__":
    unittest.main()
