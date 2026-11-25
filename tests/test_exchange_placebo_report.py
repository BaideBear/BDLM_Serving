import csv
import hashlib
import json
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from tools.exchange_placebo_report import (
    ARM_ORDER, REFERENCES, build_report, context_rows, load_manifest, main,
    paired_rows, request_rows, summary_rows, validate_totals,
)


MODEL = "toy"
DATASETS = ("gsm8k", "math500", "humaneval")
TASKS = [(i, DATASETS[i // 10], f"{DATASETS[i // 10]}/{i}") for i in range(30)]


def correct_upto(n):
    return [1 if i < n else 0 for i in range(30)]


def make_rows(q, policy, mask, correct, n_base, f_base):
    rows = []
    for i, (index, dataset, task_id) in enumerate(TASKS):
        f = f_base + (i % 4)
        query = q * f * 6
        rows.append(dict(
            model=MODEL, policy=policy, nominal_q=q, fresh=0, mask=mask,
            edit=q - mask, index=index, dataset=dataset, task_id=task_id,
            max_q=32, N=n_base + 3 * i, F=f, Q=query, physical_Q=query,
            correct=correct[i], failed=0, truncated=0))
    return rows


def summary_row(rows, dataset="all"):
    subset = rows if dataset == "all" else [r for r in rows if r["dataset"] == dataset]
    n = sum(r["N"] for r in subset)
    f = sum(r["F"] for r in subset)
    q = sum(r["Q"] for r in subset)
    physical = sum(r["physical_Q"] for r in subset)
    return dict(
        model=MODEL, policy=rows[0]["policy"], nominal_q=rows[0]["nominal_q"],
        fresh=0, mask=rows[0]["mask"], edit=rows[0]["edit"], dataset=dataset,
        requests=len(subset), correct=sum(r["correct"] for r in subset),
        failed=sum(r["failed"] for r in subset),
        truncated=sum(r["truncated"] for r in subset),
        N=n, F=f, Q=q, physical_Q=physical,
        tpf=mean(r["N"] / r["F"] for r in subset),
        token_efficiency=n / q, q_per_f=q / f,
        max_q=max(r["max_q"] for r in subset))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_arm_dir(path, arms):
    requests, summaries = [], []
    for policy, rows in arms.items():
        requests.extend(rows)
        for dataset in ("all", *DATASETS):
            summaries.append(summary_row(rows, dataset))
    write_csv(path / f"{MODEL}-online-requests.csv", requests)
    write_csv(path / f"{MODEL}-online-summary.csv", summaries)


def write_manifest(path):
    samples = [dict(index=i, dataset=DATASETS[i // 10],
                    task_id=f"{DATASETS[i // 10]}/{i}", seed=17000 + i)
               for i in range(30)]
    path.write_text(json.dumps(dict(samples=samples)))


def fixture(root):
    ranked, measured = root / "ranked", root / "measured"
    write_arm_dir(ranked, {
        "t2t_q4_f0_m4_hard_noedit": make_rows(4, "t2t_q4_f0_m4_hard_noedit", 4,
                                              correct_upto(25), 100, 10),
        "t2t_q4_f0_m3_hard_exchange_edit4": make_rows(4, "t2t_q4_f0_m3_hard_exchange_edit4", 3,
                                                      correct_upto(26), 110, 10),
        "t2t_q8_f0_m8_hard_noedit": make_rows(8, "t2t_q8_f0_m8_hard_noedit", 8,
                                              correct_upto(26), 120, 12),
        "t2t_q8_f0_m7_hard_exchange_edit4": make_rows(8, "t2t_q8_f0_m7_hard_exchange_edit4", 7,
                                                      correct_upto(24), 115, 13),
    })
    write_arm_dir(measured, {
        "t2t_q4_f0_m3_hard_exchange_random_a": make_rows(4, "t2t_q4_f0_m3_hard_exchange_random_a", 3,
                                                         correct_upto(24), 90, 11),
        "t2t_q4_f0_m3_hard_exchange_random_b": make_rows(4, "t2t_q4_f0_m3_hard_exchange_random_b", 3,
                                                         correct_upto(27), 105, 10),
        "t2t_q8_f0_m7_hard_exchange_random_a": make_rows(8, "t2t_q8_f0_m7_hard_exchange_random_a", 7,
                                                         correct_upto(26), 118, 12),
        "t2t_q8_f0_m7_hard_exchange_random_b": make_rows(8, "t2t_q8_f0_m7_hard_exchange_random_b", 7,
                                                         correct_upto(25), 125, 14),
    })
    write_manifest(root / "manifest.json")
    baselines = root / "baselines" / "summary.csv"
    rows = []
    for q, correct in ((4, 18), (8, 23)):
        for mode, marks in (("ibs", correct), ("obs", correct - 1)):
            n, f, query = 15000, 11000, 43000 + 1000 * q
            rows.append(dict(
                model=MODEL, policy=f"{mode}_{q}", role="baseline", run="baselines",
                dataset="all", requests=30, N=n, F=f, Q=query, physical_Q=query,
                correct=marks, failed=0, truncated=0, tpf=n / f,
                efficiency=n / query, q_per_f=query / f,
                physical_efficiency=n / query))
    write_csv(baselines, rows)
    return SimpleNamespace(model=MODEL, qs=[4, 8], manifest=root / "manifest.json",
                           measured=measured, ranked=ranked, baselines=baselines,
                           out=root / "out")


class ExchangePlaceboReportTest(unittest.TestCase):
    def test_metrics_and_paired_deltas(self):
        with tempfile.TemporaryDirectory() as d:
            report = build_report(fixture(Path(d)))
        self.assertEqual(report.qs, [4, 8])
        for q in (4, 8):
            for arm in ARM_ORDER:
                a = report.arms[q][arm]
                s = a.summary
                self.assertEqual(s["requests"], 30)
                self.assertEqual(s["correct"], sum(r["correct"] for r in a.requests))
                self.assertAlmostEqual(s["tpf"], sum(r["N"] / r["F"] for r in a.requests) / 30)
                self.assertAlmostEqual(s["physical_efficiency"],
                                       sum(r["N"] for r in a.requests)
                                       / sum(r["physical_Q"] for r in a.requests))
                self.assertEqual(s["physical_Q"], sum(r["physical_Q"] for r in a.requests))

        def outcome(q, ref):
            p = report.paired[q][ref]
            return (p["only_ranked_correct"], p["only_reference_correct"],
                    p["both_correct"], p["neither_correct"], p["delta_correct"])

        self.assertEqual(outcome(4, "zero"), (1, 0, 25, 4, 1))
        self.assertEqual(outcome(4, "placebo_a"), (2, 0, 24, 4, 2))
        self.assertEqual(outcome(4, "placebo_b"), (0, 1, 26, 3, -1))
        self.assertEqual(outcome(8, "zero"), (0, 2, 24, 4, -2))
        self.assertEqual(outcome(8, "placebo_a"), (0, 2, 24, 4, -2))
        self.assertEqual(outcome(8, "placebo_b"), (0, 1, 24, 5, -1))
        for q in (4, 8):
            for ref in REFERENCES:
                p = report.paired[q][ref]
                self.assertAlmostEqual(p["delta_tpf"], report.arms[q]["ranked"].summary["tpf"]
                                       - report.arms[q][ref].summary["tpf"])
                self.assertEqual(p["delta_Q"], report.arms[q]["ranked"].summary["Q"]
                                 - report.arms[q][ref].summary["Q"])
                self.assertEqual(p["delta_physical_Q"],
                                 report.arms[q]["ranked"].summary["physical_Q"]
                                 - report.arms[q][ref].summary["physical_Q"])

    def test_cli_writes_detail_and_figures(self):
        with tempfile.TemporaryDirectory() as d:
            args = fixture(Path(d))
            main(["--measured", str(args.measured), "--ranked", str(args.ranked),
                  "--out", str(args.out), "--model", MODEL,
                  "--manifest", str(args.manifest),
                  "--baselines", str(args.baselines)])
            for name in ("exchange-placebo-requests.csv", "exchange-placebo-summary.csv",
                         "exchange-placebo-paired.csv", "exchange-placebo-paired-requests.csv",
                         "exchange-placebo-context.csv", "exchange-placebo-checks.csv",
                         "exchange-placebo-study.svg", "exchange-placebo-delta.svg",
                         "exchange-placebo-report.md", "exchange-placebo-run.json"):
                self.assertTrue((args.out / name).is_file(), name)
            requests = list(csv.DictReader((args.out / "exchange-placebo-requests.csv").open()))
            self.assertEqual(len(requests), 240)
            self.assertEqual({r["arm"] for r in requests}, set(ARM_ORDER))
            per_dataset = Counter((r["nominal_q"], r["arm"], r["dataset"]) for r in requests)
            self.assertEqual(set(per_dataset.values()), {10})
            summaries = list(csv.DictReader((args.out / "exchange-placebo-summary.csv").open()))
            self.assertEqual(len(summaries), 8)
            self.assertEqual({r["arm"] for r in summaries}, set(ARM_ORDER))
            self.assertEqual({r["dataset"] for r in summaries}, {"all"})
            paired = list(csv.DictReader((args.out / "exchange-placebo-paired.csv").open()))
            self.assertEqual(len(paired), 6)
            self.assertEqual({r["reference"] for r in paired}, set(REFERENCES))
            self.assertEqual({r["dataset"] for r in paired}, {"all"})
            context = list(csv.DictReader((args.out / "exchange-placebo-context.csv").open()))
            self.assertEqual({r["policy"] for r in context},
                             {"ibs_4", "obs_4", "ibs_8", "obs_8"})
            self.assertEqual({r["dataset"] for r in context}, {"all"})
            self.assertTrue(all(r["reference"] in REFERENCES for r in
                                csv.DictReader((args.out / "exchange-placebo-paired-requests.csv").open())))
            by_policy = {r["policy"]: r for r in summaries}
            ranked = by_policy["t2t_q4_f0_m3_hard_exchange_edit4"]
            svg = (args.out / "exchange-placebo-study.svg").read_text()
            self.assertIn(f"{float(ranked['macro_tpf']):.3f}", svg)
            self.assertIn(f"{int(ranked['physical_Q']):,}", svg)
            self.assertIn("IBS", svg)
            delta = (args.out / "exchange-placebo-delta.svg").read_text()
            self.assertIn("+2", delta)
            self.assertIn("-2", delta)
            report = (args.out / "exchange-placebo-report.md").read_text()
            self.assertIn("不能合并成 60 题", report)
            self.assertIn("不进入任何配对比较", report)
            self.assertIn("IBS/OBS", report)
            run = json.loads((args.out / "exchange-placebo-run.json").read_text())
            self.assertTrue(run["complete"])
            self.assertEqual(run["zero_sources"], {"4": "ranked", "8": "ranked"})

    def test_missing_placebo_fails_without_output(self):
        with tempfile.TemporaryDirectory() as d:
            args = fixture(Path(d))
            requests = list(csv.DictReader((args.measured / f"{MODEL}-online-requests.csv").open()))
            write_csv(args.measured / f"{MODEL}-online-requests.csv",
                      [r for r in requests if not r["policy"].endswith("_random_b")])
            summaries = list(csv.DictReader((args.measured / f"{MODEL}-online-summary.csv").open()))
            write_csv(args.measured / f"{MODEL}-online-summary.csv",
                      [r for r in summaries if not r["policy"].endswith("_random_b")])
            with self.assertRaisesRegex(ValueError, "hard_exchange_random_b"):
                build_report(args)
            self.assertFalse(args.out.exists())

    def test_incomplete_arm_fails(self):
        with tempfile.TemporaryDirectory() as d:
            args = fixture(Path(d))
            path = args.measured / f"{MODEL}-online-requests.csv"
            rows = list(csv.DictReader(path.open()))
            target = "t2t_q4_f0_m3_hard_exchange_random_b"
            kept = [r for r in rows if r["policy"] != target]
            kept += [r for r in rows if r["policy"] == target][:-1]
            write_csv(path, kept)
            with self.assertRaisesRegex(ValueError, "29 requests"):
                build_report(args)

    def test_task_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as d:
            args = fixture(Path(d))
            path = args.measured / f"{MODEL}-online-requests.csv"
            rows = list(csv.DictReader(path.open()))
            next(r for r in rows if r["policy"] == "t2t_q4_f0_m3_hard_exchange_random_b")["task_id"] = "gsm8k/999"
            write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "frozen30"):
                build_report(args)

    def test_summary_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as d:
            args = fixture(Path(d))
            path = args.measured / f"{MODEL}-online-summary.csv"
            rows = list(csv.DictReader(path.open()))
            row = next(r for r in rows if r["policy"] == "t2t_q4_f0_m3_hard_exchange_random_b"
                       and r["dataset"] == "all")
            row["correct"] = str(int(row["correct"]) + 1)
            write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                build_report(args)

    def test_zero_falls_back_to_hard(self):
        with tempfile.TemporaryDirectory() as d:
            args = fixture(Path(d))
            req_path = args.ranked / f"{MODEL}-online-requests.csv"
            sum_path = args.ranked / f"{MODEL}-online-summary.csv"
            requests = list(csv.DictReader(req_path.open()))
            summaries = list(csv.DictReader(sum_path.open()))
            zero = {"t2t_q4_f0_m4_hard_noedit", "t2t_q8_f0_m8_hard_noedit"}
            write_csv(Path(d) / "hard" / f"{MODEL}-online-requests.csv",
                      [r for r in requests if r["policy"] in zero])
            write_csv(Path(d) / "hard" / f"{MODEL}-online-summary.csv",
                      [r for r in summaries if r["policy"] in zero])
            write_csv(req_path, [r for r in requests if r["policy"] not in zero])
            write_csv(sum_path, [r for r in summaries if r["policy"] not in zero])
            report = build_report(args)
            self.assertEqual(report.arms[4]["zero"].source, "hard")
            self.assertEqual(report.arms[8]["zero"].source, "hard")

    def test_context_is_optional_and_never_paired(self):
        with tempfile.TemporaryDirectory() as d:
            args = fixture(Path(d))
            args.baselines = Path(d) / "missing.csv"
            report = build_report(args)
            self.assertEqual(report.context, {4: {}, 8: {}})
            checks = {c["check"]: c for c in report.checks}
            self.assertEqual(checks["context_available"]["status"], "missing")
            self.assertEqual(context_rows(report), [])
            self.assertTrue(all(r["reference"] in REFERENCES for r in report.paired_requests))
            self.assertEqual(len(request_rows(report)), 240)
            self.assertEqual(len(summary_rows(report)), 8)
            self.assertEqual(len(paired_rows(report)), 6)


    def test_duplicate_request_total_fails(self):
        with tempfile.TemporaryDirectory() as d:
            report = build_report(fixture(Path(d)))
            report.arms[4]["ranked"].requests[0]["index"] = \
                report.arms[4]["ranked"].requests[1]["index"]
            with self.assertRaisesRegex(ValueError, "Duplicated request total"):
                validate_totals(report)

    def test_outputs_are_deterministic_and_clean(self):
        with tempfile.TemporaryDirectory() as d:
            args = fixture(Path(d))
            outputs = []
            for name in ("out1", "out2"):
                out = Path(d) / name
                main(["--measured", str(args.measured), "--ranked", str(args.ranked),
                      "--out", str(out), "--model", MODEL,
                      "--manifest", str(args.manifest),
                      "--baselines", str(args.baselines)])
                outputs.append(out)
            files = sorted(p.name for p in outputs[0].iterdir() if p.is_file())
            self.assertEqual(files, sorted(p.name for p in outputs[1].iterdir() if p.is_file()))
            for name in files:
                first = (outputs[0] / name).read_bytes()
                second = (outputs[1] / name).read_bytes()
                self.assertEqual(hashlib.sha256(first).hexdigest(),
                                 hashlib.sha256(second).hexdigest(), name)
            for name in files:
                if name.endswith((".csv", ".md", ".json")):
                    text = (outputs[0] / name).read_text()
                    self.assertNotIn("/data1", text, name)
                    self.assertNotIn("/home", text, name)
                    self.assertNotIn("n164", text, name)
                    self.assertNotIn(str(args.out), text, name)
            subprocess.run(["git", "init", "-q", str(outputs[0])], check=True)
            subprocess.run(["git", "-C", str(outputs[0]), "add", "-A"], check=True)
            result = subprocess.run(
                ["git", "-C", str(outputs[0]), "diff", "--cached", "--check"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
