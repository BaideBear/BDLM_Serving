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
    ARM_ORDER, REFERENCES, age_tables, build_report, context_rows, load_manifest, main,
    paired_rows, read_csv, request_rows, summary_rows, validate_totals,
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


def check_generated_diff(out):
    # Use only the generated fixture directory's disposable index.
    subprocess.run(["git", "init", "-q", str(out)], check=True)
    subprocess.run(["git", "-C", str(out), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(out), "diff", "--cached", "--check"], check=True)


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
            check_generated_diff(outputs[0])


def age_fixture(root):
    args = fixture(root)
    args.age_measured = root / "n164-private-age-input"
    arms = {}
    for q, salt, correct, n, f in ((4, "a", 26, 100, 10), (4, "b", 27, 105, 10),
                                   (8, "a", 27, 130, 12), (8, "b", 26, 135, 12)):
        policy = f"t2t_q{q}_f0_m{q-1}_hard_exchange_oldhalf_{salt}"
        rows = make_rows(q, policy, q - 1, correct_upto(correct), n, f)
        if (q, salt) == (4, "a"):
            rows[0]["correct"], rows[26]["correct"] = 0, 1  # one loss, two wins vs zero
        for row in rows:
            row["physical_Q"] += row["F"]
        arms[policy] = rows[::-1]  # pairing is by manifest identity, not CSV order
    write_arm_dir(args.age_measured, arms)
    return args, arms


def age_cli(args, out):
    return ["--age-measured", str(args.age_measured), "--measured", str(args.measured),
            "--ranked", str(args.ranked), "--manifest", str(args.manifest),
            "--model", MODEL, "--out", str(out)]


class ExchangeAgeReportTest(unittest.TestCase):
    def test_age_cli_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            args, arms = age_fixture(Path(d))
            main(age_cli(args, args.out))
            points = read_csv(args.out / "exchange-age-summary.csv")
            results = read_csv(args.out / "exchange-age-paired.csv")
            svg = (args.out / "exchange-age-01.svg").read_text()
            md = (args.out / "exchange-age-report.md").read_text()
            self.assertEqual({p.name for p in args.out.iterdir()}, {
                "exchange-age-summary.csv", "exchange-age-paired.csv",
                "exchange-age-01.svg", "exchange-age-report.md"})
            self.assertEqual(len(points), 12)
            self.assertEqual({(p["nominal_q"], p["arm"]) for p in points}, {
                (q, arm) for q in ("4", "8") for arm in (*ARM_ORDER, "oldhalf_a", "oldhalf_b")})
            self.assertTrue(all(p["requests"] == "30" and p["dataset"] == "all" for p in points + results))
            for p in points:
                if p["policy"] in arms:
                    rows = arms[p["policy"]]
                    for key in ("correct", "N", "F", "Q", "physical_Q"):
                        self.assertEqual(int(p[key]), sum(r[key] for r in rows))
                    self.assertAlmostEqual(float(p["macro_tpf"]), mean(r["N"] / r["F"] for r in rows))
                    self.assertAlmostEqual(float(p["physical_efficiency"]), int(p["N"]) / int(p["physical_Q"]))
            self.assertEqual(len(results), 8)
            by_pair = {(r["nominal_q"], r["salt"], r["reference"]): r for r in results}
            self.assertEqual(set(by_pair), {(q, s, ref) for q in ("4", "8") for s in "ab"
                                           for ref in (f"placebo_{s}", "zero")})
            by_arm = {(p["nominal_q"], p["arm"]): p for p in points}
            for (q, salt, ref), r in by_pair.items():
                old, other = by_arm[q, f"oldhalf_{salt}"], by_arm[q, ref]
                for key, delta in (("macro_tpf", "delta_tpf"), ("physical_efficiency", "delta_physical_efficiency"),
                                   ("correct", "delta_correct"), ("Q", "delta_Q"), ("physical_Q", "delta_physical_Q")):
                    self.assertAlmostEqual(float(r[delta]), float(old[key]) - float(other[key]))
                self.assertEqual(r["budget_pass"], "1" if q == "4" else "0")
                self.assertEqual(r["salt_pass"], "0" if (q, salt) == ("8", "b") else "1")
            r = by_pair["4", "a", "zero"]
            self.assertEqual((r["wins"], r["losses"], r["delta_correct"]), ("2", "1", "1"))
            self.assertIn("+1", svg)
            self.assertIn(f"{float(r['delta_tpf']):+.4f}", svg)
            self.assertIn("| Q4/1 | PASS | PASS | **PASS** |", md)
            self.assertIn("| Q8/1 | PASS | FAIL | **FAIL** |", md)
            # Absolute input paths and source basenames are not public identifiers.
            repeat = Path(d) / "repeat"
            main(age_cli(args, repeat))
            for p in args.out.iterdir():
                self.assertEqual(p.read_bytes(), (repeat / p.name).read_bytes())
                text = p.read_text()
                for private in (str(Path(d)), args.age_measured.name, "/data1/", "/home/"):
                    self.assertNotIn(private, text)
                self.assertEqual(text, "\n".join(line.rstrip() for line in text.rstrip().splitlines()) + "\n")
            check_generated_diff(args.out)

    def test_age_uses_historical_controls_and_separate_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args, _ = age_fixture(root)
            # Earlier measured tables may also contain a noedit arm; ignore it.
            for kind in ("requests", "summary"):
                name = f"{MODEL}-online-{kind}.csv"
                history = read_csv(args.ranked / name)
                zero = [r for r in history if r["policy"].endswith("noedit")]
                write_csv(root / "hard" / name, zero)
                write_csv(args.ranked / name, [r for r in history if not r["policy"].endswith("noedit")])
                write_csv(args.measured / name, read_csv(args.measured / name) + zero)
            points, _ = age_tables(args)
            self.assertEqual({(r["arm"], r["source"]) for r in points}, {
                ("zero", "hard"), ("ranked", "ranked"), ("placebo_a", "measured"),
                ("placebo_b", "measured"), ("oldhalf_a", "age_measured"), ("oldhalf_b", "age_measured")})
            args.out.mkdir()
            sentinel = args.out / "exchange-placebo-report.md"
            sentinel.write_text("existing report\n")
            with self.assertRaises(SystemExit) as error:
                main(age_cli(args, args.out))
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(list(args.out.iterdir()), [sentinel])
            self.assertEqual(sentinel.read_text(), "existing report\n")

    def test_each_comparison_constrains_budget(self):
        # Each case fails just one requirement that the other checks would miss.
        cases = [("b", 26, 105, "placebo_b"),  # correct below own uniform, above zero
                 ("b", 27, 104, "placebo_b"),  # TPF below own uniform, above zero
                 ("a", 26, 99, "zero"),       # TPF below zero, above own uniform
                 ("a", 25, 100, "zero")]      # correct tied with zero
        for salt, correct, n, ref in cases:
            with self.subTest(salt=salt, correct=correct, n=n), tempfile.TemporaryDirectory() as d:
                args, arms = age_fixture(Path(d))
                policy = f"t2t_q4_f0_m3_hard_exchange_oldhalf_{salt}"
                arms[policy] = make_rows(4, policy, 3, correct_upto(correct), n, 10)
                write_arm_dir(args.age_measured, arms)
                _, pairs = age_tables(args)
                q4 = [r for r in pairs if r["nominal_q"] == 4]
                self.assertEqual([(r["salt"], r["reference"]) for r in q4 if not r["comparison_pass"]],
                                 [(salt, ref)])
                self.assertTrue(all(not r["budget_pass"] for r in q4))

    def test_incomplete_age_tables_do_not_write(self):
        for failure, message in (("absent", "hard_exchange_oldhalf_b"), ("task", "frozen30"),
                                 ("summary", "disagrees"), ("uniform", "hard_exchange_random_a")):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as d:
                args, _ = age_fixture(Path(d))
                directory = args.measured if failure == "uniform" else args.age_measured
                filename = "summary" if failure == "summary" else "requests"
                path = directory / f"{MODEL}-online-{filename}.csv"
                rows = read_csv(path)
                if failure in ("absent", "uniform"):
                    suffix = "oldhalf_b" if failure == "absent" else "random_a"
                    rows = [r for r in rows if r["nominal_q"] != "8" or not r["policy"].endswith(suffix)]
                elif failure == "task":
                    rows[-1]["task_id"] = "wrong"
                else:
                    next(r for r in rows if r["dataset"] == "all")["physical_Q"] = "1"
                write_csv(path, rows)
                with self.assertRaisesRegex(ValueError, message):
                    age_tables(args)
                with self.assertRaises(SystemExit) as result:
                    main(age_cli(args, args.out))
                self.assertEqual(result.exception.code, 2)
                self.assertFalse(args.out.exists())


if __name__ == "__main__":
    unittest.main()
