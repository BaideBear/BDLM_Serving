"""CPU-only tests; all run output and report files are synthetic."""

from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import kv_factorial_report as report


def sha(label):
    return hashlib.sha256(label.encode()).hexdigest()


def event(stage, clock, width, state):
    return dict(block=0, stage=stage, clock=clock, hard_clock=6 - clock, q=width,
                positions=list(range(width)), trajectory_sha256=sha("canvas " + state),
                accepted_sha256=sha("accepted " + state))


def request(task, arm):
    index = task["index"]
    policy = report.POLICIES[arm]
    trace = [event("bootstrap", 5, 256, "shared pre-anchor")]
    endpoint = 8
    if index == 0:
        kv, score = arm[0] == "k", arm[1] == "k"
        anchor = event("selected", 4, 32, f"score={score}")
        picked = [*range(31), 33]
        anchor["positions"] = picked
        anchor["intervention"] = dict(
            base=list(range(32)), targeted=picked, random=[*range(31), 34],
            victim=31, candidate=33, random_candidate=34, executed=picked,
            fresh=[0], masks=[1, 2], stratum="slack", clock=2,
            persist_kv=kv, use_score=score, action_rows=picked if score else picked[:-1],
            pre_sha256=sha("shared pre-state"), pre_row_sha256=sha("stale row"),
            forward_row_sha256=sha("candidate row"),
            after_row_sha256=sha("candidate row" if kv else "stale row"),
            post_other_kv_sha256=sha("other rows"), post_rng_sha256=sha("rng"),
            post_nonkv_sha256=sha(f"nonkv score={score}"),
            after_sha256=anchor["trajectory_sha256"], after_admissions=1,
            strict_rows=[], revoked_rows=[], resident_l15=[],
        )
        trace.append(anchor)
        if not kv:
            selected = event("selected", 3, 32, f"later {arm}")
            if arm == "dk":
                selected["positions"] = picked
            trace.append(selected)
        endpoint = {"kk": 18, "kd": 12, "dk": 15, "dd": 10}[arm]
    trace.append(event("verify", 1, 256, arm if index == 0 else "ineligible"))
    physical = [["encoder", 17], *[["decoder", e["q"]] for e in trace], ["encoder", 256]]
    work = [(e["stage"], e["q"]) for e in trace] + [("reencode", 256)]
    f = Counter(p for p, _ in work)
    q = {p: sum(n for phase, n in work if phase == p) for p in f}
    total_q = sum(q.values())
    return dict(
        **{k: task[k] for k in report.IDENTITY}, model="gemma", policy=policy, status="success",
        output_ids=list(range(endpoint)), output_text="private synthetic output", output_tokens=endpoint,
        eos=True, truncated=False, prefill_rows=17, events=trace, physical_events=physical,
        generation_forwards=len(work), generation_query_rows=total_q,
        generation_physical_padded_rows=total_q, max_q=256, forwards_by_phase=dict(f),
        q_by_phase=q, q_histogram=dict(Counter(str(n) for _, n in work)), t2t_edits=0,
        tpf=endpoint / len(work), token_efficiency=endpoint / total_q,
    )


def save(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def setup_run(folder, policies):
    folder.mkdir()
    arms = [dict(budget=32, fresh=0, mask=32, selection=p[len("t2t_q32_f0_m32_"):]) for p in policies]
    save(folder / "run.json", dict(model="gemma", block=256, batch_size=1, cap=2048,
                                   policies=policies, arms=arms))
    save(folder / "qualified.json", dict(model="gemma", full_width=256, matched=30))
    for p in policies:
        save(folder / f"complete-{p}.json", dict(policy=p, requests=30))


@pytest.fixture
def inputs(tmp_path):
    first, second, history = (tmp_path / name for name in ("first", "second", "one-seat-v2-gemma"))
    dirs = dict(kk=first, dd=first, kd=second, dk=second)
    setup_run(first, [report.POLICIES[a] for a in ("kk", "dd")])
    setup_run(second, [report.POLICIES[a] for a in ("kd", "dk")])
    setup_run(history, [report.RANK])
    tasks = json.loads(report.MANIFEST.read_text())["samples"]
    scores = []
    for task in tasks:
        i = task["index"]
        for arm, folder in dirs.items():
            row = request(task, arm)
            policy = row["policy"]
            save(folder / f"{i:02d}-{policy}.json", row)
            scores.append(dict(model="gemma", policy=policy, nominal_q=32, fresh=0, mask=32, edit=0,
                               index=i, dataset=task["dataset"], task_id=task["task_id"],
                               max_q=256, correct=int(i > 0 or arm in ("kk", "dd")), failed=0,
                               truncated=0, **{k: row[v] for k, v in report.ENDPOINTS.items()}))
            if arm == "kk":
                old = deepcopy(row)
                old["policy"] = report.RANK
                if i == 0:
                    p = old["events"][1]["intervention"]
                    for field in ("persist_kv", "use_score", "action_rows", "pre_row_sha256",
                                  "forward_row_sha256", "after_row_sha256", "post_other_kv_sha256",
                                  "post_nonkv_sha256"):
                        del p[field]
                save(history / f"{i:02d}-{report.RANK}.json", old)
    score_path = tmp_path / "scores.csv"
    # The aggregate can contain the native baseline and arrive in any order.
    baseline = dict(scores[0], policy="native", nominal_q=256, fresh="", mask="", edit="")
    with score_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(scores[0]))
        writer.writeheader()
        writer.writerows([baseline, *reversed(scores)])
    return [first, second], score_path, history, tmp_path / "report"


def table(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_report_cli(inputs):
    runs, scores, history, out = inputs
    result = subprocess.run(
        [sys.executable, "-B", str(Path(report.__file__)), "--run", str(runs[1]),
         "--run", str(runs[0]), "--scores", str(scores), "--history", str(history), "--out", str(out)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Historical rank trajectory reproduced by kk on all 30 requests" in result.stdout
    paths = {kind: out / f"gemma-kv-factorial-{kind}.csv" for kind in ("requests", "phases", "summary")}
    rows, phases, summary = (table(paths[k]) for k in ("requests", "phases", "summary"))
    assert len(rows) == 120
    assert len(summary) == 4
    assert all(r["requests"] == "30" and r["anchors"] == "1" for r in summary)
    assert [(r["wins"], r["losses"]) for r in summary] == [("1", "0"), ("0", "1"), ("1", "0"), ("0", "1")]
    assert [(r["keep_policy"], r["drop_policy"]) for r in summary] == [
        (report.POLICIES[a], report.POLICIES[b]) for a, b in (("kk", "dk"), ("kd", "dd"), ("kk", "kd"), ("dk", "dd"))
    ]
    assert [(r["first_canvas_diff"], r["first_accepted_diff"]) for r in rows[:4]] == [
        ("1", "1"), ("1", "1"), ("0", "0"), ("0", "0"),
    ]
    assert rows[4]["anchor"] == "-1" and rows[4]["output_changed"] == "0"
    assert rows[4]["first_canvas_diff"] == ""
    for i in (0, 1):
        assert (rows[i]["keep_next_query_steps"], rows[i]["keep_next_query_phase"],
                rows[i]["keep_next_query_status"]) == ("1", "verify", "immediate")
    assert (rows[0]["drop_next_query_steps"], rows[0]["drop_next_query_phase"],
            rows[0]["drop_next_query_status"]) == ("1", "selected", "immediate")
    assert (rows[1]["drop_next_query_steps"], rows[1]["drop_next_query_phase"],
            rows[1]["drop_next_query_status"]) == ("2", "verify", "later")
    assert all(r["drop_next_query_status"] == "" for r in rows[2:])
    assert float(summary[0]["keep_macro_tpf"]) == pytest.approx((18 / 4 + 29 * (8 / 3)) / 30)
    assert float(summary[0]["delta_macro_tpf"]) == pytest.approx((18 / 4 - 15 / 5) / 30)
    assert float(summary[0]["delta_pooled_efficiency"]) == pytest.approx(
        (18 + 29 * 8) / (800 + 29 * 768) - (15 + 29 * 8) / (832 + 29 * 768)
    )
    assert (summary[0]["delta_F"], summary[0]["delta_Q"]) == ("-1", "-32")
    late = next(r for r in phases if r["index"] == "0" and r["contrast"] == "kv_at_score_keep"
                and r["phase"] == "selected")
    assert (late["keep_Q"], late["drop_Q"], late["delta_Q"]) == ("32", "64", "-32")
    assert any(r["phase"] == "reencode" for r in phases)
    for path in paths.values():
        contents = path.read_text()
        assert all(s not in contents for s in ("private synthetic output", "output_ids", "sha256", sha("rng")))


def test_candidate_query_does_not_cross_blocks():
    anchor = dict(event("selected", 4, 32, "anchor"), intervention=dict(candidate=3))
    later = dict(event("bootstrap", 5, 256, "another block"), block=1)
    assert report.next_candidate_query([anchor, later], 0) == ("", "", "never")


@pytest.mark.parametrize(("field", "value", "message"), [
    ("after_row_sha256", sha("candidate row"), "KV persistence"),
    ("post_nonkv_sha256", sha("wrong score state"), "same-score anchor: post_nonkv_sha256"),
    ("post_other_kv_sha256", sha("wrong other rows"), "unmatched anchor plan/state: post_other_kv_sha256"),
    ("post_rng_sha256", sha("wrong rng"), "unmatched anchor plan/state: post_rng_sha256"),
    ("action_rows", list(range(31)), "sampler action rows"),
])
def test_reject_bad_anchor(inputs, field, value, message):
    runs, _, _, out = inputs
    path = runs[1] / f"00-{report.POLICIES['dk']}.json"
    row = json.loads(path.read_text())
    row["events"][1]["intervention"][field] = value
    save(path, row)
    with pytest.raises(ValueError, match=message):
        report.main(*inputs)
    assert not out.exists()


@pytest.mark.parametrize(("arm", "index", "section", "field", "value", "message"), [
    ("kk", 0, 2, "clock", 2, "historical rank event core: clock"),
    ("kd", 0, 0, "trajectory_sha256", sha("changed prefix"), "pre-anchor trajectory"),
    ("kd", 1, None, "output_ids", [10] * 8, "ineligible.*historical rank replay: output_ids"),
    ("kk", 3, None, "seed", 42, "wrong frozen request or seed"),
])
def test_reject_bad_request(inputs, arm, index, section, field, value, message):
    runs, _, _, out = inputs
    path = runs[int(arm in ("kd", "dk"))] / f"{index:02d}-{report.POLICIES[arm]}.json"
    row = json.loads(path.read_text())
    target = row if section is None else row["events"][section]
    target[field] = value
    save(path, row)
    with pytest.raises(ValueError, match=message):
        report.main(*inputs)
    assert not out.exists()


def test_reject_stale_score_csv(inputs):
    _, scores, _, out = inputs
    rows = table(scores)
    scored = next(r for r in rows if r["policy"] == report.POLICIES["kk"] and r["index"] == "0")
    scored["N"] = "3"
    with scores.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="score CSV does not match"):
        report.main(*inputs)
    assert not out.exists()
