import csv
import json
from copy import deepcopy
from hashlib import sha256

import pytest

from tools import beneficiary_rows as report


def fixture(model="llada21", stratum="mask"):
    q = {"llada21": 4, "gemma": 32}[model]
    victim = 1 if stratum == "mask" else 3
    base = list(range(q))
    swap = lambda row: sorted((set(base) - {victim}) | {row})
    paths = [([None, 30, 40], [20, 30, 40]),
             ([20, None, 40], [None, None, 40], [20, None, 40], [20, 30, 40], [None, 30, 40]),
             ([None, None, 40],)]
    outputs = [[9, 20, 30], [9, 99], [9, 20, 33, 40]]
    raw = []
    for mode, selected, path, output in zip(report.MODES, (base, swap(q), swap(q+1)), paths, outputs):
        events = [dict(block=3, stage="bootstrap")]
        for t, ids in enumerate(path):
            fingerprint = sha256(str(ids).encode()).hexdigest()
            positions = selected if t == 0 else base
            event = dict(block=3, stage="selected" if t == 0 else "verify", q=len(positions),
                         accepted_sha256=fingerprint, trajectory_sha256=fingerprint,
                         beneficiary=dict(rows=[0, 1, 2], ids=ids, output_offsets=[1, 2, 3]))
            if model == "llada21":
                event.update(absolute_positions=[96+i for i in positions], phase="denoise")
            else:
                event["positions"] = positions
            if t == 0:
                event["intervention"] = dict(base=base, targeted=swap(q), random=swap(q+1),
                    victim=victim, candidate=q, random_candidate=q+1, fresh=[], masks=[0, 1, 2],
                    stratum=stratum, clock=7, executed=positions, pre_sha256="1"*64,
                    post_rng_sha256="2"*64, after_sha256=fingerprint,
                    after_admissions=sum(x is not None for x in ids))
            events.append(event)
        events.extend([dict(block=3, stage="commit"), dict(block=4, stage="bootstrap")])
        if stratum == "none":
            events, output = [dict(block=3, stage="bootstrap")], [9]
        raw.append(dict(index=0, model=model, dataset="gsm8k", task_id="task", seed=1,
                        policy=report.policy_name(q, mode), status="success", events=events,
                        output_ids=output, output_tokens=len(output), generation_forwards=len(events),
                        generation_query_rows=50, generation_physical_padded_rows=50,
                        forwards_by_phase={"denoise": 4}, q_by_phase={"denoise": 50}))
    return raw


@pytest.mark.parametrize("model", ["llada21", "gemma"])
def test_fixed_offsets_and_horizon(model):
    raw = fixture(model)
    original = deepcopy(raw)
    points, pairs = report.measure(raw)
    assert raw == original
    by = {(r["arm"], r["row_kind"]): r for r in points}
    base, rank, rand = (by[m, "common_mask"] for m in report.MODES)
    # The other common MASK is beyond base EOS, but still counts for admissions.
    assert (base["watched"], base["eligible"], base["anchor_admitted"]) == (2, 1, 1)
    assert (base["first_wait_sum"], base["stable_wait_sum"], base["carried_row_obs"]) == (1, 1, 4)
    assert [rank[k] for k in report.AGREE] == [1, 0, 1, 1]
    assert (rank["first_wait_sum"], rank["stable_wait_sum"]) == (0, 2)  # revoked again at t=4
    assert (rank["anchor_admitted"], rank["endpoint_different"], rank["endpoint_missing"]) == (2, 1, 0)
    assert (rand["first_hit"], rand["first_wait_sum"], rand["stable_wait_sum"]) == (0, 4, 4)
    assert (rand["branch_missing"], rand["carried_row_obs"]) == (0, 6)
    assert (base["branch_missing"], base["endpoint_missing"]) == (1, 0)
    victim = by["rank", "mask_victim"]
    assert (victim["eligible"], victim["anchor_admitted"], victim["endpoint_missing"]) == (1, 0, 1)
    assert (victim["first_wait_sum"], victim["stable_wait_sum"]) == (3, 3)
    assert by["random", "mask_victim"]["endpoint_different"] == 1
    for r in pairs:
        a, b = (by[m, r["row_kind"]] for m in r["pair"].split("_vs_"))
        assert r["eligible"] == a["eligible"] == b["eligible"]
        for k in report.COUNTS:
            assert r["delta_"+k] == a[k] - b[k]


def test_anchor_with_no_watched_masks_and_collision_replay():
    raw = fixture(stratum="slack")
    for r in raw:
        for e in r["events"]:
            if "intervention" in e:
                e["intervention"].update(masks=[], after_admissions=0)
            if "beneficiary" in e:
                e["beneficiary"] = dict(rows=[], ids=[], output_offsets=[])
    points, pairs = report.measure(raw)
    assert all(r["stratum"] == "slack" and r["watched"] == r["eligible"] == 0 for r in points + pairs)
    raw = fixture()
    for r in raw:
        p = r["events"][1]["intervention"]
        p.update(random=p["targeted"], random_candidate=p["candidate"])
    raw[2] = deepcopy(raw[1])
    assert len(report.measure(raw)[1]) == 6
    raw[2]["events"][2]["beneficiary"]["ids"][0] = 4321
    with pytest.raises(ValueError, match="Same state"):
        report.measure(raw)


@pytest.mark.parametrize("error,match", [
    ("offset", "offsets"), ("rows", "rows/order"), ("late", "Missing or unexpected"),
    ("null", "IDs"), ("off_q", "Off-Q"), ("state", "Same state"),
    ("pre", "pre-state"), ("prefix", "Prefixes differ"), ("admit", "admissions"),
])
def test_reject_invalid_evidence(error, match):
    raw = fixture()
    ev = raw[1]["events"]
    watch = ev[1]["beneficiary"]
    if error == "offset":
        watch["output_offsets"] = [2, 3, 4]
    elif error == "rows":
        watch["rows"].reverse()
    elif error == "late":
        del ev[5]["beneficiary"]  # still required after the reporting horizon
    elif error == "null":
        watch["ids"][1] = False
    elif error == "off_q":
        watch["ids"][1] = 30
    elif error == "state":
        ev[2]["accepted_sha256"] = raw[0]["events"][1]["accepted_sha256"]
    elif error == "pre":
        ev[1]["intervention"]["pre_sha256"] = "e"*64
    elif error == "admit":
        ev[1]["intervention"]["after_admissions"] += 1
    else:
        ev[0]["stage"] = "dense"
    with pytest.raises(ValueError, match=match):
        report.measure(raw)


def emit(path, data):
    path.write_text(json.dumps(data))


@pytest.mark.parametrize("model,q", [("llada21", 4), ("gemma", 32)])
def test_report_and_historical_replay(tmp_path, monkeypatch, model, q):
    new, old, out = (tmp_path / d for d in ("private-new", "private-archive", "out"))
    new.mkdir()
    old.mkdir()
    monkeypatch.setitem(report.DEFAULT_RUNS, model, (old, None, q))
    policies = [report.policy_name(q, arm) for arm in report.MODES]
    emit(new / "run.json", dict(model=model, policies=policies))
    for p in policies:
        emit(new / f"complete-{p}.json", dict(policy=p, requests=30))
    for task in report.read_json(report.DEFAULT_MANIFEST)["samples"]:
        i = task["index"]
        for r in fixture(model, {0: "mask", 10: "slack"}.get(i, "none")):
            r.update({k: task[k] for k in ("index", "task_id", "dataset", "seed")})
            name = f"{i:02d}-{r['policy']}.json"
            archived = report.strip_watch(r)
            emit(old / name, archived)
            if i in (0, 10):
                r["events"][1]["intervention"]["victim_queried"] = r["policy"].endswith("base")
            emit(new / name, r)
    report.main(new, out)
    outputs = {f.name: f.read_bytes() for f in out.iterdir()}
    report.main(new, out)
    assert outputs == {f.name: f.read_bytes() for f in out.iterdir()}
    for name, count in (("requests", 180), ("paired", 180), ("summary", 192)):
        with (out / f"{model}-beneficiary-{name}.csv").open() as f:
            data = list(csv.DictReader(f))
        assert len(data) == count
        assert not ({"output_ids", "ids", "rows", "output_offsets", "prompt_ids", "task_id"} & data[0].keys())
        if name == "summary":
            rows = {(r["dataset"], r["stratum"]): r for r in data
                    if r["comparison"] == "rank_vs_base" and r["row_kind"] == "mask_victim"}
            assert tuple(rows["all", "all"][k] for k in ("requests", "anchors", "eligible_requests")) == ("30", "2", "1")
            assert (rows["all", "none"]["requests"], rows["all", "slack"]["watched"]) == ("28", "0")
            assert all(rows[d, "all"]["requests"] == "10" for d in report.DATASETS[1:])
            assert rows["gsm8k", "mask"]["anchor_admitted"] == "-1"
            assert float(rows["all", "mask"]["first_wait_mean"]) == 3
            assert rows["all", "slack"]["first_wait_mean"] == ""
    path = old / f"29-{policies[-1]}.json"
    changed = report.read_json(path)
    changed["output_ids"] = [31415926]
    emit(path, changed)
    with pytest.raises(ValueError, match="Historical replay mismatch"):
        report.main(new, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
