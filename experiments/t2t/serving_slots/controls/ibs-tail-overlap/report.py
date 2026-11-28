"""Rescore the final A1 fixed-Q8 six-pair GPU tape on the frozen90 scoring host."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from bdlm_serving.evaluation import scoring
from bdlm_serving.evaluation.frozen90 import load_tasks

INDICES = (7, 13, 25, 33, 55, 71)


def require(ok, why):
    if not ok:
        raise ValueError(why)


def check(row):
    counts = Counter()
    for i, (e, c, inputs) in enumerate(zip(row["events"], row["calls"],
                                          row["forward_inputs"], strict=True)):
        pos = e["absolute_positions"]
        require(len(pos) == len(set(pos)) == c["rows"] == c["physical_rows"] ==
                len(inputs) == e["q_tokens"] and e["physical_forwards"] == c["batch"] == 1,
                "projection counts disagree")
        if e["phase"] == "prefill":
            continue
        require(0 < len(pos) <= row["q"] == 8 and row["verified_layers"] == 20, "Q or layer count")
        counts.update(F=1, Q=len(pos))
        s = e.get("successor")
        if s is None:
            continue
        head, tail = s["head_rows"], s["tail_rows"]
        require(pos == head+tail and bool(tail) and
                tail == list(range((e["block"]+1)*32, (e["block"]+1)*32+len(tail)))
                and s["head_key_end"] == max(head)+1 and s["tail_key_end"] == max(tail)+1
                and s["layer_widths"] == [len(pos)]*20, "joint projection")
        writes, discard = s["tail_writes"], s["discarded_writes"]
        require(set(writes+discard) <= set(tail) and not set(writes) & set(discard),
                "successor write classification")
        if s["promoted"]:
            require(not s["head_changed"] and not discard, "invalid successor promotion")
            if writes:
                following = row["events"][i+1]
                consumed = dict(zip(following["absolute_positions"], row["forward_inputs"][i+1],
                                    strict=True))
                expected = dict(zip(tail, s["tail_proposed"], strict=True))
                require(set(writes) <= set(following["dirty_before"]) and
                        all(consumed[p] == expected[p] for p in writes), "unpaid successor")
        else:
            require(not writes and not set(discard) & set(e["written_absolute"]), "invalid rollback")
        counts.update(joint=1, tail_Q=len(tail), promoted=int(s["promoted"]),
                      retained_writes=len(writes), discarded_writes=len(discard))
    require((counts["F"], counts["Q"]) ==
            (row["generation_forwards"], row["generation_query_rows"]), "per-request bill")
    return counts


def run(root, samples):
    raw = (root/"overlap.ndjson").read_bytes()
    rows = [json.loads(x) for x in raw.splitlines()]
    end = rows.pop()
    identity = json.loads((root/"launch.meta.json").read_text())["registration_sha256"]
    require((root/"runner.exit").read_text().strip() == "0" and
            end["complete"] is True and end["rows"] == len(rows) == 18, "incomplete run")
    require(sha256(Path(scoring.__file__).read_bytes()).hexdigest() ==
            "a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692", "scorer drift")
    tasks = load_tasks(samples)
    cases = []
    for index in INDICES:
        arm = {r["arm"]: r for r in rows if r["index"] == index}
        require(set(arm) == {"stock", "noop", "overlap"}, "missing arm")
        for k in ("events", "calls", "forward_inputs", "output_ids", "output_text", "status"):
            require(arm["stock"][k] == arm["noop"][k], "observer changed stock")
        case = {"index": index}
        for key, r in arm.items():
            require(all(r[k] == tasks[index][k] for k in ("index", "task_id", "dataset", "seed"))
                    and r["registration_sha256"] == identity, "registration or task mismatch")
            counts = check(r)
            correct = r["status"] == "success" and scoring.score(r["output_text"], tasks[index])["correct"]
            case[key] = dict(N=r["output_tokens"], correct=bool(correct), **counts)
        cases.append(case)
    summary = {}
    for k in ("stock", "overlap"):
        vals = [c[k] for c in cases]
        sums = Counter()
        for v in vals:
            sums.update(v)
        summary[k] = dict(**sums, macro_tpf=mean(v["N"]/v["F"] for v in vals),
                          pooled_te=sums["N"]/sums["Q"])
    return dict(cases=cases, summary=summary, registration=identity,
                raw_sha256=sha256(raw).hexdigest(), same_device_stock_noop=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("samples", type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.root, a.samples), sort_keys=True, indent=2))
