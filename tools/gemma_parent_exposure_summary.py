"""Merge disjoint first-successor observations; never treat these probes as quality results."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def summarize(paths, manifest):
    raw_manifest = manifest.read_bytes()
    require(sha256(raw_manifest).hexdigest() == "2e67d12f5045d486d1676a4b9cd35718b0791ad1b2a3c81fdec86a27bd773e62"
            and json.loads(raw_manifest)["samples_sha256"] == "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed",
            "wrong manifest")
    tasks = json.loads(raw_manifest)["samples"]
    cases, sources, environment = {}, [], set()
    for path in paths:
        blob = (path/"events.ndjson").read_bytes()
        events = [json.loads(line) for line in blob.splitlines()]
        require((path/"pipeline.exit").read_text().split() == ["0","0","0"], "unclean pipeline")
        w, last = events[0], events[-1]
        indices, q = w["indices"], w["q"]
        require(q in (32,512) and w["source"] == "7f160c54fa1fcd005aa92f9420c700f5df4cb47f"
                and w["profile"] == "dense_stop_joint_verified_eos" and w["transformers"] == "5.12.0",
                "wrong parent/source")
        environment.add((w["q"], w["source"], w["uuid"], w["torch"], w["transformers"], w["profile"]))
        require(w["kind"] == "worker" and last == dict(kind="finished", indices=indices, q=q)
                and [e["index"] for e in events[1:-1]] == indices, "incomplete registered coverage")
        sources.append(dict(attempt=path.name, indices=indices, sha256=sha256(blob).hexdigest(), runner=w["runner"]))
        for e in events[1:-1]:
            i = e["index"]
            require(i not in cases and 0 <= i < 90 and e["q"] == q, "overlap or wrong offer")
            if e["kind"] == "no_successor":
                require(e["decoder_F"] > 0, "no physical forward")
                cases[i] = {"width":0}
                continue
            pos = e["physical_rows"]
            start = e["origin"] + 256
            take = [p for p in pos if p >= start]
            require(e["kind"] == "first_projection" and e["task_id"] == tasks[i]["task_id"]
                    and e["seed"] == tasks[i]["seed"] and pos == sorted(set(pos))
                    and 0 < len(pos) == e["width"] <= q and e["layer_hooks"] > 0
                    and 0 < len(take) == e["successor_rows"] <= 256
                    and max(take) < start+256 and 256-len(take) == e["missing_successor_rows"], "bad row witness")
            cases[i] = dict(width=len(take), tick=e["tick"], total_width=len(pos))
    require(set(cases) == set(range(90)) and len(environment)==1, "not one complete fixed-offer panel")
    q, src, gpu, torch_version, transformers, profile = environment.pop()
    hist = Counter(r["width"] for r in cases.values())
    by_cohort = {}
    for name, ids in (("frozen90", range(90)), ("original30", range(30)), ("new60", range(30,90))):
        values = [cases[i]["width"] for i in ids]
        by_cohort[name] = dict(n=len(values), no_successor=values.count(0), first_already_full=values.count(256),
                               partial_first=sum(0 < v < 256 for v in values))
    return dict(scope="physical strict-parent first successor only; no candidate performance",
                q=q, source=src, uuid=gpu, torch=torch_version, transformers=transformers, profile=profile,
                histogram=dict(sorted(hist.items())), cohorts=by_cohort, shards=sources, cases=cases)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("paths", type=Path, nargs="+")
    args=ap.parse_args()
    print(json.dumps(summarize(args.paths,args.manifest),indent=2,allow_nan=False))
