"""Reconcile the archived Live Q128 route/phase bills and optional A0 probe."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean


def check(ok, why):
    if not ok:
        raise RuntimeError(why)


def digest(x):
    return hashlib.sha256(json.dumps(x, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_results", type=Path)
    parser.add_argument("--probe-log", type=Path)
    args = parser.parse_args()
    manifests = [hashlib.sha256() for _ in range(3)]
    def read(slot, name):
        data = (args.snapshot_results / name).read_bytes()
        manifests[slot].update(f"{hashlib.sha256(data).hexdigest()}  {name}\n".encode())
        return json.loads(data)

    cases = {}
    for i in range(90):
        pair = read(0, f"verified-eos-1128/C128-pair-{i:02}.json")
        profile = read(1, f"profile-close-1128/C128-profile-fuse1-{i:02}.json")
        pacing = read(2, f"cold-pacing-1128/C128-pace0-{i:02}.json")
        check(pair["schema"] == "verified-eos-pair-v1" and pair["q"] == profile["budget"] == pacing["budget"] == 128
              and profile["fuse"] is True and pacing["pace"] is False
              and profile["prefix"] == pacing["prefix"] == [1, 4], (i, "source arm"))
        for k in ("index", "seed", "task_id", "dataset", "samples_sha256", "scorer_sha256"):
            check(pair[k] == profile[k] == pacing[k], (i, k))
        off, on = pair["off"], pair["on"]
        phase = pacing["phase"]
        for k in ("N", "correct", "F", "decoder_F", "decoder_Q", "encoder_F", "encoder_Q",
                  "route_F", "route_Q", "input_sha256"):
            check(off[k] == profile[k] == pacing[k], (i, k))
        check(off["output_ids"] == profile["output_ids"], (i, "history IDs"))
        check(digest(off["output_ids"]) == pacing["output_digest"], (i, "phase IDs"))
        for k in ("output_ids", "N", "correct", "encoder_F", "encoder_Q", "encoder_digest"):
            check(off[k] == on[k], (i, k))
        eos_saved = (256 - on["hit"]["chunk_end"]) // 128
        check(on["tape"] == off["tape"][:on["decoder_F"]] and
              off["F"] - on["F"] == eos_saved and off["decoder_Q"] - on["decoder_Q"] == eos_saved * 128,
              (i, "terminal EOS savings"))
        cold, warm, close = (phase[f"{k}_drain_F"] for k in ("cold", "warm", "closing"))
        r_f, r_q = (Counter(on[k]) for k in ("route_F", "route_Q"))
        check(cold + warm + close == r_f["drain"] and (cold + warm) * 128 + phase["closing_drain_Q"] ==
              r_q["drain"] and r_f["verify_paid"] == profile["fused_F"], (i, "phase split"))
        check(sum(profile["categories"].values()) == off["decoder_Q"] and
              sum(r_f.values()) == on["decoder_F"] and sum(r_q.values()) == on["decoder_Q"] and
              on["F"] == on["decoder_F"] + on["encoder_F"], (i, "reconciled bills"))
        cases[i] = (on, profile, phase, eos_saved)

    cohorts = {}
    for name, indices in (("original30", range(30)), ("new60", range(30, 90)), ("frozen90", range(90))):
        bills, rf, rq, split, extra = (Counter() for _ in range(5))
        for i in indices:
            on, profile, phase, eos_saved = cases[i]
            bills.update({k: int(on[k]) for k in ("correct", "N", "F", "decoder_F", "decoder_Q",
                                                "encoder_F", "encoder_Q")})
            rf.update(on["route_F"])
            rq.update(on["route_Q"])
            split.update({k: phase[k] for k in ("cold_drain_F", "warm_drain_F", "closing_drain_F",
                                              "closing_drain_Q")})
            extra.update(eos_saved_F=eos_saved, already_fused_F=profile["fused_F"],
                         already_fused_dirty_Q=profile["fused_dirty_Q"],
                         ordinary_dirty_Q=profile["categories"]["mandatory"],
                         prefix_edits=profile["prefix_exposure"]["prefix_edits"],
                         prefix_revoked=profile["prefix_exposure"]["prefix_revoked"])
        cohorts[name] = dict(requests=len(indices), bills=bills, route_F=rf, route_Q=rq, drain=split, extra=extra,
                             macro_tpf=mean(cases[i][0]["N"]/cases[i][0]["F"] for i in indices),
                             pooled_TE=bills["N"]/bills["decoder_Q"])
    out = dict(schema="live-native-cost-v1", source_manifest_sha256={
        name: h.hexdigest() for name, h in zip(("verified_eos", "profile_close", "cold_pacing"), manifests)
    }, cohorts=cohorts)
    if args.probe_log:
        raw = args.probe_log.read_bytes()
        records = [json.loads(line) for line in raw.splitlines() if line.startswith(b'{"kind":')]
        worker, *measured, done = records
        check(worker["kind"] == "start" and done["kind"] == "complete"
              and worker["q"] == 128 and len(measured) == 6, "Incomplete probe")
        check([r["off"]["index"] for r in measured] == worker["cases"] == [0, 10, 20, 30, 40, 50],
              "Wrong pilot cohort")
        aggregate = Counter()
        comparisons = []
        for r in measured:
            a = r["off"]
            h, _, p, _ = cases[a["index"]]
            check(r["kind"] == "case" and a["input_sha256"] == h["input_sha256"], (a["index"], "A0 input"))
            same = {k: a[k] == h[k] for k in ("F", "N", "correct", "encoder_digest", "tape")}
            same["ids"] = a["ids_sha256"] == digest(h["output_ids"])
            same.update({k: a[k] == h[hk] for k, hk in (("D", "decoder_F"), ("E", "encoder_F"), ("Q", "decoder_Q"))})
            same.update({k + "_drains": a[k + "_drains"] == p[k + "_drain_F"] for k in ("cold", "warm", "closing")})
            shared_steps = min(len(a["tape"]), len(h["tape"]))
            divergent = next((k for k in range(shared_steps) if a["tape"][k] != h["tape"][k]),
                             None if len(a["tape"]) == len(h["tape"]) else shared_steps)
            comparisons.append(dict(index=a["index"], same=same, first_divergent_tick=divergent))
            aggregate.update({k: a[k] for k in ("F", "D", "E", "Q", "N", "correct", "saved")})
            if r["on"] is not None:
                aggregate.update(treatment_paid_savings_F=r["on"]["saved"])
        for key, total in done["totals"].items():
            check(total == sum(r["off"][key] for r in measured), f"Wrong probe total: {key}")
        release, = [line.decode() for line in raw.splitlines() if line.startswith(b"RELEASE ")]
        check(all(k in release.split() for k in ("rc=0", "memory_mib=0", "apps=''", "locks=closed",
                                                 f"gpu={worker['gpu']}")), "Missing A0 release")
        out["a0_probe"] = dict(worker=worker, totals=done["totals"], stock_bills=aggregate, release=release,
                               raw_sha256=hashlib.sha256(raw).hexdigest(), comparisons=comparisons,
                               treatment_requests=sum(r["on"] is not None for r in measured))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
