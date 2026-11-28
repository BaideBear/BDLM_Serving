"""Recheck sealed A1 pairs, emitting only small aggregates and fingerprints."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from .gemma_first_tile_kv import B, Q, audit
from .gemma_first_tile_kv_eval import HIST, ROOT, RT, matched
from .gemma_cold_frontier_last_eval import DEADLINE, GPU, SAMPLES, SCORER, read, sha
from .gemma_joint_verify_eval import ORDER, check, digest
from .gemma_rejected_noise_report import interval

ARMS = ("off", "on")
BASE = Path("/gs/runtime/verified-eos-1128/repo")


def bill(x, tape, enabled):
    ph = audit(x, tape, enabled)
    f, q = Counter(), Counter()
    d = {k: x[k] for k in ("N", "F", "encoder_F", "encoder_Q", "decoder_F", "decoder_Q")}
    d.update(ph, correct=int(x["correct"]), ids_sha256=digest(x["output_ids"]),
             TPF=x["N"]/x["F"], TE=x["N"]/x["decoder_Q"])
    first_debt = 0
    for head in range(ph["heads"]):
        block = [e for e in tape if e["head"] == head]
        check(block[0]["seen"] == block[0]["sc"] == block[0]["debt"] == 0 and
              block[0]["writes"] == block[1]["debt"] and
              (block[0]["writes"] == 0 if enabled else block[0]["writes"] > 0),
              "No first-tile exposure")
        first_debt += block[1]["debt"]
        debt = 0
        for e in block:
            check(0 < e["q"] <= Q and e["debt"] == debt, "Unpaid debt or overspend")
            debt = e["writes"]  # Every incoming dirty row fits and is projected in this Q32 policy.
            f[e["route"]] += 1
            q[e["route"]] += e["q"]
        check(not debt, "Outstanding at commit")
    check(f == Counter(x["route_F"]) and q == Counter(x["route_Q"]) and
          x["decoder_F"] == len(x["tape"]) and ph["writes"] == ph["dirty_Q"]
          and type(x["correct"]) is bool, "Bad measured routes or writes")
    d["next_debt_Q"] = first_debt
    for phase in ("cold", "warm", "close", "verify"):
        ticks = [e for e in tape if e["phase"] == phase]
        d[phase+"_dirty_Q"] = sum(e["debt"] for e in ticks)
        d[phase+"_sampled_Q"] = sum(e["sampled"] for e in ticks)
        d["first_head_"+phase+"_F"] = sum(e["head"] == 0 for e in ticks)
    return d, ph


def group(name, rows):
    s = dict(cohort=name, pairs=len(rows), replayed=len(rows), exposed=len(rows),
             same_ids=sum(r["same_ids"] for r in rows),
             rescues=sum(r["on_correct"] > r["off_correct"] for r in rows),
             losses=sum(r["on_correct"] < r["off_correct"] for r in rows),
             TPF_wins=sum(r["on_TPF"] > r["off_TPF"] for r in rows),
             TPF_ties=sum(r["on_TPF"] == r["off_TPF"] for r in rows),
             # Counterfactual on fixed off trajectories only; not an observed arm.
             off_one_free_F_per_head_macro_TPF=mean(r["off_N"]/(r["off_F"]-r["off_heads"]) for r in rows))
    pairs = [tuple(dict(dataset=r["dataset"], correct=r[a+"_correct"], TPF=r[a+"_TPF"]) for a in ARMS)
             for r in rows]
    for k in ("correct", "TPF"):
        s["delta_"+k] = mean(b[k]-a[k] for a,b in pairs)
        s["delta_"+k+"_lo95"], s["delta_"+k+"_hi95"] = interval(pairs, lambda x:x[k]) or (None,None)
    for a in ("off_", "on_"):
        for k in rows[0]:
            if k.startswith(a) and not k.endswith(("_sha256", "_TPF", "_TE")):
                s[k] = (max if k==a+"peak" else sum)(r[k] for r in rows)
        s[a+"macro_TPF"] = mean(r[a+"TPF"] for r in rows)
        s[a+"TE"] = s[a+"N"]/s[a+"decoder_Q"]
        for phase in ("cold", "warm", "close", "verify"):
            s[a+phase+"_F_per_head"] = s[a+phase+"_F"]/s[a+"heads"]
        s[a+"front_admit_rate"] = s[a+"first_tile_firstvisit_admits"]/(Q*s[a+"heads"])
        s[a+"deferred_mean_sampler_age"] = (s[a+"deferred_age_sum"]/s[a+"deferred_samples"]
                                             if s[a+"deferred_samples"] else None)
    return s


def report(n):
    reg = read(ROOT/"registration.json")
    gate = digest(reg)
    mark = read(ROOT/f"checkpoint-{n}.json")
    mpath = BASE/"benchmarks/frozen90/manifest.json"
    manifest = read(mpath)
    tasks = {x["index"]: x for x in manifest["samples"]}
    source = Path(__file__).parent
    refs = [sha(HIST/f"C32-pair-{i:02}.json") for i in range(90)]
    check(reg["gpu"] == GPU and reg["deadline"] == DEADLINE and reg["scorer"] == SCORER and
          reg["inputs"] == manifest["samples_sha256"] == sha(SAMPLES) and reg["manifest"] == sha(mpath) and
          reg["refs"] == digest(refs) and reg["policy"] == sha(RT/"registration.md") and
          reg["launcher"] == sha(RT/"launch.sh") and all(sha(source/(k+".py")) == v for k,v in reg["sources"].items()) and
          all(sha(BASE/"experiments/t2t/serving_slots"/(k+".py")) == v for k,v in reg["base"].items()) and
          mark == dict(n=n, pid=reg["pid"], reg=gate), "Bad registration or checkpoint")
    if n == 90:
        check(read(ROOT/"complete.json") == mark and len(list(ROOT.glob("pair-*.json"))) == n, "Not sealed")
    rows, hashes = [], []
    for i in sorted(ORDER[:n]):
        path = ROOT/f"pair-{i:02}.json"
        r = read(path)
        h = read(HIST/f"C32-pair-{i:02}.json")
        check(r["reg"] == gate and r["pid"] == reg["pid"] and r["ref"] == refs[i] and
              r["off"] == h["on"] and h["q"] == Q and h["source"] == reg["base"] and
              h["scorer_sha256"] == reg["scorer"] and h["samples_sha256"] == reg["inputs"] and
              r["off"]["input_sha256"] == r["on"]["input_sha256"] and
              all(r[k] == h[k] == tasks[i][k] for k in ("index","task_id","seed","dataset")), "Replay mismatch")
        matched(r["ticks"]["off"], r["ticks"]["on"])
        row = {k:r[k] for k in ("index", "task_id", "seed", "dataset")}
        row.update(q=Q, same_ids=int(r["off"]["output_ids"] == r["on"]["output_ids"]),
                   pair_sha256=sha(path), ref_sha256=refs[i], first_logits_sha256=r["ticks"]["off"][0]["first_logits"])
        for a in ARMS:
            d, ph = bill(r[a], r["ticks"][a], a == "on")
            check(ph == r["audit"][a], "Audit mismatch")
            row.update({a+"_"+k:v for k,v in d.items()})
        rows.append(row)
        hashes.append((i, row["pair_sha256"]))
    groups = [group("anchors3", [r for r in rows if r["index"] in (0,10,20)])]
    if n >= 30:
        groups.append(group("original30", rows[:30]))
    if n == 90:
        groups.extend((group("new60", rows[30:]), group("frozen90", rows)))
    provenance = dict(registration=reg, checkpoint=mark, raw_root=str(ROOT), historical_control=str(HIST),
                      pairs_digest=digest(hashes), report_sha256=sha(Path(__file__)),
                      macro_TPF="mean_i(N_i / (decoder_F_i + postprefill_encoder_F_i))",
                      TE="sum(N) / sum(selected history-reading decoder Q)")
    return rows, groups, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=int, choices=(3,30,90), default=90)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows, groups, provenance = report(args.checkpoint)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("pairs", rows), ("cohorts", groups)):
        with (args.out/(name+".csv")).open("w", newline="") as fp:
            writer = csv.DictWriter(fp, data[0], lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    (args.out/"provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
