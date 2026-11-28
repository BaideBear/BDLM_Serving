"""Verify a completed GPU replay against raw evidence; print only compact statistics."""
import argparse
from collections import defaultdict
from hashlib import sha1
import json
from pathlib import Path
from random import Random
from statistics import mean

from bdlm_serving.evaluation import frozen, frozen90
from experiments.t2t.serving_slots.gemma_ibs_prefix_bridge import audit, match, require
from experiments.t2t.serving_slots.gemma_ibs_prefix_bridge_eval import PINS, digest, read, sha


def aggregate(rows):
    result = dict(pairs=len(rows), exposed=sum(r["first"]["first"] is not None for r in rows),
                  rescues=sum(r["on"]["correct"] and not r["off"]["correct"] for r in rows),
                  losses=sum(r["off"]["correct"] and not r["on"]["correct"] for r in rows),
                  changed_ids=sum(r["off"]["ids_sha"] != r["on"]["ids_sha"] for r in rows))
    for arm in ("off","on"):
        group = [r[arm] for r in rows]
        total = {k:sum(x[k] for x in group) for k in ("correct","N","F","decoder_Q","encoder_F","extra_Q")}
        result[arm] = dict(**total, macro_TPF=mean(x["N"]/x["F"] for x in group),
                           pooled_TE=total["N"]/total["decoder_Q"], peak=max(x["peak"] for x in group))
    return result


def paired_uncertainty(rows, repeats=4000):
    """Paired bootstrap within frozen dataset and original/added strata; descriptive only."""
    strata = defaultdict(list)
    for r in rows:
        a,b = r["off"],r["on"]
        strata[r["dataset"],r["index"]<30].append((int(b["correct"])-int(a["correct"]),
                b["N"]/b["F"]-a["N"]/a["F"],a["N"],a["decoder_Q"],b["N"],b["decoder_Q"]))
    rng = Random(20261128)
    dist = [[],[],[]]
    for _ in range(repeats):
        sample = [x for group in strata.values() for x in rng.choices(group,k=len(group))]
        d_score,d_tpf,n0,q0,n1,q1 = map(sum,zip(*sample))
        for out, value in zip(dist,(100*d_score/len(sample),d_tpf,100*(n1/q1-n0/q0))):
            out.append(value)
    result = {}
    for key,v in zip(("correct_delta_pp","macro_TPF_delta","pooled_TE_delta_pp"),dist):
        v.sort()
        result[key]=[v[int(repeats*.025)],v[int(repeats*.975)-1]]
    return result


def verify(out, panel_path, samples, generation_config):
    panel = read(panel_path)
    reg = read(out/"pairs/registration.json")
    cfg = generation_config.read_bytes()
    spec, = [s for s in read(frozen.ROOT/"benchmarks/checkpoints/diffusiongemma.json")
             if s["name"] == "generation_config.json"]
    require(sha1(b"blob "+str(len(cfg)).encode()+b"\0"+cfg).hexdigest() == spec["git_blob_sha1"],
            "Generation config changed")
    eos = set(json.loads(cfg)["eos_token_id"])
    require(reg["panel_sha"] == sha(panel_path) and reg["panel"] == panel and reg["pins"] == PINS,
            "Panel or sources mismatch")
    tasks = frozen90.load_tasks(samples)
    keys = [(i,q) for i in panel["indices"] for q in panel["tiers"]]
    records = [json.loads(s) for s in (out/"receipts.ndjson").read_text().splitlines()]
    marker = read(out/"pairs/complete.json")
    require(marker == dict(reg=digest(reg), pairs=len(keys)) and
            [(r["index"],r["q"]) for r in records] == keys and
            {p.name for p in (out/"pairs").glob("*-q*.json")} == {f"{i:02}-q{q}.json" for i,q in keys},
            "Missing/duplicate/unregistered pair")
    inputs, hashes = {}, []
    for row in records:
        i, q = row["index"], row["q"]
        path = out/"pairs"/f"{i:02}-q{q}.json"
        full = read(path)
        hashes.append(sha(path))
        both = [full["off"], full["on"]]
        require(all(full[k] == row[k] for k in row if k not in ("off","on")) and
                all(row[k] == tasks[i][k] for k in ("index","dataset","seed","task_id")) and
                full["reg"] == marker["reg"] and full["first"] == match(*both) and
                full["plain_sha"] == digest({k:v for k,v in both[0].items() if k not in ("notes","bill")}) and
                inputs.setdefault(i,row["input_sha"]) == row["input_sha"] and both[0]["enc"][0] == both[1]["enc"][0],
                "Identity, zero-change replay or first fork mismatch")
        for arm, raw in zip(("off","on"),both):
            enc = list(map(tuple,raw["enc"]))
            bill = audit(raw["notes"],raw["trace"],list(map(tuple,raw["events"])),raw["output_ids"],
                         enc,enc[0][0],eos,q,arm=="on")
            require(bill == raw["bill"] and bill["admissions"] == bill["dirty_paid"] == 256*bill["heads"] and
                    all(bill[k] == raw[k] for k in ("decoder_F","F","encoder_F","encoder_Q","decoder_Q")) and
                    type(raw["correct"]) is bool and 0 <= raw["N"] <= len(raw["output_ids"]),
                    "Physical audit failed")
            compact = {k:raw[k] for k in ("N","F","decoder_Q","encoder_F","correct")}
            require(row[arm] == dict(**compact,extra_Q=bill["extra_Q"],peak=bill["peak"],
                                     ids_sha=digest(raw["output_ids"])), "Compact arm mismatch")
    groups = {}
    for q in panel["tiers"]:
        x = [r for r in records if r["q"] == q]
        cohorts = [("all",x),("original30_panel",[r for r in x if r["index"] < 30]),
                   ("added60_panel",[r for r in x if r["index"] >= 30])]
        if "prior_six" in panel:
            seen = set(panel["prior_six"] + panel["old_pilot"])
            cohorts += [("previously_screened15",[r for r in x if r["index"] in seen]),
                        ("not_on_these_two_panels75",[r for r in x if r["index"] not in seen]),
                        ("excluding_prior_long_index51",[r for r in x if r["index"] != 51])]
        groups[str(q)] = {tag:aggregate(group) for tag, group in cohorts if group}
        if "prior_six" in panel:
            for tag,group in cohorts:
                groups[str(q)][tag]["paired_stratified_ci95"] = paired_uncertainty(group)
    return dict(panel_sha=reg["panel_sha"], register=marker["reg"], pairs_digest=digest(hashes),
                gpu=reg["gpu"], torch=reg["torch"], indices=panel["indices"], verified=len(records),
                executions=3*len(records), cohorts=groups, raw_directory=str(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("out","panel","samples","generation-config"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.out,a.panel,a.samples,a.generation_config), indent=2, allow_nan=False))
