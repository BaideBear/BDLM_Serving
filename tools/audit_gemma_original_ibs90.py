"""Cluster-only audit of original IBS, with frozen scoring and physical Q accounting."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from bdlm_serving.evaluation.frozen import sha
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.evaluation.original30 import save_json
from bdlm_serving.evaluation.scoring import score


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def require(ok, where):
    if not ok:
        raise ValueError(where)


def audit(args):
    tasks = load_tasks(args.data / "samples.json")
    args.output.mkdir(exist_ok=False, parents=True)
    results, ledger, comparisons = [], [], []
    for c in args.chunks:
        policy = f"ibs_{c}"
        archived = c != 512
        run = args.new if not archived else args.data / (
            "gemma-optimus-reference90" if c in (32, 128) else "gemma-optimus-primary90")
        meta = read(run / "run.json")
        require(all(meta[k] == v for k, v in dict(
            model="gemma", block=256, cap=2048, seed_offset=0, suite="frozen90",
            samples_sha256=sha(args.data / "samples.json"), hard_q_cap=False, batch_size=1,
            revision="cd40195554d3cfcb56f34b9adfb66efc790ed5b6",
            upstream="7343a37b2a69337caa452f17b59727dc6665f039", attention="sdpa",
            dtype="bfloat16", experts="grouped_mm", transformers="5.12.0",
            acceptance="optimus_dynamic_sparse_argmax_0.9", hashes={
                "sequence.py": "99c80047d517113579e04fd57443471a759a738dc30b206ba1342e0d4f9b3da2",
                "unmasking.py": "bbc3c3328e5497aa08aed6e04150ab0383f95dc06b4ff570a69adaed4a769746",
            }).items()) and policy in meta["policies"], run)
        require(read(run / f"complete-{policy}.json") == dict(policy=policy, requests=90), run)
        require(len(list(run.glob(f"[0-9][0-9]-{policy}.json"))) == 90, run)
        qual = read(run / "qualified-logits.json")
        require(qual["first_step_logits"] == "bit_identical" and qual["index"] == 0
                and qual["chunk"] == (256 if archived else 512), qual)
        if archived:
            old = [args.original / "ibs" / f"{i:02d}-{policy}.json" for i in range(30)]
            require(read(run / f"reproduced30-{policy}.json") ==
                    dict(policy=policy, matched=30, reference_sha256=[sha(p) for p in old]), run)
            paired = [r for r in rows(args.data / "optimus-scores" / f"gemma-{policy}-zero-paired.csv")
                      if int(r["window"]) == 90]
            require(len(paired) == 90, policy)
            pair = {int(r["index"]): r for r in paired}
        else:
            require(meta["policies"] == [policy] and meta["chunk"] == 512
                    and meta["extended_window_enabled"] is False, run)
        arm = []
        for t in tasks:
            i = t["index"]
            path = run / f"{i:02d}-{policy}.json"
            row = read(path)
            ids = {k: t[k] for k in ("index", "task_id", "dataset", "seed")}
            require(all(row[k] == v for k, v in dict(ids, model="gemma", policy=policy,
                                                       status="success").items()), path)
            if archived and i < 30:
                require(row == read(old[i]), path)
            events, tr = row["events"], row["trace"]
            require(events[0][0] == "encoder" and events[0][1] > 0 and tr, path)
            dec = [q for tag, q in events[1:] if tag == "decoder"]
            enc = [q for tag, q in events[1:] if tag == "encoder"]
            require(len(dec) + len(enc) == len(events)-1 and all(q == 256 for q in enc)
                    and dec == [s["q"] for s in tr] and all(
                        0 < s["q"] <= 256 and not s["cross"] and 0 <= s["accepted"] <= s["q"]
                        and s["block"] * 256 < s["visible_end"] <= (s["block"]+1)*256 for s in tr), path)
            n, f, q = row["output_tokens"], len(events)-1, sum(enc) + sum(dec)
            require(f == row["generation_forwards"] and q == row["generation_query_rows"]
                    == row["generation_physical_padded_rows"] and n/f == row["tpf"]
                    and n/q == row["token_efficiency"] and row["max_decoder_q"] == max(dec)
                    and row["max_q"] == max(dec+enc), path)
            seq = row["output_ids"]
            require(n <= len(seq) <= 2048 and row["eos"] != row["truncated"]
                    and (row["eos"] or len(seq) == 2048), path)
            correct = int(score(row["output_text"], t)["correct"])
            if archived:
                pr = pair[i]
                require(pr["baseline_policy"] == policy and pr["model"] == "gemma" and
                        (pr["task_id"], pr["dataset"], int(pr["seed"])) ==
                        (t["task_id"], t["dataset"], t["seed"]) and
                        tuple(int(pr[f"baseline_{key}"]) for key in ("correct", "N", "F", "Q")) ==
                        (correct, n, f, q), path)
            else:
                ref = read(args.data / "gemma-optimus-primary90" / f"{i:02d}-ibs_256.json")
                require(all(ref[k] == v for k, v in ids.items()), path)
                comparisons.append(dict(ids, output_ids_equal=(seq == ref["output_ids"]),
                                         events_equal=events == ref["events"], trace_equal=tr == ref["trace"]))
            arm.append(dict(ids, policy=policy, raw_sha256=sha(path), correct=correct, N=n, F=f,
                            Q=q, decoder_Q=sum(dec), prefill_Q=events[0][1], reencode_Q=sum(enc),
                            decoder_F=len(dec), max_decoder_Q=max(dec), over_C=sum(w > c for w in dec),
                            eos=row["eos"]))
        ledger.extend(arm)
        for cohort, part in (("original-0-29", arm[:30]), ("added-30-89", arm[30:]), ("all-0-89", arm)):
            n, f, q = (sum(r[k] for r in part) for k in ("N", "F", "Q"))
            entry = dict(policy=policy, cohort=cohort, requests=len(part), N=n, F=f, Q=q,
                         correct=sum(r["correct"] for r in part),
                         macro_tpf=mean(r["N"]/r["F"] for r in part), efficiency=n/q,
                         decoder_Q=sum(r["decoder_Q"] for r in part),
                         reencode_Q=sum(r["reencode_Q"] for r in part),
                         max_decoder_Q=max(r["max_decoder_Q"] for r in part),
                         over_C=sum(r["over_C"] for r in part), cross=0)
            if archived and args.published:
                pub, = (r for r in rows(args.published) if r["q"] == str(c)
                        and r["method"] == "IBS" and r["cohort"] == cohort)
                require((n, f, q, entry["correct"]) == tuple(int(pub[k])
                        for k in ("N", "F", "physical_Q", "correct")) and math.isclose(
                        entry["macro_tpf"], float(pub["macro_tpf"]), rel_tol=1e-12) and
                        math.isclose(n/q, float(pub["pooled_token_efficiency"]), rel_tol=1e-12), cohort)
            results.append(entry)
            print(json.dumps(entry), flush=True)
    save_json(args.output / "requests.json", ledger)
    if comparisons:
        save_json(args.output / "vs-archived-ibs_256.json", comparisons)
    save_json(args.output / "summary.json", results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "original", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--new", type=Path, help="Completed real C512 run")
    p.add_argument("--published", type=Path, help="Published four-tier gemma-optimus.csv")
    p.add_argument("--chunks", type=int, choices=(32, 64, 128, 256, 512),
                   nargs="+", default=[32, 64, 128, 256])
    args = p.parse_args()
    if 512 in args.chunks and not args.new:
        p.error("--new required for C512")
    audit(args)
