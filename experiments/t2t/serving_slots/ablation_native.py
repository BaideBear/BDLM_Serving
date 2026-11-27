"""Read-only frozen90 native reference and paired report for the 00/10/11 GPU test.

Run ``python -B -m experiments.t2t.serving_slots.ablation_native`` for native
results only. --rescore also runs the frozen scorer (including sandboxed
HumanEval); neither mode loads model weights. Native is a quality/work
reference, not a same-hardware latency or same-cap control.
Use --study DIR --model NAME --q TIER in the model's tokenizer environment to
compare one tier of a completed block-ablation-v2 run; do not use partials.

Integration: load_native(model) -> (tasks, references); compare(model, tasks,
references, arms, processor) takes three streams of request rows at ONE tier.
The parent verifies its generation seals, hardware, and checkpoint before
passing rows. See candidate() for the normalized receipt. No requested offer
or nominal batch size is accepted as evidence of physical work.
"""
import argparse
import csv
import gzip
import hashlib
import json
from math import isclose
from pathlib import Path
from statistics import fmean

from bdlm_serving.evaluation import scoring
from bdlm_serving.evaluation.frozen90 import load_tasks
from tools import prefix_native as native

ROOT = Path("/data1/project_data/gx-erqw-20261123/runtime/t2t-native-1125")
ID = ("index", "dataset", "task_id", "seed")
ARMS = ("00", "10", "11")
SCORER = "a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692"
SAMPLES = "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed"
PINS = {
    "manifest.json": "1294449151750ab00a6ee843f0cbbb31ef1f507fe3c854a4c372661d636a0a86",
    "prepared.json": "da8cf31346e457fbc6108c8e9e124ca06f46f05dc432da6b43ba0d6e783489de",
    "verified.json": "c25be5a470dff608cc7d3e030502f47dea80cf8301a78f2627bd04d08c8aec4f",
    "capture-llada21.json": "5a4ac0f89797491b1a0b8c34cfc71f330f8e9dacb40f3ed6b88a2f9b1d58dfd5",
    "capture-gemma.json": "d0f5ba4fbf7506258bc897e9e4836dcedbbb369c914a891b7e07785903f2eb86",
    "native-gemma-paired.csv": "76f437ab6d11b89442a152fa5d7a8efd13e8fc65ab59a13ea0825421aab82845",
    "native-llada21-paired.csv": "c634c01181489bd37a12c2b20bd33d0b648f560b2c4008504a78042270eb8888",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pinned(path):
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == PINS[path.name], f"Changed archive receipt: {path}")
    return data


def id_of(row):
    return tuple(row[k] for k in ID)


def ids_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def load_native(model, root=ROOT, *, rescore=False):
    """Validate original30 + captured60, per-case scorer receipts and physical tape."""
    policy, b = native.MODELS[model]
    data, archive = Path(root)/"prefix90", Path(root)/"frozen90-native"
    check_scorer()
    tasks = load_tasks(data/"samples.json")
    manifest = json.loads(pinned(archive/"manifest.json"))
    require(native.sha256(data/"samples.json") == manifest["samples_sha256"] == SAMPLES and
            manifest["models"][model]["policy"] == policy, "Native cohort/registration mismatch")
    receipts = (json.loads(pinned(archive/"prepared.json"))["original30"][model] +
                json.loads(pinned(archive/f"capture-{model}.json"))["records"])
    verified = json.loads(pinned(archive/"verified.json"))["models"][model]
    require(verified["original30_preserved"] and verified["requests"] == 90, "Incomplete native capture")
    csv_rows = csv.DictReader(pinned(data/f"native-{model}-paired.csv").decode().splitlines())
    scores = [s for s in csv_rows if s["window"] == "90"]
    rows = []
    for t, rec, v, s in zip(tasks, receipts, verified["rows"], scores, strict=True):
        i = t["index"]
        raw = (archive/model/f"{i:02}-{policy}.json").read_bytes()
        require(hashlib.sha256(raw).hexdigest() == rec["copy_sha256"] == rec["source_sha256"],
                f"Native generation bytes differ: {model}/{i}")
        r = json.loads(raw)
        if model == "llada21":
            native.verify_llada(t, r)
            require(r["phase"] == "formal" and r["generation_cap"] == 2048 and
                    all(c["batch"] == 1 and c["rows"] == c["physical_rows"] for c in r["calls"]) and
                    all(e["phase"] in ("denoise", "commit") and e["q_tokens"] == b
                        for e in r["events"][1:]), f"Non-native decoder tape: {i}")
        else:
            native.verify_gemma(t, r)
        require(id_of(r) == id_of(rec) == id_of(t) and
                all(str(t[k]) == s[k] for k in ID) and all(v[k] == r[k] == rec[k] for k in
                ("index", "seed", "task_id", "policy", "output_tokens", "generation_forwards",
                 "generation_query_rows")), f"Native generation/score identity mismatch: {model}/{i}")
        require(s["model"] == model and s["baseline_policy"] == s["arm_policy"] == policy and
                s["baseline_correct"] == s["arm_correct"] in ("0", "1"), f"Wrong self-pair: {i}")
        n, f, q = r["output_tokens"], r["generation_forwards"], r["generation_physical_padded_rows"]
        correct = s["baseline_correct"] == "1"
        for side in ("arm_", "baseline_"):
            require((n, f, q, int(r["truncated"])) ==
                    tuple(int(s[side+k]) for k in ("N", "F", "Q", "truncated")) and
                    isclose(n/f, float(s[side+"tpf"])) and isclose(n/q, float(s[side+"token_efficiency"])),
                    f"Frozen score and measured work disagree: {model}/{i}")
        if rescore:
            require(scoring.score(r["output_text"], t)["correct"] == correct, f"Native rescore: {model}/{i}")
        rows.append(dict(zip(ID, id_of(t)), model=model, N=n, F=f, Q=q, correct=correct,
                         input_sha256=ids_digest(r["prompt_ids"]) if model == "llada21" else None,
                         prefill_Q=r["prefill_rows"], encoder_F=r.get("reencode_forwards", 0),
                         encoder_Q=r.get("reencode_query_rows", 0)))
    return tasks, rows


def check_scorer():
    require(native.sha256(scoring.__file__) == SCORER, "Unexpected frozen scorer")


def candidate(model, task, ref, row, arm, tier, processor):
    """Parent receipt: ID, model, arm, q, cap, input_sha256, output_ids, N/F/Q,
    correct, scorer_sha256, status (success/generation_failed), generation_calls.
    One [decoder|encoder, useful_rows, physical_padded_rows] per real call for
    this request, generation only. Include every post-prefill reencode/drain.
    Counts are request-attributed, never global batched dispatch throughput.
    processor is the parent-verified local tokenizer (LLaDA) or processor (Gemma).
    """
    b = native.MODELS[model][1]
    who = f"{model}/{task['index']}/{arm}"
    require(id_of(row) == id_of(task) == id_of(ref) and ref["model"] == model and
            (row["arm"], row["model"], row["q"], row["cap"], row["scorer_sha256"]) ==
            (arm, model, tier, 2048, SCORER), f"Unmatched generation protocol: {who}")
    if model == "llada21":
        tok = processor
        digest = ref["input_sha256"]
        special = set(tok.all_special_ids) | {tok.eos_token_id, tok.convert_tokens_to_ids("<|mask|>")}
    else:
        from .mask_pair_gemma import input_digest
        # The native Gemma row stores only prefill length; reconstruct the full
        # registered processor output. A length match alone cannot establish pairing.
        inp = processor.apply_chat_template([dict(role="user", content=task["prompt"])],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        require(inp["input_ids"].shape[1] == ref["prefill_Q"], f"Gemma native input length: {who}")
        digest, tok = input_digest(inp), processor.tokenizer
        special = set(tok.all_special_ids) | {0, 1, 50, 106}
    require(digest == row["input_sha256"], f"Unmatched generation input: {who}")
    tape = row["generation_calls"]
    require(tape and any(c[0] != "encoder" for c in tape) and
            all(typ in ("encoder", "decoder", "full_build") and type(u) is int and type(p) is int
            and 0 < u <= p and (p <= tier if typ == "decoder" else
                                arm == "11" and tier < 2*b and p <= b if typ == "full_build" else
                                model == "gemma" and u == p == b)
            for typ, u, p in tape), f"Invalid physical work: {who}")
    ids = row["output_ids"]
    require(isinstance(ids, list) and len(ids) <= 2048 and all(type(x) is int and x >= 0 for x in ids),
            f"Invalid output: {who}")
    n, f, q = sum(x not in special for x in ids), len(tape), sum(p for _, _, p in tape)
    require(all(type(row[k]) is int for k in ("N", "F", "Q")) and (n, f, q) ==
            (row["N"], row["F"], row["Q"]), f"Unpaid work or wrong N: {who}")
    if row["status"] == "generation_failed":
        require(not ids, f"Failed request delivered tokens: {who}")
        correct = False
    else:
        eos = {tok.eos_token_id} if model == "llada21" else {1, 50, 106}
        require(row["status"] == "success" and ids and not eos.intersection(ids[:-1]) and
                (ids[-1] in eos or len(ids) == 2048), f"Not first EOS or cap: {who}")
        correct = scoring.score(tok.decode(ids, skip_special_tokens=True), task)["correct"]
    require(type(row["correct"]) is bool and row["correct"] == correct, f"Frozen rescore mismatch: {who}")
    return dict(zip(ID, id_of(task)), N=n, F=f, Q=q, correct=correct)


def summary(rows):
    n, f, q = (sum(r[k] for r in rows) for k in ("N", "F", "Q"))
    return dict(requests=len(rows), correct=sum(r["correct"] for r in rows), N=n, F=f, Q=q,
                macro_TPF=fmean(r["N"]/r["F"] for r in rows), pooled_TPF=n/f, efficiency=n/q)


def compare(model, tasks, references, arms, processor):
    """All three same-tier arms, paired by frozen identity (not completion order)."""
    check_scorer()
    require(set(arms) == set(ARMS) and len(tasks) == len(references) == 90 and
            [t["index"] for t in tasks] == list(range(90)), "Expected frozen90 and three arms")
    report, tier = {}, None
    for arm in ARMS:
        rows = list(arms[arm])
        cases = {r["index"]: r for r in rows}
        require(len(rows) == 90 and set(cases) == set(range(90)), f"Duplicate or missing {arm} case")
        if tier is None:
            tier = cases[0]["q"]
            b = native.MODELS[model][1]
            require(type(tier) is int and tier in (b//8, b//4, b//2, b, 2*b), "Unregistered tier")
        measured = [candidate(model, t, ref, cases[t["index"]], arm, tier, processor)
                    for t, ref in zip(tasks, references, strict=True)]
        report[arm] = {}
        for scope in ("all", "original30", "new60", "gsm8k", "math500", "humaneval"):
            pairs = [(a, n) for a, n in zip(measured, references, strict=True)
                     if scope == "all" or a["dataset"] == scope or
                     (scope == "original30" and a["index"] < 30) or
                     (scope == "new60" and a["index"] >= 30)]
            a, n = map(summary, zip(*pairs))
            report[arm][scope] = dict(arm=a, native=n, rescued=sum(x["correct"] and not y["correct"]
                for x, y in pairs), lost=sum(y["correct"] and not x["correct"] for x, y in pairs),
                delta_F=a["F"]-n["F"], delta_Q=a["Q"]-n["Q"], delta_macro_TPF=a["macro_TPF"]-n["macro_TPF"])
    return dict(model=model, q=tier, by_arm=report)


def compare_study(study, model, tier, tasks, refs, processor):
    """Consume the registered batch-one runner format, not a synthetic padded batch."""
    meta_bytes = (study/"run.json").read_bytes()
    stamp = hashlib.sha256(meta_bytes).hexdigest()
    meta = json.loads(meta_bytes)
    seal = native.read_json(study/"complete.json")
    b = native.MODELS[model][1]
    tiers = (b//8, b//4, b//2, b, 2*b)
    names = {f"{i:02}-q{q}-{a}.json.gz" for i in range(90) for a in ARMS for q in tiers}
    require(meta["protocol"] == "block-ablation-v2" and meta["model"] == model and meta["device"] == "cuda"
            and meta["physical_request_batch"] == 1 and meta["maximum_output"] == 2048
            and meta["dense_top_all"] is True and meta["scorer"] == SCORER and meta["samples"] == SAMPLES
            and sorted(meta["order"]) == list(range(90)) and sorted(meta["tiers"]) == list(tiers)
            and sorted(meta["arms"]) == list(ARMS) and tier in tiers and seal["registration"] == stamp
            and seal["rows"] == len(names) and set(seal["cases"]) == names, "Incomplete/different study")
    arms, boundary = {a: [] for a in ARMS}, []
    for a in ARMS:
        for i in range(90):
            name = f"{i:02}-q{tier}-{a}.json.gz"
            with gzip.open(study/name, "rt") as f:
                r = json.load(f)
            require(r["index"] == i and r["q"] == tier and r["arm"] == a and r["registration"] == stamp
                    and all(r[k] == seal["cases"][name][k] for k in ("N", "F", "Q", "correct", "TPF", "efficiency"))
                    and r["score"]["correct"] is r["correct"], f"Unsealed or conflicting case: {name}")
            work = []
            for tick, event in enumerate(r["trace"]):
                w = event["q"]
                pos = next((event[k] for k in ("positions", "absolute_positions", "rows") if k in event), None)
                require(event["tick"] == tick and event["offer"] == tier and pos and
                        len(set(pos)) == len(pos) == w, f"Missing/incorrect decoder measurement: {name}:{tick}")
                build = event.get("route", event.get("phase")) == "build_sample"
                if build:
                    offset = r["prefill_Q"] if model == "gemma" else 0
                    require(a == "11" and tier < 2*b and (min(pos)-offset)%b == 0 and len(pos) <= b
                            and sorted(pos) == list(range(min(pos),max(pos)+1)), "Malformed full build")
                work.append(("full_build" if build else "decoder", w, w))
                if model == "llada21":
                    require(event["layer_physical_q"] == [w]*20, f"Projection mismatch: {name}:{tick}")
                else:
                    require(type(event["encoded"]) is bool, f"Missing encoder observation: {name}:{tick}")
                    if event["encoded"]:
                        work.append(("encoder", b, b))
            if tier == 2*b:
                for e in r.get("boundary_parity_events", ()):
                    boundary.append(dict(e, **{k:r[k] for k in ID}, arm=a, q=tier))
            require(r["projection_modules"] == (20 if model == "llada21" else 85) and
                    r["encoder_Q"] == sum(p for kind, _, p in work if kind == "encoder") and
                    r["decoder_Q"] == sum(p for kind, _, p in work if kind != "encoder") and
                    len(work) == r["F"] and r["TPF"] == r["N"]/r["F"] and
                    r["efficiency"] == r["N"]/r["Q"], f"Inconsistent physical work: {name}")
            arms[a].append(dict({k:r[k] for k in (*ID, "arm", "q", "input_sha256", "output_ids",
                                                   "correct", "N", "F", "Q")}, model=model,
                                cap=2048, scorer_sha256=SCORER, generation_calls=work,
                                status="generation_failed" if r.get("failure") else "success"))
    result = compare(model, tasks, refs, arms, processor)
    if tier == 2*b:
        result["boundary_parity"] = boundary_parity(model, boundary)
    return result


def boundary_parity(model, events):
    """Eligible two-block q=2B boundary receipts only (not every decode call).
    state_sha256 covers tokens, KV, positions, mask, pending debt and RNG before
    the call. transition_sha256 covers arm-independent sampled/written state
    and KV after it. Parent supplies measured forward widths, not offers.
    """
    b = native.MODELS[model][1]
    states, seen = {}, set()
    for e in events:
        arm = e["arm"]
        require(arm in ARMS and e["q"] == 2*b and e["decoder_widths"] == [2*b] and
                e["projected_blocks"] == 2, "All three arms require one dense joint boundary call")
        for key in ("state_sha256", "transition_sha256"):
            value = e[key]
            require(isinstance(value, str) and len(value) == 64 and
                    all(c in "0123456789abcdef" for c in value), "Missing state/transition digest")
        group = states.setdefault((id_of(e), e["state_sha256"]), {})
        require(not group or e["transition_sha256"] == next(iter(group.values())),
                "Different transition from an identical q=2B input state")
        group[arm] = e["transition_sha256"]
        seen.add(arm)
    matched = sum(set(g) == set(ARMS) for g in states.values())
    return dict(arms_seen=sorted(seen), same_state_triples=matched, same_state_pairs_only=
                sum(len(g) == 2 for g in states.values()), unmatched_states=sum(len(g) == 1 for g in states.values()),
                three_arm_parity_observed=bool(matched))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--model", choices=native.MODELS)
    p.add_argument("--rescore", action="store_true")
    p.add_argument("--study", type=Path)
    p.add_argument("--q", type=int)
    args = p.parse_args()
    if (args.study is None) != (args.q is None) or (args.study and not args.model):
        p.error("A comparison requires --study, --model and --q together")
    for model in ([args.model] if args.model else native.MODELS):
        tasks, refs = load_native(model, args.root, rescore=args.rescore)
        if args.study:
            ckpt = json.loads(pinned(args.root/"frozen90-native"/"manifest.json"))["models"][model]["checkpoint"]
            proc = (native.load_llada_tools if model == "llada21" else native.load_gemma_tools)(ckpt)[0]
            print(json.dumps(compare_study(args.study, model, args.q, tasks, refs, proc), allow_nan=False))
            continue
        print(json.dumps(dict(model=model, score_source="rescored" if args.rescore else "pinned_per_case",
                              **summary(refs)), allow_nan=False))
