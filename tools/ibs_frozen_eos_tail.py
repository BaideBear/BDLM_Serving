"""Count historical strict-IBS calls after the delivered EOS was inside its frozen frontier."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


def require(ok, why):
    if not ok:
        raise ValueError(why)


def after_certificate(trace, eos_pos, b, q):
    """Trace tuples are (block, new frozen prefix length, selected positions, post-step debt)."""
    first = clean = None
    cnt = Counter()
    last_block, last_front = -1, 0
    for k, (block, ready, pos, due) in enumerate(trace):
        require(last_block <= block <= last_block+1 or k == 0, "invalid block sequence")
        if block != last_block:
            last_front = 0
        require(last_front <= ready <= b and 0 < len(pos) <= q and len(pos) == len(set(pos)) and
                all(block*b+last_front <= p < (block+1)*b for p in pos) and
                all(block*b+ready <= p < (block+1)*b for p in due), "retired row selected or owed")
        if block == eos_pos//b and block*b+ready > eos_pos:
            if first is None:
                first = k
                cnt["first_debt_Q"] = len(due)
                cnt["first_has_debt"] = int(bool(due))
            if not due and clean is None:
                clean = k
        if first is not None and k > first:
            cnt["following_F"] += 1
            cnt["following_Q"] += len(pos)
        if clean is not None and k > clean:
            cnt["following_clean_F"] += 1
            cnt["following_clean_Q"] += len(pos)
        last_block, last_front = block, ready
    require(first is not None and last_block == eos_pos//b, "missing certificate for delivered EOS")
    cnt.update(certificates=1, with_later_F=int(bool(cnt["following_F"])),
               clean_with_later_F=int(bool(cnt["following_clean_F"])),
               with_room_after_one_payment=int(cnt["following_F"] > cnt["first_has_debt"]))
    return cnt


def analyze(root, model, q):
    if model == "llada21":
        b = 32
        group = root / ("llada21-optimus-hardcap-" + {4:"low",8:"endpoints",16:"endpoints",32:"anchor"}[q])
        policy = f"optcap_ibs_{q}"
    else:
        b = 256
        group = (root/"gemma-fixed-caps-q32q64-v1" if q in (32,64) else
                 root.parent/"private-optimus-cap90-v1/gemma-optimus-cap90-v1")
        policy = f"ibs_cap_{q}"
    blob = (root/"fixed-family-reference-case-ledger-v1.json").read_bytes()
    seal = sha256(blob).hexdigest()
    require(seal == "ff9caccbf776416f0237ddcc73af774fbe567c57929b359583f70fba128d64ce", "ledger changed")
    refs = json.loads(blob)["arms"][model+":"+policy]["items"]
    require(len(refs) == 90, "incomplete frozen90")
    cohorts = {}
    digest = sha256()
    for i, ref in enumerate(refs):
        blob = (group/f"{i:02}-{policy}.json").read_bytes()
        digest.update(sha256(blob).digest())
        r = json.loads(blob)
        require(r["policy"] == policy and r["model"] == model and ref["index"] == i and
                all(r[k] == ref[k] for k in ("index","task_id","dataset","seed")) and
                (r["output_tokens"], r["generation_forwards"], r["generation_query_rows"]) ==
                (ref["N"], ref["F"], ref["Q"]), "wrong baseline input, identity, or bill")
        if model == "llada21":
            prompt = len(r["prompt_ids"])
            trace = [(e["block"], e["cached_prefix_after"], e["absolute_positions"], e["written_absolute"])
                     for e in r["events"] if e["phase"] != "prefill"]
        else:
            prompt = 0
            trace = []
            due = set()
            for e in r["trace"]:
                pos = e["positions"]
                due = (due-set(pos)) | set(e["admitted"])
                trace.append((e["block"], e["ready"], pos, due))
        c = Counter(requests=1, original_F=ref["F"], original_decoder_Q=sum(len(t[2]) for t in trace))
        if r["status"] == "success" and r["eos"]:
            c.update(after_certificate(trace, prompt+len(r["output_ids"])-1, b, q))
        else:
            c["failed_or_no_eos"] += 1
        for name in ("all", "original30" if i < 30 else "new60"):
            cohorts.setdefault(name, Counter()).update(c)
    return dict(model=model, q=q, reference_sha256=seal, records_sha256=digest.hexdigest(), cohorts=cohorts)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    a = p.parse_args()
    print(json.dumps([analyze(a.root,model,q) for model,tiers in
          (("llada21",(4,8,16,32)),("gemma",(32,64,128,256))) for q in tiers],indent=2))
