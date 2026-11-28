"""Verify the Q16 paired receipts with no model or scorer."""
import argparse
import json
from pathlib import Path

from tools.llada_ibs_eos_eval import CKPT, DATA, compact, digest, file_sha, read
from tools.llada_ibs_eos_q16_eval import ARMS, LABEL, Q, SIX, audit, first_fork, require
from tools.llada_ibs_eos_raw_audit import EOS, MASK, replay


def verify(out):
    reg = read(out/"registration.json")
    reg_id = digest(reg)
    notes = [json.loads(line) for line in (out/"receipts.ndjson").read_text().splitlines()]
    require(tuple(n["index"] for n in notes) == SIX, "Six-case run incomplete or wrong panel")
    excluded = set(reg["excluded"])
    declared = {t["id"] for t in read(CKPT/"tokenizer.json")["added_tokens"] if t["special"]}
    require(declared | {EOS, MASK} <= excluded, "Missing registered special-token exclusion")
    equal_ids = []
    for n in notes:
        i = n["index"]
        f = out/"pairs"/f"{i:02}.json"
        a = read(f)
        require(file_sha(f) == n["raw_sha"] and a["reg"] == n["reg"] == reg_id and
                a["order"] == n["order"], f"{i}: archive hash or run identity drift")
        rows, tapes = a["arms"], a["tapes"]
        stock, tracked, apply = (rows[k] for k in ARMS)
        require(stock == tracked and all(tapes["stock"][k] == tapes["tracked"][k]
                for k in ("certificate", "writes")), f"{i}: observer altered paid stock")
        hist = read(DATA/f"llada21-optimus-hardcap-endpoints/{i:02}-{LABEL}.json")
        closed = tapes["stock"]["certificate"]
        prefix = closed["event"]+1 if closed else len(stock["events"])
        require(hist["output_ids"] == stock["output_ids"] and hist["prompt_ids"] == stock["prompt_ids"]
                and hist["events"][:prefix] == stock["events"][:prefix], f"{i}: historical stock mismatch")
        for arm in ARMS:
            t = tapes[arm]
            raw = replay(rows[arm], t, q=Q)
            if raw is not None:
                raw["debt_after"] = []
            require(raw == t["certificate"], f"{i}: {arm} reconstructed certificate mismatch")
            require(all(n["arms"][arm][k] == v for k,v in audit(
                rows[arm], t, arm, eos=EOS, mask=MASK, excluded=excluded).items()), f"{i}: {arm} receipt mismatch")
            for obs in t["ticks"]:
                obs["applied"] = tuple(obs["applied"])
        require(first_fork(tracked, apply, tapes["tracked"], tapes["apply"]) == n["first"],
                f"{i}: first-fork receipt mismatch")
        if apply["output_ids"] == stock["output_ids"]:
            equal_ids.append(i)
    final = read(out/"complete-six.json")
    totals = {arm:{k:sum(n["arms"][arm][k] for n in notes) for k in
                    ("F", "Q", "N", "extra", "eligible", "paid_D", "paid_Q", "correct")} for arm in ARMS}
    losses = [n["index"] for n in notes if n["arms"]["stock"]["correct"] and not n["arms"]["apply"]["correct"]]
    failures = [n["index"] for n in notes if n["arms"]["apply"]["failure"] and not n["arms"]["stock"]["failure"]]
    benefit = all(totals["apply"][k] < totals["stock"][k] for k in ("F", "Q")) and not losses and not failures
    require(final == dict(reg=reg_id, indices=list(SIX), totals=totals, losses=losses,
                new_failures=failures, benefit_gate=benefit, first_forks=sum(n["first"] is not None for n in notes)),
            "Frozen panel total or gate mismatch")
    return dict(status="pass", stock_apply_equal_ids=equal_ids, **final,
                sha256={k:file_sha(out/k) for k in ("complete-six.json", "receipts.ndjson",
                        "registration.json", "returned.identity")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    print(compact(verify(parser.parse_args().result)))
