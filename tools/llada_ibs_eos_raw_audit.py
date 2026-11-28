"""Read-only Q4 archive audit, with no model, torch, or scorer imports."""
import argparse
import json
from pathlib import Path

from tools.llada_ibs_eos_eval import (
    ARMS, CAP, DATA, KEYS, LABEL, ORDER, Q, SIX, UUID, compact, digest, file_sha, qualify, read, require, summary,
)
from tools.llada_ibs_eos_stop import B

EOS, MASK = 156892, 156895


def replay(row, tape, *, q=Q):
    """Reconstruct the EOS search from charged input IDs and observed write IDs."""
    p = len(row["prompt_ids"])
    live = dict(enumerate(row["prompt_ids"]))
    initialized, due = set(), {}
    F = Q_sum = tick = 0
    found = None
    for j, (e, c, ids) in enumerate(zip(row["events"], row["calls"], row["forward_inputs"], strict=True)):
        positions = e["absolute_positions"]
        require(positions == sorted(set(positions)) and
                len(ids) == len(positions) == e["q_tokens"] == c["rows"] == c["physical_rows"] and
                c["batch"] == e["physical_forwards"] == 1, "Incorrect physical ledger")
        for pos, token in zip(positions, ids, strict=True):
            if tape is not None or e["phase"] == "prefill":
                require(live.get(pos, MASK) == token, "Forward did not consume the observed canvas")
        initialized.update(positions)
        if e["phase"] == "prefill":
            require(j == 0 and positions == list(range(p//B*B)), "Invalid prefill")
            continue
        F += 1
        Q_sum += len(positions)
        require(0 < len(positions) <= q and c["kv_rows"] == max(positions)+1 == e["visible_kv"],
                "Incorrect physical cap or K/V extent")
        require(sorted(due) == e["dirty_before"] and due.keys() <= set(positions), "Lost ordinary debt")
        if e["phase"] == "certificate_debt":
            require(found is not None and found["event"] == j-1 and positions == sorted(due) and
                    j == len(row["events"])-1 and not (e["written_absolute"] or
                    e["edited_positions"] or e["sampled_absolute"]), "Invalid terminal payment")
            due = {}
            continue
        require(e["phase"] == "optimus" and e["offer_tick"] == tick and e["physical_cap"] == q,
                "Missing ordinary tick")
        start = e["block"] * B
        before, after = e["cached_prefix_before"], e["cached_prefix_after"]
        require(all(start+before <= pos < start+B for pos in positions), "Projection crossed IBS frontier")
        due = dict.fromkeys(e["written_absolute"])
        require(all(pos >= max(p, start+after) and pos in positions for pos in due),
                "Unpaid write retired or unmeasured write")
        if tape is not None:
            paid_writes = dict(tape["writes"][tick])
            require(due.keys() == paid_writes.keys(), "Write IDs do not cover physical debt")
            live.update(paid_writes)
            for pos in set(e["edited_positions"])-due.keys():
                live[pos] = MASK
            due = paid_writes
            frozen = range(start, start+after)
            require(all(pos in initialized and live.get(pos, MASK) != MASK for pos in frozen),
                    "Frozen frontier included uninitialized or MASK context")
            eos_pos = next((pos for pos in range(max(p, start), min(p+CAP, start+after)) if live[pos] == EOS), None)
            if found is None and eos_pos is not None:
                found = dict(event=j, eos=eos_pos, block=e["block"], step=e["step"], ready=after,
                             debt_before=sorted(due))
        tick += 1
    if tape is not None:
        require(tick == len(tape["writes"]), "Wrong write tape length")
    require((not due or row["request_failure"]) and row["verified_layers"] == 20 and
            F == row["generation_forwards"] and Q_sum == row["generation_query_rows"] ==
            row["generation_physical_padded_rows"], "Inconsistent final ledger")
    return found


def audit(out):
    reg = read(out/"registration.json")
    require(reg["order"] == list(ORDER) and reg["uuid"] == UUID and reg["panel"]["q"] == Q and
            reg["panel"]["samples_sha256"] == file_sha(DATA/"samples.json") and
            reg["panel"]["ledger_sha256"] == file_sha(DATA/"fixed-family-reference-case-ledger-v1.json"),
            "Original registration or historical bills drifted")
    reg_id = digest(reg)
    ledger = read(DATA/"fixed-family-reference-case-ledger-v1.json")["arms"]["llada21:"+LABEL]["items"]
    notes = [json.loads(line) for line in (out/"receipts.ndjson").read_text().splitlines()]
    require(tuple(n["index"] for n in notes) == ORDER, "Wrong or repeated cases")
    require({p.name for p in (out/"pairs").glob("*.json")} == {f"{i:02}.json" for i in ORDER},
            "Unexpected case files")
    equal = 0
    for n in notes:
        i = n["index"]
        f = out/"pairs"/f"{i:02}.json"
        require(file_sha(f) == n["raw_sha"] and n["reg"] == reg_id, f"{i}: receipt hash mismatch")
        a = read(f)
        require(a["reg"] == reg_id and a["order"] == n["arm_order"], f"{i}: arm or registration drift")
        rows, tapes = a["arms"], a["tapes"]
        try:
            for arm in ARMS:
                r = rows[arm]
                found = replay(r, tapes[arm] if arm != "stock" else None)
                if found is not None and arm == "certificate":
                    found["debt_after"] = []
                require(found == tapes[arm]["certificate"], "Stored first certificate differs from replay")
                receipt = n["arms"][arm]
                require(all(r[k] == n[k] for k in KEYS) and digest(r["prompt_ids"]) == n["input_sha"] and
                        digest(r["output_ids"]) == receipt["ids_sha"] and
                        (r["generation_forwards"], r["generation_query_rows"], r["output_tokens"],
                         r["request_failure"], r["eos"]) ==
                        tuple(receipt[k] for k in ("F", "Q", "N", "failure", "eos")), "Wrong arm receipt")
            stock, observed, cert = (rows[k] for k in ARMS)
            old = read(DATA/f"llada21-optimus-hardcap-low/{i:02}-{LABEL}.json")
            require(all(stock[k] == old[k] for k in (*KEYS, "prompt_ids", "output_ids", "events",
                        "extra", "status", "request_failure")), "Historical stock changed")
            require(all(n["arms"]["stock"][k] == ledger[i][k] for k in ("F", "Q", "N", "correct")),
                    "Historical bill or correctness changed")
            require(qualify(stock, observed, cert, tapes["instrumented"], tapes["certificate"], EOS, MASK)
                    == n["certificate"], "Paired certificate differs from receipt")
            cohort = ("stock_success" if not stock["request_failure"] else
                      "stock_failure_certified" if n["certificate"] else "stock_failure_retained")
            require(cohort == n["cohort"], "Incorrect failure cohort")
            if cohort == "stock_success":
                require(stock["output_ids"] == cert["output_ids"], "Unequal successful return")
                equal += 1
        except ValueError as err:
            raise ValueError(f"{i}: {err}") from err
    six = summary(notes[:len(SIX)], reg_id)
    full = summary(notes, reg_id)
    require(six == read(out/"qualified-six.json") and full == read(out/"complete.json"), "Aggregate drift")
    return dict(status="pass", replayed=len(notes), stock_success_exact=equal, reg=reg_id,
                sha256={name:file_sha(out/name) for name in ("complete.json", "receipts.ndjson",
                        "registration.json", "qualified-six.json", "returned.identity")}, **full["cohorts"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    print(compact(audit(parser.parse_args().result)))
