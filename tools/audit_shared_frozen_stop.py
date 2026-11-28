"""Replay registered strict-IBS GPU tapes; do not run inference or export token IDs."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from bdlm_serving.methods.t2t.frozen_stop import frozen_stop


PILOT = (0, 10, 20, 30, 40, 50)


def llada(pair, eos_id):
    row = pair["arms"]["instrumented"]
    arm = pair["arms"]["certificate"]
    tape = pair["tapes"]["instrumented"]
    it = iter(tape["writes"])
    tokens, dirty, hit = {}, set(), None
    prompt = len(row["prompt_ids"])
    for i, e in enumerate(row["events"]):
        if e["phase"] == "prefill":
            continue
        assert e["phase"] == "optimus"
        start, ready = e["block"] * 32, e["cached_prefix_after"]
        assert set(e["dirty_before"]) == dirty
        dirty.difference_update(e["absolute_positions"])
        values = next(it)
        written = {pos for pos, _ in values}
        assert sorted(written) == e["written_absolute"] and len(values) == len(written)
        tokens.update(values)
        for pos in set(e["edited_positions"]) - written:
            tokens.pop(pos)  # Native edit back to MASK.
        dirty.update(written)
        lo, hi = max(prompt, start), min(prompt + row["generation_cap"], start + ready)
        first = next((p for p in range(lo, hi) if tokens[p] == eos_id), None)
        d = frozen_stop(start=start, width=32, ready=ready, generated_from=prompt,
                        generated_until=prompt+row["generation_cap"], eos_position=first,
                        debt=dirty, offer=e["physical_cap"])
        if d.action != "continue" and hit is None:
            hit = (i, first, ready, d)
    assert next(it, None) is None
    cert = tape["certificate"]
    assert (cert is None) == (hit is None)
    if hit is None:
        assert arm["calls"] == row["calls"] and arm["output_ids"] == row["output_ids"]
        return len(tape["writes"]), "no_frozen_eos", 0
    i, first, ready, d = hit
    assert (cert["event"], cert["eos"], cert["ready"], cert["debt_before"]) == (i, first, ready, list(d.rows))
    assert arm["events"][:i+1] == row["events"][:i+1]
    extra = arm["events"][i+1:]
    assert len(extra) == int(d.action == "pay") and arm["output_ids"][-1] == eos_id
    if extra:
        pay, = extra
        assert (pay["phase"], pay["absolute_positions"], pay["q_tokens"], pay["physical_forwards"]) == (
            "certificate_debt", list(d.rows), len(d.rows), 1)
        assert not pay["written_absolute"] and not pay["sampled_absolute"] and not pay["edited_positions"]
        assert arm["calls"][-1]["physical_rows"] == len(d.rows)
    return len(tape["writes"]), d.reason, len(d.rows)


def gemma(pair):
    base, arm = (pair["arms"][k] for k in ("control", "paid"))
    assert base["ids"] == arm["ids"]
    hit = None
    assert len(base["trace"]) == len(base["states"])
    for e, s in zip(base["trace"], base["states"]):
        start = 256*e["block"]
        p = s["eos_index"]
        d = frozen_stop(start=start, width=256, ready=s["ready"], generated_from=0,
                        generated_until=2048, eos_position=None if p is None else start+p,
                        debt=s["debt"], offer=e["offer"])
        if d.action != "continue" and hit is None:
            hit = (e["tick"], p, s["ready"], d)
    if hit is None or hit[3].reason == "ordinary_block_end":
        assert arm["trigger"] is None and arm["trace"] == base["trace"] and arm["payment"] is None
        return len(base["trace"]), (hit[3].reason if hit else "no_frozen_eos"), 0
    i, first, ready, d = hit
    trigger = arm["trigger"]
    assert (trigger["tick"], trigger["eos_index"], trigger["ready"], trigger["debt"]) == (i, first, ready, list(d.rows))
    assert arm["trace"][:i+1] == base["trace"][:i+1]
    assert len(arm["trace"]) == i+1+int(d.action == "pay")
    if d.action == "pay":
        assert arm["payment"] == dict(tick=i+1, positions=list(d.rows), Q=len(d.rows))
        assert arm["trace"][-1]["positions"] == list(d.rows) and arm["decoder"][-1]["sample"] is None
        assert arm["events"][-1] == ["D", len(d.rows)] and not arm["states"][-1]["debt"]
    else:
        assert arm["payment"] is None
    return len(base["trace"]), d.reason, len(d.rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llada-dir", type=Path, required=True)
    parser.add_argument("--gemma-log", type=Path, required=True)
    parser.add_argument("--llada-tokenizer", type=Path, required=True)
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("Replay requires assertion checks enabled")
    eos_token = json.loads((args.llada_tokenizer/"tokenizer_config.json").read_bytes())["eos_token"]
    eos, = [x["id"] for x in json.loads((args.llada_tokenizer/"tokenizer.json").read_bytes())["added_tokens"]
            if x["content"] == eos_token]
    report = {"indices": list(PILOT), "scope": "GPU pilot tape replay, no new inference"}
    ll = {}
    for i in PILOT:
        raw = (args.llada_dir/"pairs"/f"{i:02}.json").read_bytes()
        p = json.loads(raw)
        assert p["arms"]["instrumented"]["index"] == i
        ll[i] = (llada(p, eos), sha256(raw).hexdigest())
    gg = {}
    with args.gemma_log.open("rb") as f:
        for line in f:
            if not line.startswith(b"RECEIPT "):
                continue
            p = json.loads(line[len(b"RECEIPT "):])
            if p["kind"] == "case":
                i = p["index"]
                assert i in PILOT and i not in gg
                gg[i] = (gemma(p), sha256(line).hexdigest())
            if p["kind"] == "gate_passed":
                assert p["indices"] == list(PILOT)
                break
        else:
            raise ValueError("Registered Gemma six-case gate not found")
    for model, panel in (("llada21_q4", ll), ("gemma_q32", gg)):
        assert set(panel) == set(PILOT)
        cases = [panel[i][0] for i in PILOT]
        report[model] = dict(stock_ticks=sum(x[0] for x in cases),
                             first_decision_counts=dict(Counter(x[1] for x in cases)),
                             actual_payment_forwards=sum(x[2] > 0 for x in cases),
                             actual_payment_Q=sum(x[2] for x in cases),
                             source_sha256=[panel[i][1] for i in PILOT])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
