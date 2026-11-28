"""Extract consented compact measurements only; never copy prompt/model output or raw KV."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seals(records):
    worker = start = None
    for obj in records:
        kind = obj["kind"]
        if kind == "worker":
            if worker:
                raise ValueError("Mixed workers")
            worker = obj
        elif kind == "state_start":
            start = obj
        elif kind == "state_seal":
            if not (worker and start and obj["identical_pre_state"] and obj["completed"] == 5
                    and len(obj["arms"]) == 5 and {a["fork_q"] for a in obj["arms"]} == {4, 8, 16, 32, 64}):
                raise ValueError("Unpaired fork")
            arms = sorted(obj["arms"], key=lambda a: a["fork_q"])
            first = arms[0]
            state = first["pre_state"]
            dd = digest(state)
            features = dict(state["features"])
            for key in ("head_mask_rows", "dirty_rows", "resident_rows"):
                v = features[key]
                if len(v) != len(set(v)):
                    raise ValueError("Duplicate state row")
                features[key] = len(v)
            out = dict(request_id=first["task_id"], model="llada21",
                       executor_id=worker["executor_id"]+"@"+worker["source_sha256"],
                       continuation_q=start["continuation_q"], state=features, state_digest=dd,
                       pre_delivered_n=state["pre_delivered_n"], seed=start["seed"],
                       index=obj["index"], history_steps=start["pre_steps"],
                       history_q=start["history_q"], samples_sha256=worker["samples_sha256"],
                       witness=state["witness"], driver_sha256=worker["driver_sha256"],
                       prompt_sha256=start["prompt_sha256"], arms=[])
            for a in arms:
                if (a["pre_state"] != state or a["pre_delivered_n"] != state["pre_delivered_n"]
                        or a["prompt_sha256"] != start["prompt_sha256"] or a["task_id"] != first["task_id"]
                        or worker["source_sha256"] != obj["source_sha256"]
                        or a["source_sha256"] != worker["source_sha256"]
                        or any(a[k] != start[k] or a[k] != obj[k] for k in ("pre_steps", "seed", "index"))):
                    raise ValueError("Mismatched actual start state")
                bill = a["bills"]
                for k in ("decoder_f", "decoder_q", "encoder_f", "encoder_q"):
                    if (bill["fork"][k]+bill["continuation"][k] != a["suffix_"+k]
                            or a["generation"][k] != a["suffix_"+k]+bill["history"][k]):
                        raise ValueError("Receipt reconstruction failed")
                if a["end_reason"] not in ("eos", "truncated", "decode_failure"):
                    raise ValueError("Keep unrecognized endpoint for separate failure adjudication")
                out["arms"].append(dict(q=a["fork_q"], state_digest=dd, final_n=a["final_n"],
                                        terminal=a["end_reason"], correct=a["quality"]["correct"],
                                        first_step_decoder_q=bill["fork"]["decoder_q"],
                                        suffix_decoder_f=a["suffix_decoder_f"], suffix_encoder_f=a["suffix_encoder_f"],
                                        suffix_decoder_q=a["suffix_decoder_q"], final_ids_sha256=a["final_ids_sha256"]))
            yield out
            start = None


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("worker", type=Path)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    with args.worker.open() as f:
        for i, row in enumerate(seals(json.loads(line) for line in f if line.endswith("\n"))):
            print(json.dumps(row, sort_keys=True, separators=(",", ":")))
            if args.limit is not None and i+1 >= args.limit:
                break
