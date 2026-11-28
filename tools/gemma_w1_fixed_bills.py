"""Recover Gemma Flow W1 decoder-history rows from the frozen GPU call events."""
import json
from hashlib import sha256
from pathlib import Path
from statistics import mean
import sys


def bill(root, manifest):
    rows, digests = [], []
    tasks = json.loads(manifest.read_bytes())["samples"]
    for i,t in enumerate(tasks):
        blob = (root/f"{i:02}-flow_w1.json").read_bytes()
        r = json.loads(blob)
        calls = r["events"]
        assert t["index"] == r["index"] == i and r["model"] == "diffusiongemma" and r["policy"] == "flow_w1"
        assert all(r[k] == t[k] for k in ("seed","dataset","task_id")) and r["status"] == "success"
        assert calls[0][0] == "encoder" and calls[0][1] > 0
        gen = calls[1:]
        decoder = [q for stage,q in gen if stage == "decoder"]
        encoder = [q for stage,q in gen if stage == "encoder"]
        assert decoder == [s["q"] for s in r["trace"]] and all(s["width"] == 1 for s in r["trace"])
        assert len(decoder) + len(encoder) == len(gen) == r["generation_forwards"]
        assert sum(decoder)+sum(encoder) == r["generation_query_rows"]
        rows.append(dict(F=len(gen),N=r["output_tokens"],decoder_Q=sum(decoder),encoder_Q=sum(encoder)))
        digests.append(sha256(blob).hexdigest())
    assert len(rows) == 90
    return dict(ordered_source_sha256=sha256("".join(digests).encode()).hexdigest(),
                **{c:dict(**{k:sum(x[k] for x in g) for k in rows[0]},
                          macro_TPF=mean(x["N"]/x["F"] for x in g))
                   for c,g in (("original30",rows[:30]),("new60",rows[30:]),("frozen90",rows))})


if __name__ == "__main__":
    print(json.dumps(bill(Path(sys.argv[1]),Path(sys.argv[2])),indent=2))
