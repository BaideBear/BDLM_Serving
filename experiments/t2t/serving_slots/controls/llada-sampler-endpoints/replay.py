"""Summarize existing slot receipts. CPU only; no generated text or IDs emitted."""
import gzip
import json
from collections import Counter
from pathlib import Path

root = Path("/data1/project_data/gx-erqw-20261123/results")
flow = root / "llada-flow-active-w2-h4-v1"
with gzip.open(root / "llada-ibs4-frontier-swap-a2-partial19-v1/attempt-02/07.json.gz") as f:
    pkt = json.load(f)
mask = pkt["mask"]
events = [e for e in pkt["raw"]["bare"]["events"] if e["phase"] != "prefill"]
print("IBS7", [(e["absolute_positions"], [p for p, m in zip(e["absolute_positions"], e["selected_masked"]) if m],
                e["written_absolute"]) for e in events[:2]])
counts, exposed = Counter(), set()
for i in range(90):
    r = json.loads((flow / f"{i:02}-active_w2.json").read_bytes())
    for t, e in enumerate(r["ticks"]):
        counts["ticks"] += 1
        hit = False
        for j, s in enumerate(e["slots"]):
            h = e["head"] + 32 * j
            masks = [k for k, u in enumerate(s["ids"]) if u == mask and not s["prompt"][k] and not s["fin0"]]
            writes = [k for k in masks if s["new"][k] != mask]
            edits = [k for k, (u, v) in enumerate(zip(s["ids"], s["new"])) if mask not in (u, v)
                     and u != v and not s["prompt"][k] and not s["fin0"]]
            if i == 7 and t == j == 0:
                o = json.loads((flow / "first-forward/07-rows.json").read_bytes())["outputs"]["compact"]
                conf = o["confidence"]
                above = [k for k in masks if conf[k] > .5]
                high = [k for k in masks if conf[k] >= .9]
                assert above == writes and high and above == [k for k, b in enumerate(o["accept"]) if b]
                print("compact7", [h+k for k in masks], [h+k for k in above], [h+k for k in high],
                      {h+k: conf[k] for k in above if k not in high})
            if (i, t) == (1, 6):
                print("compact1", j, [h+k for k in masks], [h+k for k in writes], [h+k for k in edits], s["frontier"])
            if s["frontier"] == 0 and len(writes) == 1 and edits:
                counts["eligible_slots"] += 1
                hit = True
        if hit:
            counts["eligible_ticks"] += 1
            counts[f"open{e['open']}"] += 1
            exposed.add(i)
print(dict(counts), "requests", len(exposed), "original", sum(i < 30 for i in exposed), "new", sum(i >= 30 for i in exposed))
