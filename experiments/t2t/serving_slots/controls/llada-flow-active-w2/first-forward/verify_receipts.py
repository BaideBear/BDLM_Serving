"""CPU-only check of the remote diagnostic receipts; export only row differences."""
import hashlib
import json
from pathlib import Path
import sys


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def export(d):
    rows = list(map(json.loads,(d/"compact.ndjson").read_text().splitlines()))
    done = json.loads((d/"complete.json").read_bytes())
    if (done["receipt_sha256"]!=digest(rows) or
        [r["index"] for r in rows]!=done["cases"] or done["cases"]!=[7,13,25,33,55,71]):
        raise ValueError("diagnostic identity or digest")
    output = []
    for s in rows:
        r=json.loads((d/f'{s["index"]:02d}-rows.json').read_bytes())
        if digest(r)!=s["row_receipt_sha256"] or r["layer_q"]!=[[s["prefill_Q"],64,64,64,32]]*20:
            raise ValueError("changed raw evidence or physical bill")
        for key,c in r["comparisons"].items():
            rs=c["rows"]
            if len(rs)!=32 or [x["row"] for x in rs]!=list(range(32)):
                raise ValueError("head coverage")
            if all(x["bit_equal"] for x in rs)!=s["comparisons"][key]["equal"]:
                raise ValueError("bitwise row coverage")
        output.append(dict(index=r["index"],prompt_rows=r["inputs"]["prompt_rows"],
                           head=r["inputs"]["head"],embed_dtype=r["inputs"]["embed_dtype"],
                           logit_dtype=r["outputs"]["dense"]["logit_dtype"],
                           row_deltas={k:v["rows"] for k,v in r["comparisons"].items()}))
    return output


if __name__=="__main__":
    print(json.dumps(export(Path(sys.argv[1])),sort_keys=True,separators=(",",":")))
