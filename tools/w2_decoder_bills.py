"""Export small cohort bills from frozen Gemma W2 GPU event tapes; no inference."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean


def bill(r, n, arm):
    assert r["model"] == "diffusiongemma" and r["status"] == "success"
    assert r["index"] == n and r["policy"] == arm and r["events"][0][0] == "encoder"
    paid = r["events"][1:]
    dec = [q for kind, q in paid if kind == "decoder"]
    enc = [q for kind, q in paid if kind == "encoder"]
    assert dec == [e["q"] for e in r["trace"]]
    assert all(e["q"] == 256*(e["active"] if arm == "active_w2" else 2) for e in r["trace"])
    assert len(enc)+len(dec) == len(paid) == r["generation_forwards"] and all(q == 256 for q in enc)
    assert sum(enc)+sum(dec) == r["generation_physical_padded_rows"]
    assert abs(r["output_tokens"]/len(paid)-r["tpf"]) < 1.e-9
    return (r["output_tokens"], len(paid), sum(dec), sum(enc), len(dec), len(enc))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prefix90", type=Path)
    root = ap.parse_args().prefix90
    assert __debug__
    source = root/"gemma-fixed512-active-w2"
    out = {}
    for arm in ("flow_w2", "active_w2"):
        seals = sha256()
        rows = []
        for i in range(90):
            name = f"{i:02}-{arm}.json"
            blob = (source/name).read_bytes()
            r = json.loads(blob)
            if arm == "flow_w2":
                assert json.loads((root/"gemma-wave-reference90"/name).read_bytes()) == r
            seals.update(sha256(blob).digest())
            rows.append(bill(r, i, arm))
        out[arm] = {"ordered_source_sha256":seals.hexdigest()}
        for k, part in (("original30", rows[:30]), ("new60",rows[30:]), ("frozen90", rows)):
            n, f, dq, eq, df, ef = map(sum, zip(*part))
            out[arm][k] = dict(requests=len(part), N=n, F=f, decoder_Q=dq, encoder_Q=eq,
                               decoder_F=df, encoder_F=ef, efficiency=n/dq, macro_TPF=mean(x[0]/x[1] for x in part))
    print(json.dumps(out, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
