"""Reconstruct fixed-Q512 row roles from the sealed Live v1 device traces."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path


ROLES = ("mandatory","prefix","unresolved","clean-other","verify","rebuild")


def check(ok, why):
    if not ok:
        raise ValueError(why)


def classify(tape):
    tr, ph = tape["trace"], tape["physical_trace"]
    check(ph[0]["kind"]=="encoder" and not ph[0]["paid"] and all(p["paid"] for p in ph[1:]),
          "pure prompt is the only unbilled model call")
    origin = len(ph[0]["positions"])
    dec = [p for p in ph[1:] if p["kind"]=="decoder"]
    check(len(dec)==len(tr),"unmatched decoder calls")
    resolved = defaultdict(set)
    costs, edits, phase, paid = Counter(), Counter(), Counter(), Counter()
    due = set()
    for t,(e,d) in enumerate(zip(tr,dec)):
        h = e["ordinal"]
        check(e["tick"]==d["tick"]==t and e["offer"]==512 and
              e["head_mask_before"]==256-len(resolved[h]),"unmatched physical head state")
        physical = d["positions"]
        check(physical==e["positions"]==sorted(set(physical)) and 0<len(physical)==e["decoder_Q"]<=512
              and due==set(e["dirty_before_absolute"]) and due<=set(physical),"unpaid or unknown rows")
        seen, dirty = set(), set()
        for p in e["parts"]:
            o = p["ordinal"]
            rows, debt, prefix, sample = map(set,(p["rows"],p["dirty_before"],p["prefix"],p["sampled"]))
            accept, edit, revoke = map(set,(p["admitted"],p["edited"],p["revoked"]))
            check(h<=o<=h+1 and rows<=set(range(256)) and len(rows)==len(p["rows"])
                  and (debt|prefix|sample)<=rows and not debt&prefix and prefix<=resolved[o],
                  "invalid role membership")
            check((accept|edit|revoke)<=sample and not accept&resolved[o] and
                  (edit|revoke)<=resolved[o],"unmeasured or inconsistent token change")
            stripe = {origin+256*o+j for j in rows}
            check(not seen&stripe,"row billed twice")
            seen |= stripe
            dirty.update(origin+256*o+j for j in debt)
            for j in rows:
                # Same precedence as cross_race.receipt, including mixed verify/drain.
                role=(p["phase"] if p["phase"] in ("verify","rebuild") else
                      "mandatory" if j in debt else "prefix" if j in prefix else
                      "clean-other" if j in resolved[o] else "unresolved")
                costs[role]+=1
                edits[role]+=int(j in edit)
                paid[role]+=int(j in debt)
            phase["successor_Q" if o>h else "head_Q"]+=len(rows)
            if e["route"]=="rebuild_sample":
                phase["sampled_rebuild_Q"]+=len(rows)
            resolved[o].update(accept)
            resolved[o].difference_update(revoke)
        check(seen==set(physical) and dirty==due,"parts differ from measured projection/debt")
        due=(due-seen)|set(e["written_absolute"])
        check(due==set(e["dirty_after_absolute"]), "invalid end-of-step debt")
    check(not due,"debt at end of delivered request")
    return dict(roles=costs,edit_positions=edits,dirty_positions=paid,placement=phase)


def audit(root, pair_csv, manifest):
    with pair_csv.open(newline="") as fh:
        pairs={int(r["index"]):r for r in csv.DictReader(fh)}
    m=json.loads(manifest.read_bytes())
    check(set(pairs)==set(range(90)), "expected 90 prior pairs")
    cohorts={}
    for i in range(90):
        r=json.loads((root/f"C512-single1-{i:02d}.json").read_bytes())
        raw=(root/f"C512-single1-{i:02d}.ledger.json.gz").read_bytes()
        digest=sha256(raw).hexdigest()
        check(r["schema"]=="single512-v1" and r["on"] is True and r["samples_sha256"]==m["samples_sha256"]
              and all(r[k]==m["samples"][i][k] for k in ("index","task_id","dataset","seed"))
              and r["index"]==i and digest==r["ledger_sha256"]==pairs[i]["on_ledger_sha256"]
              and r["layers"]==30 and r["projections"]==85, "not the qualified Q512 receipt")
        st=classify(json.loads(gzip.decompress(raw)))
        check(sum(st["roles"].values())==r["decoder_Q"]==int(pairs[i]["on_decoder_Q"])
              and sum(st["dirty_positions"].values())==int(pairs[i]["on_refresh_rows"]),
              "roles or total physical refresh disagree with existing audit")
        for group in ("frozen90","original30" if i<30 else "new60"):
            c=cohorts.setdefault(group,dict(n=0,roles=Counter(),edit_positions=Counter(),
                           dirty_positions=Counter(),placement=Counter(),decoder_Q=0,decoder_F=0))
            c["n"]+=1
            c["decoder_Q"]+=r["decoder_Q"]
            c["decoder_F"]+=r["decoder_F"]
            for k,bill in st.items():
                c[k].update(bill)
    for c in cohorts.values():
        for col in ("roles","edit_positions","dirty_positions"):
            c[col]={k:c[col][k] for k in ROLES}
    return dict(source_pair_csv_sha256=sha256(pair_csv.read_bytes()).hexdigest(),cohorts=cohorts)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("root","pair_csv","manifest"):
        ap.add_argument(k,type=Path)
    a=ap.parse_args()
    print(json.dumps(audit(a.root,a.pair_csv,a.manifest),indent=2,sort_keys=True))
