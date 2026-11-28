"""Re-execute the sealed v1 audit on its sealed #07 bytes; record its first rejection."""
import importlib.util
import sys
from pathlib import Path

from replay import D, filehash, json, need, read


def main():
    prior=D.parent/"llada-ibs-q8-prefix-change-rank-h4-six-v1"
    src=prior/"replay.py"
    need(filehash(src)=="3baac24439cdd45d80059ef50eefb9e9a57d361aa70fc29bea9bdba76aa0a8df","v1 audit")
    spec=importlib.util.spec_from_file_location("sealed_v1_auditor",src)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    raw=prior/"07-none.json"
    need(filehash(raw)=="4c9aaab631241e268aa5b1e0b5e8f90af603da24f8967b18f5df2bdd82ac613a","v1 raw")
    r=read(raw)
    failures=[]
    def trace(frame,event,arg):
        if frame.f_code is mod.audit.__code__ and event=="exception" and isinstance(arg[1],ValueError):
            l=frame.f_locals
            n=l["n"]
            actual=r["tape"][n]
            expect=json.loads(json.dumps(l["plan"]))
            failures.append(dict(tick=n,location=frame.f_lineno,error=str(arg[1]),base=l["base"],
                live_paid=actual["paid"],slack=8-len(l["base"]),
                same_mask=actual["mask"]==l["e"]["selected_masked"],same_tick=actual["tick"]==n,
                expected_candidates=expect["candidates"],live_candidates=actual["plan"]["candidates"],
                differing=sorted(k for k in expect if expect[k]!=actual["plan"][k]),
                expected_picks=[expect["right"],expect["change"]],
                actual_picks=[actual["plan"]["right"],actual["plan"]["change"]]))
        return trace
    sys.settrace(trace)
    try:
        mod.audit(r["row"],"none",live=True,tape=r["tape"])
    except ValueError as e:
        need(str(e)=="live version/role tape","unexpected original error")
    finally:
        sys.settrace(None)
    f,=failures
    need(f["tick"]==2 and len(f["expected_candidates"])==3 and f["live_candidates"]==[] and
         f["differing"]==["candidates","scores"] and f["base"]==f["live_paid"] and
         f["same_tick"] and f["same_mask"] and f["expected_picks"]==f["actual_picks"]==[[],[]],
         "different v1 failure")
    return dict(raw_sha=filehash(raw),audit_sha=filehash(src),first_v1_failure=f)


if __name__=="__main__":
    print(json.dumps(main(),sort_keys=True))
