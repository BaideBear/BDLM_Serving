"""Preregistered Q16 paid-agreement risk cap: archived replay and narrow adapter."""
import argparse
import json
import os
from pathlib import Path
from types import FunctionType
from unittest.mock import patch

from tools import llada_ibs_eos_q16_agreement as hook, llada_ibs_eos_q16_eval as base
from tools.llada_ibs_eos_q16_raw_audit import verify
from tools.llada_ibs_eos_raw_audit import EOS, MASK, replay

ARMS = ("stock", "observer", "apply", "capped")
PRIOR = base.BASE/"results/llada-ibs-q16-paid-agreement-h0-v1"
ALLOWANCE = .5
OLD_HOOK = "f66f253604d859c07f8a38817b821e630a2b8f859f5d1d9d45a7240406e6da1f"
OLD_AUDIT = "1d1e61371e7931242eccc5ef14475967f5598e53e07be05f280ef538fcc60da6"
FIELDS = ("block","tick","positions","proposals","probabilities","native")


def trace(tape, excluded, *, expect_capped=None, cls=None):
    """Independent oracle over current and immediately previous physical observations."""
    prev, votes = None, {}
    impl = cls() if cls else None
    for t in tape:
        tag = (t["block"], t["tick"])
        base.require(prev is None or tag[1]>prev[1], "Nonphysical tick order")
        current = {p:(v,prob) for p,v,prob,n in zip(*(t[k] for k in FIELDS[2:]),strict=True)
                   if not n and prob>.5 and v not in excluded}
        risk = sorted((1-min(prob,votes[p][1]),p) for p,(v,prob) in current.items()
                      if prev==(tag[0],tag[1]-1) and p in votes and votes[p][0]==v)
        repeated = {p for _,p in risk}
        full = tuple(p for p in current if p in repeated)
        chosen, spend = set(), 0.
        for s,p in risk:
            if spend+s>ALLOWANCE:
                break
            spend += s
            chosen.add(p)
        kept = tuple(p for p in full if p in chosen)
        if expect_capped is not None:
            base.require(tuple(t["eligible"])==(kept if expect_capped else full),
                         "Capped or unconditional eligibility differs from independent oracle")
        if impl is not None:
            actual = impl.observe(**{k:t[k] for k in FIELDS},excluded_ids=excluded)
            base.require(tuple(actual)==kept,"Shared ledger differs from frozen risk oracle")
        yield t, full, kept, risk, spend
        prev, votes = tag, current


def replay_exposure(cls=None):
    base.require(base.file_sha(PRIOR/"raw-audit.json")==OLD_AUDIT and
                 verify(PRIOR)==base.read(PRIOR/"raw-audit.json"),"Invalid historical paid-IBS archive")
    excluded=set(base.read(PRIOR/"registration.json")["excluded"])
    panel=[]
    for i in base.SIX:
        raw=base.read(PRIOR/"pairs"/f"{i:02}.json")
        arms={}
        for arm in ("tracked","apply"):
            points=[]
            for t,full,kept,risk,spend in trace(raw["tapes"][arm]["ticks"],excluded,expect_capped=False,cls=cls):
                rejected=sorted(set(full)-set(kept))
                if rejected:
                    points.append(dict(block=t["block"],tick=t["tick"],kept=list(kept),refused=rejected,
                                       risk=[list(x) for x in risk],spent=spend))
            arms[arm]=points
        panel.append(dict(index=i,arms=arms))
    return dict(panel=panel,source=OLD_AUDIT,apply_exposed=sum(bool(x["arms"]["apply"]) for x in panel),
                refused={a:sum(len(p["refused"]) for x in panel for p in x["arms"][a]) for a in ("tracked","apply")})


def install(fixed,arm,tape,excluded,cls):
    base.require(arm in ARMS and base.file_sha(Path(hook.__file__))==OLD_HOOK,"Wrong arm or changed Q16 adapter")
    if arm=="capped":
        # Rebind only the ledger factory for this request, without a global patch.
        install_capped=FunctionType(hook.install_run.__code__,dict(vars(hook),EvidenceLedger=cls))
        return install_capped(fixed,"apply",tape,excluded)
    return hook.install_run(fixed,"tracked" if arm=="observer" else arm,tape,excluded)


def audit(row,tape,arm,*,eos,mask,excluded):
    if arm!="capped":
        return base.audit(row,tape,"tracked" if arm=="observer" else arm,eos=eos,mask=mask,excluded=excluded)

    def check_capped(ticks, excluded_ids):
        for t,_,keep,_,_ in trace(ticks,excluded_ids,expect_capped=True):
            yield t,keep
    audit_capped=FunctionType(base.audit.__code__,dict(vars(base),independent_eligible=check_capped))
    return audit_capped(row,tape,"apply",eos=eos,mask=mask,excluded=excluded)


def first_removed(full,capped,a,b,excluded):
    pt=zip(trace(a["ticks"],excluded,expect_capped=False),trace(b["ticks"],excluded,expect_capped=True))
    first=None
    for (x,all_extras,kept,risk,spend),(y,_,capped_kept,*_) in pt:
        base.require(all(x[k]==y[k] for k in FIELDS),"Tape disagrees before any refusal")
        base.require(kept==capped_kept,"Capped eligibility changed on identical measured history")
        refused=sorted(set(all_extras)-set(kept))
        if refused:
            first=(x["tick"],refused,kept,risk,spend)
            break
    if first is None:
        base.require(full==capped and a["writes"]==b["writes"] and a["certificate"]==b["certificate"] and
                     len(a["ticks"])==len(b["ticks"]),"No refusal but policy changed work")
        return None
    tick,refused,keep,risk,spend=first
    i=tick+int(full["events"][0]["phase"]=="prefill")
    base.require(full["prompt_ids"]==capped["prompt_ids"] and full["events"][:i]==capped["events"][:i] and
                 all(full[k][:i+1]==capped[k][:i+1] for k in ("calls","forward_inputs")) and
                 a["writes"][:tick]==b["writes"][:tick],"Paid history diverged before refusal")
    l,r=full["events"][i],capped["events"][i]
    allowed={"written_absolute","accepted","cached_prefix_after"}
    base.require({k:v for k,v in l.items() if k not in allowed}=={k:v for k,v in r.items() if k not in allowed} and
                 l["accepted"]-r["accepted"]==len(refused),"Refusal changed native writes or physical rows")
    lw,rw=dict(a["writes"][tick]),dict(b["writes"][tick])
    base.require(set(lw)-set(rw)==set(refused) and rw=={p:v for p,v in lw.items() if p not in refused} and
                 set(l["written_absolute"])-set(r["written_absolute"])==set(refused),
                 "Same-tick write fork is not exactly removed extras")
    return dict(tick=tick,block=l["block"],selected=l["absolute_positions"],native_t2t=l["edited_positions"],
                incoming_dirty=l["dirty_before"],removed=refused,kept=keep,risk=risk,spent=spend)


def preflight(panel_path, exposure_path=None):
    from bdlm_serving.methods.t2t import capped_agreement, paid_agreement

    p, hist = base.read(panel_path), base.read(PRIOR/"registration.json")
    original = hist["panel"]
    base.require(p["arms"] == list(ARMS) and p["cases"] == list(base.SIX) and
                 (p["uuid"],p["q"],p["cap"],p["allowance"],p["floor"]) ==
                 (base.UUID,base.Q,base.CAP,ALLOWANCE,.5) and p["archive_sha256"] == OLD_AUDIT and
                 p["capped_sha256"] == base.file_sha(Path(capped_agreement.__file__)) and
                 original["evidence_sha"] == base.file_sha(Path(paid_agreement.__file__)) and
                 hist["panel_sha"] == base.file_sha(panel_path.with_name("q16-agreement-registration.json")) and
                 original == base.read(panel_path.with_name("q16-agreement-registration.json")) and
                 base.file_sha(Path(hook.__file__)) == OLD_HOOK and
                 original["samples_sha256"] == base.file_sha(base.DATA/"samples.json") and
                 original["ledger_sha256"] == base.file_sha(base.DATA/"fixed-family-reference-case-ledger-v1.json") and
                 all(base.file_sha(base.CONTROLS/k) == v for k,v in original["opt_controls"].items()),
                 "Changed preregistered inputs or controls")
    cls = capped_agreement.CappedEvidenceLedger
    exposure = replay_exposure(cls)
    base.require(exposure["apply_exposed"] == 5 and exposure["refused"] == {"tracked":106,"apply":77},
                 "Changed frozen exposure")
    if exposure_path is not None:
        base.require(exposure == base.read(exposure_path),"Offline exposure sidecar changed")
    return cls, hist, exposure


def compare(index, raw, excluded):
    rows, tapes = raw["arms"], raw["tapes"]
    old = base.read(PRIOR/"pairs"/f"{index:02}.json")
    for now, before in (("stock","stock"),("observer","tracked"),("apply","apply")):
        base.require(base.compact([rows[now],tapes[now]]) ==
                     base.compact([old["arms"][before],old["tapes"][before]]),
                     f"{index}: {now} did not reproduce complete archived tape and model ledger")
    base.require(rows["stock"] == rows["observer"] and all(tapes["stock"][k] == tapes["observer"][k]
                 for k in ("certificate","writes")), f"{index}: observer changed paid stock")
    bills = {}
    for arm in ARMS:
        row,tape = rows[arm],tapes[arm]
        bills[arm] = audit(row,tape,arm,eos=EOS,mask=MASK,excluded=excluded)
        found = replay(row,tape,q=base.Q)
        base.require((dict(found,debt_after=[]) if found else None) == tape["certificate"],
                     f"{index}: {arm} reconstructed certificate differs")
    return bills, first_removed(rows["apply"],rows["capped"],tapes["apply"],tapes["capped"],excluded)


def expected_fork(first, exposure, j):
    expected = exposure["panel"][j]["arms"]["apply"]
    return (first is None) == (not expected) and (first is None or
           (first["tick"],first["removed"]) == (expected[0]["tick"],expected[0]["refused"]))


def summary(notes):
    totals = {}
    for a in ARMS:
        m = [n["arms"][a] for n in notes]
        s = {k:sum(v[k] for v in m) for k in ("F","Q","N","extra","eligible","paid_D","paid_Q","correct")}
        totals[a] = dict(**s,pooled_tpf=s["N"]/s["F"] if s["F"] else None,
                         pooled_te=s["N"]/s["Q"] if s["Q"] else None,
                         failures=[n["index"] for n in notes if n["arms"][a]["failure"]])
    return dict(indices=list(base.SIX),totals=totals,first_forks=sum(n["first"] is not None for n in notes),
                exact_ids_vs_apply=[n["index"] for n in notes if n["arms"]["apply"]["ids_sha"] ==
                                    n["arms"]["capped"]["ids_sha"]],
                quality_flips_vs_apply=[[n["index"],n["arms"]["apply"]["correct"],n["arms"]["capped"]["correct"]]
                                        for n in notes if n["arms"]["apply"]["correct"] !=
                                        n["arms"]["capped"]["correct"]])


def run(a, cls, archived, exposure):
    import torch
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.methods.optimus import fixed, upstream
    from experiments.t2t.serving_slots import ibs_prefix_pair as pair, shared_llada_worker as worker

    base.require(a.out and os.environ["CUDA_VISIBLE_DEVICES"] == base.UUID and torch.cuda.device_count() == 1 and
                 str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-") == base.UUID[4:],
                 "Only registered H0 may run")
    h = worker.load(base.CKPT)
    tok,model = h["tok"],h["model"]
    excluded = set(tok.all_special_ids) | {EOS,MASK} | {i for i,t in tok.added_tokens_decoder.items() if t.special}
    excluded |= {t["id"] for t in base.read(base.CKPT/"tokenizer.json")["added_tokens"] if t["special"]}
    base.require((h["eos_id"],h["mask_id"]) == (EOS,MASK) and archived["excluded"] == sorted(excluded) and
                 archived["checkpoint"] == h["fingerprint"] and archived["scorer"] == h["scorer"] and
                 archived["torch"] == str(torch.__version__), "Model, scorer, tokenizer or Torch drift")
    tasks = load_tasks(base.DATA/"samples.json")
    reg = dict(panel=base.read(a.panel),panel_sha=base.file_sha(a.panel),archive=OLD_AUDIT,
               exposure_sha=base.digest(exposure),source=os.environ["BDLM_Q16_RISK_SOURCE"],
               pid=os.getpid(),uid=os.getuid(),ticks=int(Path("/proc/self/stat").read_text().split(") ")[1].split()[19]),
               boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),checkpoint=h["fingerprint"],
               weights=h["weights"],torch=archived["torch"],scorer=h["scorer"],excluded=sorted(excluded))
    a.out.joinpath("pairs").mkdir()
    base.save(a.out/"registration.json",reg)
    reg_id = base.digest(reg)
    notes = []

    def controls(name):
        base.require(name == "Optimus","Unexpected control name")
        return base.CONTROLS

    with patch.object(pair,"Q",base.Q),patch.dict(pair.LABELS,{0:base.LABEL}), \
            patch.object(upstream,"control_directory",controls):
        for j,i in enumerate(base.SIX):
            task = tasks[i]
            hist = base.read(base.DATA/f"llada21-optimus-hardcap-endpoints/{i:02}-{base.LABEL}.json")
            base.require(all(hist[k] == task[k] for k in base.KEYS),"Task or seed drift")
            prompt,_ = worker.canonical_prompt(tok,task,hist["prompt_ids"])
            rows,tapes = {},{}
            order = ARMS[j%4:] + ARMS[:j%4]
            for arm in order:
                tape = tapes[arm] = dict(certificate=None,ticks=[],writes=[])
                torch.manual_seed(task["seed"])
                with patch.object(fixed.FixedOptimus,"run",install(fixed,arm,tape,excluded,cls)):
                    rows[arm] = pair.generate(model,tok,task,prompt,base.CAP,0)
            path = a.out/"pairs"/f"{i:02}.json"
            raw = dict(arms=rows,tapes=tapes,reg=reg_id,order=order)
            base.save(path,raw)
            bills,first = compare(i,raw,excluded)
            base.require(expected_fork(first,exposure,j),f"{i}: actual first fork differs from frozen replay")
            for arm,row in rows.items():
                bills[arm]["correct"] = row["status"] == "success" and bool(scoring.score(row["output_text"],task)["correct"])
            stamp = dict(index=i,reg=reg_id,arms=bills,first=first,raw_sha=base.file_sha(path),order=order)
            notes.append(stamp)
            print(base.compact(stamp),flush=True)
    base.save(a.out/"complete-six.json",dict(reg=reg_id,**summary(notes)))


def verify_run(out,exposure):
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen90 import load_tasks

    reg = base.read(out/"registration.json")
    tasks = load_tasks(base.DATA/"samples.json")
    notes = [json.loads(line) for line in (out/"receipts.ndjson").read_text().splitlines()]
    base.require(tuple(n["index"] for n in notes) == base.SIX,"Incomplete or wrong panel")
    reg_id = base.digest(reg)
    base.require(reg["exposure_sha"] == base.digest(exposure) and
                 reg["archive"] == OLD_AUDIT and reg["source"] == os.environ["BDLM_Q16_RISK_SOURCE"],
                 "Run source or exposure differs")
    for j,n in enumerate(notes):
        i = n["index"]
        path = out/"pairs"/f"{i:02}.json"
        raw = base.read(path)
        base.require(base.file_sha(path) == n["raw_sha"] and raw["reg"] == n["reg"] == reg_id and
                     raw["order"] == n["order"] == list(ARMS[j%4:]+ARMS[:j%4]),f"{i}: raw identity differs")
        bills,first = compare(i,raw,set(reg["excluded"]))
        for arm,row in raw["arms"].items():
            bills[arm]["correct"] = row["status"] == "success" and bool(scoring.score(row["output_text"],tasks[i])["correct"])
        base.require(bills == n["arms"] and base.compact(first) == base.compact(n["first"]) and
                     expected_fork(first,exposure,j),f"{i}: recomputed bill or first fork differs")
    complete = base.read(out/"complete-six.json")
    base.require(complete == dict(reg=reg_id,**summary(notes)),"Recomputed summary differs")
    worker = (out/"worker.identity").read_text().split()
    base.require(worker == [str(reg[k]) for k in ("pid","ticks","uid")],"Physical worker identity differs")
    return dict(status="pass",summary=complete,sha256={k:base.file_sha(out/k) for k in
                ("registration.json","complete-six.json","receipts.ndjson","worker.identity","returned.identity")})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("check","run","verify"))
    p.add_argument("--panel",type=Path,required=True)
    p.add_argument("--exposure",type=Path)
    p.add_argument("--out",type=Path)
    a = p.parse_args()
    cls,hist,exposure = preflight(a.panel,a.exposure)
    if a.mode == "check":
        from bdlm_serving.methods.optimus import fixed

        for arm in ARMS:
            install(fixed,arm,{},set(),cls)
        print(base.compact(exposure))
    elif a.mode == "verify":
        print(base.compact(verify_run(a.out,exposure)))
    else:
        base.require(a.exposure is not None,"Live run needs sealed offline exposure")
        run(a,cls,hist,exposure)


if __name__ == "__main__":
    main()
