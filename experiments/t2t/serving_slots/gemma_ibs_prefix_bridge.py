"""One slack-only prefix replay on the hard-capped, single-block Optimus IBS plan."""
from contextlib import ExitStack, contextmanager
from hashlib import sha256
from unittest.mock import patch

B = 256
CASES = (0, 10, 20, 30, 50, 70, 40, 60, 80)
TIERS = (32, 64)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def suffix(base, resolved, dirty, seen, frontier, cap):
    picked = set(base)
    require(cap in TIERS and 0 < len(base) <= cap and list(base)==sorted(picked) and
            0 <= frontier <= B and set(range(frontier)) <= resolved & seen - dirty and
            min(base)>=0 and max(base)<B, "Invalid IBS state")
    if not ((picked-resolved) & seen) or dirty-picked:
        return []
    extra = []
    for i in range(frontier-1, -1, -1):
        if (len(extra) >= min(4, cap-len(base)) or i in picked or i >= base[-1] or
                i not in resolved & seen - dirty):
            break
        extra.append(i)
    return extra[::-1]


def tensor_sha(t):
    import torch
    return sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


@contextmanager
def instrument(rt, model, unmask, cap, enabled):
    import numpy as np

    original_plan, original_ctl = rt.limited_rows, rt.controller
    original_sample = unmask.low_confidence_dynamic_sparse
    tape, seen = [], set()
    start = ctl = None
    first = False

    def controller(controls, mode, c, offset, previous):
        nonlocal start, ctl
        require(mode=="ibs" and c==cap and previous is None, "Not head-only IBS")
        ctl, width = original_ctl(controls, mode, c, offset, previous)
        require(width==B, "Wrong head width")
        start = offset
        seen.clear()
        return ctl, width

    def plan(candidate, prior, d, offer):
        nonlocal first
        require(offer==cap and start is not None, "Wrong offer")
        orig = original_plan(candidate, prior, d, cap)
        base = orig.tolist()
        dirty, resolved = set(np.flatnonzero(d).tolist()), set(np.flatnonzero(prior).tolist())
        edge = int(ctl.cached_prefix_len)
        eligible = suffix(base, resolved, dirty, seen, edge, cap)
        extra = eligible if enabled else []
        actual = sorted(base+extra)
        mask = sorted(set(base)-resolved)
        require(set(actual)-resolved==set(mask) and len(actual)<=cap and max(actual)==base[-1] and
                (not (dirty-set(base)) or not extra), "Mask, debt, span or cap changed")
        entry = dict(head=start//B,ready=edge,base=base,eligible=eligible,extra=extra,
                     actual=actual,mask=mask,dirty=sorted(dirty),proposed=len(candidate),
                     warm=bool(set(mask)&seen),decision=[],signal=None)
        if not first and eligible:
            entry["first"] = True
            first = True
        tape.append(entry)
        seen.update(actual)
        return np.array(actual,dtype=orig.dtype) if extra else orig

    def before(_, args, kw):
        e = tape[-1]
        if e.get("first"):
            base = [e["actual"].index(i) for i in e["base"]]
            ids, sc = kw["decoder_input_ids"],kw["self_conditioning_logits"]
            e["base_ids"] = tensor_sha(ids[:,base])
            e["base_sc"] = tensor_sha(sc[:,base]) if sc is not None else None

    def sample(raw, proposal, prior, sel, *, row_width):
        e = tape[-1]
        require(sel.tolist()==e["mask"] and row_width==B and raw.shape[0]==len(e["mask"]) and
                not e["decision"], "Unmeasured or extra prefix sample")
        if e.get("first"):
            e["signal"] = tensor_sha(raw)
        out = original_sample(raw,proposal,prior,sel,row_width=row_width)
        keep = out[sel].bool()
        e["decision"] = list(zip(sel[keep].tolist(),proposal[keep].tolist()))
        require(e["decision"], "MASK call made no progress")
        return out

    with ExitStack() as stack:
        stack.enter_context(patch.object(rt,"controller",controller))
        stack.enter_context(patch.object(rt,"limited_rows",plan))
        stack.enter_context(patch.object(unmask,"low_confidence_dynamic_sparse",sample))
        stack.enter_context(model.register_forward_pre_hook(before,with_kwargs=True))
        yield tape


def audit(tape, trace, events, seq, enc, prompt, eos, cap, enabled):
    require(len(tape)==len(trace) and len(trace)>0,"Empty or mismatched trace")
    ledger = [("encoder",len(prompt))]
    expected_enc = [(prompt,list(range(len(prompt))))]
    seen, dirty, tokens = set(), set(), {}
    head = -1
    extra_q = opportunities = mask_F = warm_F = 0
    delivered = []
    for j,(e,s) in enumerate(zip(tape,trace,strict=True)):
        if e["head"]!=head:
            require(e["head"]==head+1 and not dirty and (head<0 or len(tokens)==B),
                    "Unpaid/unfinished preceding head")
            head=e["head"]
            seen, tokens = set(), {}
        base, actual = e["base"],e["actual"]
        elig=suffix(base,set(tokens),dirty,seen,e["ready"],cap)
        require(e["dirty"]==sorted(dirty) and (dirty<=set(base) or (set(base)<=dirty and not e["mask"])) and
                e["eligible"]==elig and e["extra"]==(elig if enabled else []) and
                e["mask"]==sorted(set(base)-set(tokens))==sorted(set(actual)-set(tokens)) and
                actual==sorted(base+e["extra"]) and len(actual)==s["q"]<=cap and
                s["positions"]==[B*head+i for i in actual] and s["tick"]==j and s["offer"]==cap and
                s["block"]==head and s["cross"]==0 and s["proposed_q"]==e["proposed"] and
                len(base)<=s["proposed_q"] and s["visible_end"]==head*B+base[-1]+1 and
                e["warm"]==bool(set(e["mask"])&seen),"Physical plan or classification mismatch")
        seen.update(actual)
        require(set(range(actual[-1]+1))<=seen,"KV hole")
        decision=dict(e["decision"])
        require(len(decision)==len(e["decision"])==s["accepted"] and
                bool(decision)==bool(e["mask"]) and set(decision)<=set(e["mask"]) and
                s["admitted"]==[B*head+i for i in sorted(decision)],"Admission mismatch")
        dirty=(dirty-set(actual))|set(decision)
        tokens.update(decision)
        require(set(range(s["ready"])) <= set(tokens)-dirty,"Frontier crossed debt")
        ledger.append(("decoder",len(actual)))
        extra_q+=len(e["extra"])
        opportunities+=bool(elig)
        mask_F+=bool(e["mask"])
        warm_F+=e["warm"]
        if s["ready"]==B:
            require(len(tokens)==B and not dirty,"Unready retirement")
            block=[tokens[i] for i in range(B)]
            stop=next((i+1 for i,t in enumerate(block) if t in eos),None)
            delivered.extend(block[:stop])
            terminal=stop is not None or head==7
            require(terminal==(j+1==len(trace)),"Wrong EOS or max length")
            if not terminal:
                offset=len(prompt)+head*B
                expected_enc.append((block,list(range(offset,offset+B))))
                ledger.append(("encoder",B))
    require(events==ledger and enc==expected_enc and delivered==seq and trace[-1]["ready"]==B,
            "Final IDs/encoder inputs/physical bill mismatch")
    return dict(decoder_F=len(trace),decoder_Q=sum(s["q"] for s in trace),
                encoder_F=len(enc)-1,encoder_Q=B*(len(enc)-1),F=len(ledger)-1,
                peak=max(s["q"] for s in trace),heads=head+1,mask_F=mask_F,warm_F=warm_F,
                eligible_ticks=opportunities,extra_Q=extra_q,base_Q=sum(len(e["base"]) for e in tape),
                dirty_paid=sum(len(set(e["dirty"])&set(e["actual"])) for e in tape),
                admissions=sum(s["accepted"] for s in trace))


def match(a,b):
    ta,tb=a["notes"],b["notes"]
    first=next((j for j,e in enumerate(tb) if e.get("first")),None)
    if first is None:
        require(a==b,"Unexposed trajectories differ")
        return dict(first=None)
    x,y=ta[first],tb[first]
    require(ta[:first]==tb[:first] and a["trace"][:first]==b["trace"][:first] and
            all(x[k]==y[k] for k in ("head","ready","base","eligible","mask","dirty","warm",
                                      "first","base_ids","base_sc","proposed")) and
            x["extra"]==[] and y["extra"]==y["eligible"] and set(y["actual"])-set(x["actual"])==set(y["extra"]),
            "Not the registered first fork")
    diff=next((j for j in range(first,min(len(ta),len(tb))) if
               (ta[j]["head"],ta[j]["mask"],ta[j]["decision"])!=(tb[j]["head"],tb[j]["mask"],tb[j]["decision"])),None)
    return dict(first=first,first_head=x["head"],first_added=len(y["extra"]),first_base_Q=len(x["base"]),
                first_spare_rows=y["extra"],first_logits_changed=x["signal"]!=y["signal"],decision_tick=diff)
