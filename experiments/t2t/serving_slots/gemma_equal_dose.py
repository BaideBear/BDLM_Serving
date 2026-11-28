"""Registered six actual A3 Q128 rollouts; only the placement of extra admits varies."""
import json
from hashlib import sha256

from bdlm_serving.methods.t2t.confidence_dose import ConfidenceDose
from . import gemma_paid_agreement_eval as ev
from .gemma_ibs_certified_eos import B, require
from .gemma_paid_agreement import Agreement


class Current(Agreement):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.ledger = ConfidenceDose(floor=kw["floor"])


def counted(arm, excluded, is_current):
    prev, block, tick = {}, -1, -2
    decisions = {}
    for o in arm["agreement"]["observations"]:
        by_pos = {}
        for pos, token, prob, take in zip(o["positions"], o["proposals"],
                                          o["probabilities"], o["native"], strict=True):
            if not take and token not in excluded and prob > .5:
                by_pos[pos] = (token, prob)
        repeat = sorted(p for p,(v, _) in by_pos.items() if o["block"] == block
                        and o["tick"] == tick+1 and prev.get(p) == v)
        current = sorted(sorted(by_pos, key=lambda p: (-by_pos[p][1], p))[:len(repeat)])
        actual = current if is_current else repeat
        require(actual == sorted(o["extra"]), "Wrong row/dose on this trajectory")
        t = arm["trace"][o["tick"]]
        base = {p for p,k in zip(o["positions"],o["native"],strict=True) if k}
        require(set(t["admitted"]) == base.union(actual), "Real writes differ from policy")
        prev, block, tick = {p: v for p,(v, _) in by_pos.items()}, o["block"], o["tick"]
        decisions[tick] = (o, repeat, current)
    return decisions


def compact(arm):
    return dict(N=arm["N"], correct=arm["correct"], scoring=arm["scoring"], bill=arm["bill"],
                delivered_sha256=sha256(json.dumps(arm["ids"],separators=(",", ":")).encode()).hexdigest())


def compare(arms, rt, torch, model, data, proc, controls, unmask, task, scoring, special):
    cur = ev.run_arm(rt, torch, model, data, proc, controls, unmask, task, "apply",
                     scoring, special, agreement_type=Current)
    rep = arms["apply"]
    ra, ca = counted(rep, special, False), counted(cur, special, True)
    fork = None
    for tick in range(min(len(rep["trace"]),len(cur["trace"]))):
        tr, tc, dr, dc = rep["trace"][tick], cur["trace"][tick], rep["decoder"][tick], cur["decoder"][tick]
        require((tick in ra) == (tick in ca), "Samplers disagreed on a common input")
        if tick in ra:
            o1, r1, c1 = ra[tick]
            o2, r2, c2 = ca[tick]
            require({k:v for k,v in o1.items() if k != "extra"} ==
                    {k:v for k,v in o2.items() if k != "extra"} and (r1,c1)==(r2,c2),
                    "Different observations on the first common path")
            if r1 != c1:
                require(all(tr[k]==tc[k] for k in ("positions","q","offer","block")) and
                        all(dr[k]==dc[k] for k in ("ids","pos","rng")) and
                        all(dr["sample"][k]==dc["sample"][k] for k in ("rows","proposal")) and
                        [o1["block"]*B+i for i in dr["sample"]["rows"]] == o1["positions"],
                        "Unmatched physical inputs on the fork")
                fork = dict(tick=tick, block=o1["block"], k=len(r1), repeat=r1, current=c1,
                            measured=o1, input_rng=dr["rng"], paid_Q=tr["q"])
                break
        require(tr==tc and dr==dc and rep["states"][tick]==cur["states"][tick],
                "Trajectories diverged before placement changed")
    if fork is None:
        require(ev.stable(rep)==ev.stable(cur), "Unexposed arm drifted")
    return dict(arms={"stock":compact(arms["stock"]), "repeat":compact(rep), "current":compact(cur)},
                fork=fork, doses={k:[(t,len(v[1])) for t,v in d.items() if v[1]]
                                  for k,d in (("repeat",ra),("current",ca))})


if __name__ == "__main__":
    ev.main(compare)
