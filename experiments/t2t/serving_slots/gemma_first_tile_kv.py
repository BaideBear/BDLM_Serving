"""Build, but do not consume, the very first measured Q32 tile of each block."""
from hashlib import sha256

from .gemma_joint_verify_eval import check

B, Q = 256, 32
FRONT = set(range(Q))
FULL = set(range(B))


def build_only(enabled, seen, sc, accepted, dirty, age, positions, sample, route):
    check(sc <= seen <= FULL and accepted <= sc and dirty <= seen, "Inconsistent physical/sampler state")
    if seen:
        return False
    check(not (sc or accepted or dirty or age) and positions == sample == FRONT
          and route == "sparse", "Unexpected block opening")
    return enabled


def h(t):
    return sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def arm_class(base_class, enabled, tape):
    class FirstTileKV(base_class):
        def _project(self, states, pos):
            logits=super()._project(states,pos)
            if not self.tick:
                import torch
                self.first_logit_sha=h(logits.view(torch.uint8))
            return logits

        def _apply(self, states, pos, sample, prefix, cold, offer, route):
            head, = states
            check(offer == self.fixed_q == Q and self.tail is None and not self.rebuild and self.early
                  and self.top_gate == "off" and self.fuse_first_verify and self.law.num == 1
                  and self.law.den == 8 and not self.reuse_paid_masks, "Wrong reference configuration")
            rows,s = head.rows,head.sampler
            origin=self.origin+B*head.ordinal
            projected={p-origin for p in pos}
            sampled={p-origin for p in sample}
            seen,sc,acc,debt=map(set,(rows.initialized,s.sc,rows.accepted,rows.dirty))
            check(debt <= projected and len(projected)<=Q and acc <= sc, "Illegal physical plan")
            if route in ("verify", "verify_paid"):
                check(sc == seen == FULL, "Unsampled row at terminal verification")
            skip=build_only(enabled,seen,sc,acc,debt,s.age,projected,sampled,route)
            first=not seen
            can_sample=head.can_sample
            if first:
                check(not prefix and set(cold)==set(pos), "Wrong first cold tile")
                canvas=h(s.ids)
                rng=h(s.rng.get_state())
            if skip:
                sample,route=set(),"first_tile_kv"
            # This calls the existing full physical projector, including LM head.
            # Direct consumption of its logits is disabled only at the build.
            e=super()._apply(states,pos,sample,prefix,cold,offer,route)
            p,=e["parts"]
            sampled=set(p["sampled"])
            after_sc=set(s.sc)
            check(after_sc==sc|sampled and sampled<=projected and rows.accepted<=after_sc and
                  p["age_after"]==p["age_before"]+bool(sampled),"Unexpected sampler state change")
            if first:
                check(rows.initialized==FRONT and set(self.bank.live)==set(pos) and
                      len(self.bank.visited)==len(self.model.model.decoder.layers)==30 and
                      all(set(pos)<=written for written in self.bank._written.values()) and
                      len(self.bank._written)==30,"Incomplete first-tile KV writes")
                if skip:
                    check(not(sampled or after_sc or rows.dirty or rows.accepted or s.age) and
                          canvas==h(s.ids) and rng==h(s.rng.get_state()), "Build consumed RNG, made writes, or advanced")
            delta=sampled-sc
            phase=("verify" if route in ("verify","verify_paid") else "cold" if seen!=FULL
                   else "warm" if can_sample else "close")
            tape.append(dict(tick=e["tick"],head=head.ordinal,phase=phase,route=route,q=len(pos),
                        first=first,canvas=canvas if first else None,rng=rng if first else None,
                        first_logits=self.first_logit_sha if not e["tick"] else None,
                        seen=len(seen),sc=len(sc),first_samples=sorted(delta),
                        kv_only_revisits=sorted(delta & seen),writes=len(set(p["input_changed"])|
                        set(p["admitted"])|set(p["revoked"])),debt=len(debt),
                        sampled=len(sampled),age=p["age_before"],accepted=len(acc),
                        first_front_admits=len(delta & FRONT & set(p["admitted"]))))
            return e
    return FirstTileKV


def audit(x,t,enabled):
    check(len(t)==x["decoder_F"] and sum(e["q"] for e in t)==x["decoder_Q"] and
          x["F"]==len(t)+x["encoder_F"] and x["encoder_Q"]==B*x["encoder_F"],"Wrong bill")
    heads=x["encoder_F"]+1
    g=dict(heads=heads,build_F=0,first_sampled=0,first_writes=0,first_tile_firstvisit_admits=0,
           deferred_samples=0,deferred_age_sum=0,sample_ticks=0,dirty_Q=0,writes=0,peak=max(e["q"] for e in t))
    check(g["peak"]==Q and [e["tick"] for e in t]==list(range(len(t))) and
          {e["head"] for e in t}==set(range(heads)), "Missing block or bad width")
    for j in range(heads):
        b=[e for e in t if e["head"]==j]
        first=b[0]
        check(first["first"] and sum(e["first"] for e in b)==1 and first["q"]==Q and
              first["sampled"]==(0 if enabled else Q) and first["age"]==0 and
              first["route"]==("first_tile_kv" if enabled else "sparse") and
              b[-1]["phase"]=="verify", "Bad opening or close")
        coverage=set()
        deferred=[]
        age=0
        for e in b:
            fresh=set(e["first_samples"])
            check(e["age"]==age and e["sc"]==len(coverage) and not fresh & coverage
                  and e["debt"]<=e["q"] and e["writes"]<=e["sampled"]<=e["q"],"Coverage, writes or clock drift")
            deferred.extend(e["kv_only_revisits"])
            if e["kv_only_revisits"]:
                check(enabled and e["seen"]==B and e["phase"]=="warm", "Premature or fabricated warm sample")
                g["deferred_age_sum"]+=e["age"]*len(e["kv_only_revisits"])
            coverage.update(fresh)
            age+=bool(e["sampled"])
        check(coverage==FULL and sorted(deferred)==(sorted(FRONT) if enabled else []) and age<=48,
              "Missing real first samples or hidden starvation")
        g["build_F"]+=first["route"]=="first_tile_kv"
        g["first_sampled"]+=first["sampled"]
        g["first_writes"]+=first["writes"]
        g["deferred_samples"]+=len(deferred)
    for k,src in (("dirty_Q","debt"),("writes","writes"),("first_tile_firstvisit_admits","first_front_admits")):
        g[k]=sum(e[src] for e in t)
    g["sample_ticks"]=sum(bool(e["sampled"]) for e in t)
    for phase in ("cold","warm","close","verify"):
        p=[e for e in t if e["phase"]==phase]
        g[phase+"_F"],g[phase+"_Q"]=len(p),sum(e["q"] for e in p)
    check(sum(g[k+"_F"] for k in ("cold","warm","close","verify"))==len(t),"Unclassified call")
    return g
