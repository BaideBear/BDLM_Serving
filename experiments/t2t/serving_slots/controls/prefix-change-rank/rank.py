"""Causal row versions; no token proposals, logits or model dependency."""
from math import lcm

B, Q, MASK, EOS = 32, 8, 156895, 156892
SCALE = lcm(*range(1, B+1))


class Versions:
    def __init__(self):
        self.paid = {}
        self.changed = {}

    def plan(self, pool, k, start, end):
        candidates = sorted(pool)
        if not 0 <= k <= min(2,len(candidates)) or len(set(candidates))!=len(candidates):
            raise ValueError("invalid quota or duplicate")
        def deps(i):
            return [j for j in range(start,end) if self.changed.get(j,-1)>self.paid[i]]
        facts = [(i,self.paid[i],deps(i)) for i in candidates]
        scores = {i:sum(SCALE//(1+abs(i-j)) for j in js) for i,_,js in facts}
        ranked = sorted(candidates,key=lambda i:(scores[i],i),reverse=True)
        return dict(right=candidates[-k:] if k else [], change=sorted(ranked[:k]),
                    candidates=facts, scores=scores)

    def project(self, tick, rows):
        self.paid.update((r,2*tick) for r in rows)

    def write(self, tick, rows):
        self.changed.update((r,2*tick+1) for r in rows)
