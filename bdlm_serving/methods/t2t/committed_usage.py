"""Observed within-request token use in the most recent 256 committed output IDs."""
from collections import Counter, deque
from math import log1p


class CommittedUsage:
    def __init__(self):
        self.tail = deque(maxlen=256)
        self.totals, self.counts = Counter(), Counter()

    @staticmethod
    def contexts(left, right):
        return (None if left is None else (0,left),
                None if right is None else (1,right),
                None if left is None or right is None else (2,left,right))

    def commit(self, ids):
        self.tail.extend(ids)
        self.totals.clear()
        self.counts.clear()
        seq = tuple(self.tail)
        for i,token in enumerate(seq):
            for key in self.contexts(seq[i-1] if i else None, seq[i+1] if i+1<len(seq) else None):
                if key is not None:
                    self.totals[key] += 1
                    self.counts[key,token] += 1

    def features(self, old, before, mask):
        rows=[]
        for i in old:
            left = before[i-1] if i else self.tail[-1] if self.tail else None
            right = before[i+1] if i+1<len(before) else None
            row=[]
            for k in self.contexts(None if left==mask else left, None if right==mask else right):
                known = self.counts[k,before[i]]
                row.extend((log1p(min(16,known)),log1p(min(16,self.totals[k]-known))))
            rows.append(row)
        return rows


def read_usage_weights(path):
    import json
    import numpy as np
    from bdlm_serving.evaluation.frozen import sha
    from .embedding_context import CALIBRATION

    if path is None:
        raise ValueError("Committed usage requires private --context-weights")
    data = json.loads(path.read_text())
    weights = np.asarray(data["weights"], dtype=np.float64)
    if (data.get("schema") != "llada-committed-usage-v1"
            or data.get("calibration_sha256") != sha(CALIBRATION)
            or weights.shape != (34,) or not np.isfinite(weights).all()):
        raise ValueError("Unexpected committed usage calibration")
    return weights


class UsageRanker:
    def __init__(self, unit, weights):
        self.unit, self.weights = unit, weights

    def scores(self, before, fresh, old, position, *, usage, mask):
        import numpy as np
        from .embedding_context import context

        pos = np.asarray(position)
        embedding = context(self.unit, old, [before[i] for i in old],
                            [(i, before[i]) for i in sorted(fresh)], pos)
        return np.column_stack((pos, embedding, usage.features(old, before, mask))) @ self.weights
